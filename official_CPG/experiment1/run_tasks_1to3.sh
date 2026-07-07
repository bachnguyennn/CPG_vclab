#!/bin/bash
# Tasks 1-3 reproduction on Apple Silicon (MPS):
#   Phase A: real 100-epoch baselines (success criterion for CPG)
#   Phase B: CPG finetune -> gradual prune -> pick/retrain (per paper settings)
#   Phase C: inference from final checkpoint on all 3 tasks (zero-forgetting check)
# Exit codes from CPG_cifar100_main_normal.py are control flow (2=grow, 6=prune goal met) — no set -e.

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate cpg
export PYTORCH_ENABLE_MPS_FALLBACK=1

NUM_TASKS=3

dataset=(
    'None'                # dummy
    'aquatic_mammals'
    'fish'
    'flowers'
)

echo "=== Phase A: baselines (100 epochs x $NUM_TASKS tasks) === $(date)"

BASELINE_ARCH='vgg16_bn_cifar100'
for TASK_ID in `seq 1 $NUM_TASKS`; do
    echo "--- baseline ${dataset[TASK_ID]} $(date)"
    python packnet_cifar100_main_normal.py \
        --arch $BASELINE_ARCH \
        --dataset ${dataset[TASK_ID]} --num_classes 5 \
        --lr 1e-2 \
        --weight_decay 4e-5 \
        --save_folder checkpoints/baseline/experiment1/$BASELINE_ARCH/${dataset[TASK_ID]} \
        --epochs 100 \
        --mode finetune \
        --logfile logs/baseline_cifar100_acc.txt
done

echo "=== Phase B: CPG === $(date)"

GPU_ID=0
setting='scratch_mul_1.5'
baseline_cifar100_acc='logs/baseline_cifar100_acc.txt'
max_allowed_network_width_multiplier=1.5

arch='custom_vgg_cifar100'
finetune_epochs=100
network_width_multiplier=1.0
pruning_ratio_interval=0.1
lr=1e-2
lr_mask=5e-4
gradual_prune_lr=1e-3
num_classes=5
batch_size=32
total_num_tasks=20

for task_id in `seq 1 $NUM_TASKS`; do
    echo "--- CPG task $task_id: ${dataset[task_id]} $(date)"

    # Training the network on current tasks
    state=2
    while [ $state -eq 2 ]; do
        if [ "$task_id" != "1" ]
        then
            python CPG_cifar100_main_normal.py \
                --arch $arch \
                --dataset ${dataset[task_id]} --num_classes $num_classes \
                --lr $lr \
                --lr_mask $lr_mask \
                --batch_size $batch_size \
                --weight_decay 4e-5 \
                --save_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/scratch \
                --load_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id-1]}/gradual_prune \
                --epochs $finetune_epochs \
                --mode finetune \
                --network_width_multiplier $network_width_multiplier \
                --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier \
                --baseline_acc_file $baseline_cifar100_acc \
                --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune/record.txt \
                --log_path checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/train.log \
                --total_num_tasks $total_num_tasks
        else
            python CPG_cifar100_main_normal.py \
                --arch $arch \
                --dataset ${dataset[task_id]} --num_classes $num_classes \
                --lr $lr \
                --lr_mask $lr_mask \
                --batch_size $batch_size \
                --weight_decay 4e-5 \
                --save_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/scratch \
                --epochs $finetune_epochs \
                --mode finetune \
                --network_width_multiplier $network_width_multiplier \
                --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier \
                --baseline_acc_file $baseline_cifar100_acc \
                --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune/record.txt \
                --log_path checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/train.log \
                --total_num_tasks $total_num_tasks
        fi

        state=$?
        if [ $state -eq 2 ]
        then
            network_width_multiplier=$(bc <<< $network_width_multiplier+0.5)
            echo "New network_width_multiplier: $network_width_multiplier"
            continue
        elif [ $state -eq 3 ]
        then
            echo "You should provide the baseline_cifar100_acc.txt as criterion to decide whether the capacity of network is enough for new task"
            exit 0
        fi
    done
    nrof_epoch=0
    nrof_epoch_for_each_prune=20
    start_sparsity=0.0
    end_sparsity=0.1
    nrof_epoch=$nrof_epoch_for_each_prune

    # Prune the model after training
    if [ $state -ne 5 ]
    then
        echo $state
        # gradually pruning
        python CPG_cifar100_main_normal.py \
            --arch $arch \
            --dataset ${dataset[task_id]} --num_classes $num_classes \
            --lr $gradual_prune_lr \
            --lr_mask 0.0 \
            --batch_size $batch_size \
            --weight_decay 4e-5 \
            --save_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune \
            --load_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/scratch \
            --epochs $nrof_epoch \
            --mode prune \
            --initial_sparsity=$start_sparsity \
            --target_sparsity=$end_sparsity \
            --pruning_frequency=10 \
            --pruning_interval=4 \
            --baseline_acc_file $baseline_cifar100_acc \
            --network_width_multiplier $network_width_multiplier \
            --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier \
            --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune/record.txt \
            --log_path checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/train.log \
            --total_num_tasks $total_num_tasks

        if [ $? -ne 6 ]
        then
            for RUN_ID in `seq 1 9`; do
                nrof_epoch=$nrof_epoch_for_each_prune
                start_sparsity=$end_sparsity
                if [ $RUN_ID -lt 9 ]
                then
                    end_sparsity=$(bc <<< $end_sparsity+$pruning_ratio_interval)
                else
                    end_sparsity=$(bc <<< $end_sparsity+0.05)
                fi

                python CPG_cifar100_main_normal.py \
                    --arch $arch \
                    --dataset ${dataset[task_id]} --num_classes $num_classes \
                    --lr $gradual_prune_lr \
                    --lr_mask 0.0 \
                    --batch_size $batch_size \
                    --weight_decay 4e-5 \
                    --save_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune \
                    --load_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune \
                    --epochs $nrof_epoch \
                    --mode prune \
                    --initial_sparsity=$start_sparsity \
                    --target_sparsity=$end_sparsity \
                    --pruning_frequency=10 \
                    --pruning_interval=4 \
                    --baseline_acc_file $baseline_cifar100_acc \
                    --network_width_multiplier $network_width_multiplier \
                    --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier \
                    --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune/record.txt \
                    --log_path checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/train.log \
                    --total_num_tasks $total_num_tasks

                if [ $? -eq 6 ]
                then
                    break
                fi
            done
        fi
    fi

    # Choose the checkpoint that we want
    python tools/choose_appropriate_pruning_ratio_for_next_task.py \
        --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune/record.txt \
        --baseline_acc_file $baseline_cifar100_acc \
        --allow_acc_loss 0.0 \
        --dataset ${dataset[task_id]} \
        --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier \
        --network_width_multiplier $network_width_multiplier \
        --log_path checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/train.log

    if [ $task_id != 1 ] && [ $state -ne 5 ]
    then
    	# Retrain piggymask and weight
    	python CPG_cifar100_main_normal.py \
    	    --arch $arch \
    	    --dataset ${dataset[task_id]} --num_classes $num_classes \
    	    --lr $gradual_prune_lr \
    	    --lr_mask 1e-4 \
    	    --batch_size $batch_size \
    	    --weight_decay 4e-5 \
    	    --save_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/retrain \
    	    --load_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune \
    	    --epochs 30 \
    	    --mode finetune \
    	    --network_width_multiplier $network_width_multiplier \
    	    --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier \
    	    --baseline_acc_file $baseline_cifar100_acc \
    	    --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/retrain/record.txt \
    	    --log_path checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/train.log \
    	    --total_num_tasks $total_num_tasks \
    	    --finetune_again

        # If there is any improve from retraining, use that checkpoint
        python tools/choose_retrain_or_not.py \
            --save_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/gradual_prune \
            --load_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[task_id]}/retrain
    fi
done

echo "=== Phase C: inference (zero-forgetting check) === $(date)"

for TASK_ID in `seq 1 $NUM_TASKS`; do
    python CPG_cifar100_main_normal.py \
        --arch $arch \
        --dataset ${dataset[TASK_ID]} --num_classes 5 \
        --load_folder checkpoints/CPG/experiment1/$setting/$arch/${dataset[NUM_TASKS]}/gradual_prune \
        --mode inference \
        --baseline_acc_file $baseline_cifar100_acc \
        --network_width_multiplier $network_width_multiplier \
        --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier \
        --log_path logs/cifar100_inference_1to3.log
done

echo "=== DONE === $(date)"
