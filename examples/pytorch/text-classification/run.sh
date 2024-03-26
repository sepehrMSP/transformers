#!/bin/bash

# Create the output folder
mkdir -p easl_output
export TASK_NAME=mnli

# Loop through different batch sizes
for ((i=4; i<=16; i++))
do
    batch_size=$((2**i))

    # Run the command with the current batch size
    echo "Running with batch size: $batch_size"
    taskset -c 0-47 python run_glue.py --model_name_or_path google-bert/bert-base-cased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_train_batch_size $batch_size \
        --learning_rate 2e-5 \
        --num_train_epochs 5 \
        --output_dir out/$TASK_NAME/ \
        --dataloader_num_workers=48 \
        | tee easl_output/output_batch_size_${batch_size}.txt

    echo "Finished running with batch size: $batch_size"
done
