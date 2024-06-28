#!/bin/bash

function main() {
    seed=0
    lr_u=1e-7
    lr_w=1e-7
    lr_lambda=1e-2
    batch_size=64
    micro_batch_size=2
    val_batch_size=1
    epoch=1
    exp_name=exp1_Yi-34B_demo

    data_dir=data_tiny/
    num_partitions=`ls -l ${data_dir}/train_*.json|wc -l`
    log_dir=log_${exp_name}/
    save_dir=models/${exp_name}/
    wandb_proj='bilevel-optimization'
    alpha=100
    mkdir -p ${log_dir}
    mkdir -p ${save_dir}

    local exp_id=${exp_name}_lr-u-${lr_u}_lr-w-${lr_w}_lr_lambda-${lr_lambda}_seed-${seed}_alpha-${alpha}_epoch-${epoch}

          echo "$(date): ${exp_id}"

          accelerate launch --config_file fsdp_config.yaml python/main.py \
              --wandb_run_name ${exp_id} \
              --wandb_project ${wandb_proj} \
              --train-data ${data_dir}/train_\*.json \
              --val-data ${data_dir}/val.json \
              --test-data ${data_dir}/test.json \
              --model 01-ai/Yi-34B \
              --micro_batch_size ${micro_batch_size} \
              --global_batch_size $batch_size \
              --max-length 1024 \
              --tokenizer-name 01-ai/Yi-34B \
              --optimizer "name=adamw" \
              --init_lr 5e-4 \
              --minmax_init_lr_u ${lr_u} \
              --minmax_init_lr_w ${lr_w} \
              --minmax_init_lr_lambda ${lr_lambda} \
              --lr_scheduler "name=cosine, min_lr=0.0" \
              --validation_model_mode "train" \
              --minmax_validation_sampler stochastic \
              --sampler stochastic \
              --pseudo_random \
              --logging_conf_file conf/common.log_conf \
              --init_alpha ${alpha} \
              --tau 1 \
              --seed ${seed} \
              --epoch $epoch \
              --num_outer_iter 1 \
              --bf16 \
              --lisa \
              --model-type llama \
              --use_wandb \
              --val_batch_size ${val_batch_size} \
              --eval_frequency 50 \
              --num_partitions ${num_partitions} \
              --response_loss_only \
              --save_dir ${save_dir} \
              > ${log_dir}/${exp_id}.log \
              2> ${log_dir}/${exp_id}.err
}


main "$@"
