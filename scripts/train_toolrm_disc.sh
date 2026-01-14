set -x

DATA_HOME='<YOUR_LOCAL_DATA_DIR>'
MODEL_CKPT_HOME='<YOUR_MODEL_CHECKPOINT_DIR>'

# train ToolRM-Disc-Qwen3-4B-Instruct-2507 on 8 * NVIDIA A100-80G
read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ${MODEL_CKPT_HOME}/ToolRM-Disc_qwen3-4b-instruct-2507 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain Qwen/Qwen3-4B-Instruct-2507 \
   --bf16 \
   --max_epochs 2 \
   --max_len 16384 \
   --zero_stage 3 \
   --learning_rate 4e-6 \
   --dataset ${DATA_HOME}/toolrm_train_disc.parquet \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --attn_implementation flash_attention_2 \
   --load_checkpoint \
   --packing_samples \
   --gradient_checkpointing
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
