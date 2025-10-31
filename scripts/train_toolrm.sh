set -x

DATA_HOME="<YOUR_LOCAL_DATA_DIR>"
PROJECT_NAME="<YOUR_PROJECT_NAME>"
EXPERIMENT_NAME="<YOUR_EXPERIMENT_NAME>"
MODEL_CKPT_HOME='<YOUR_MODEL_CHECKPOINT_DIR>'

# step-1: train ToolRM-Qwen3-4B-Thinking-2507 on 8 * NVIDIA A100-80G
VERL_USE_MODELSCOPE=True \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_HOME/toolrm_train_think.parquet \
    data.val_files=$DATA_HOME/toolrm_test_think.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=16384 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Thinking-2507 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path='./verl/utils/reward_score/toolrm_reward_function.py' \
    custom_reward_function.name='compute_score_pairwise' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto$@

# step-2: 
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir verl/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME/global_step_220/actor \
    --target_dir $CKPT_HOME/$EXPERIMENT_NAME