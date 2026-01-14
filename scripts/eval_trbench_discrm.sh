BASE_DATA_DIR="./data"
OUTPUT_DIR="./trbench_results"

# evaluate Skywork-Reward-V2 on TRBench-BFCL
MODEL_NAME="Skywork-Reward-V2-Qwen3-4B"
MODEL_DIR="Skywork/Skywork-Reward-V2-Qwen3-4B"
python eval/eval_trbench_local_discrm.py \
    --local_llm_name ${MODEL_NAME} \
    --local_llm_dir ${MODEL_DIR} \
    --base_data_dir ${BASE_DATA_DIR} \
    --output_dir ${OUTPUT_DIR}

# evaluate ToolRM-Disc on TRBench-BFCL
MODEL_CKPT_DIR="./model_ckpt"
MODEL_NAME="ToolRM-Disc-Qwen3-4B-Instruct-2507"
MODEL_DIR=${MODEL_CKPT_DIR}/${MODEL_NAME}
python eval/eval_trbench_local_discrm.py \
    --local_llm_name ${MODEL_NAME} \
    --local_llm_dir ${MODEL_DIR} \
    --base_data_dir ${BASE_DATA_DIR} \
    --output_dir ${OUTPUT_DIR}