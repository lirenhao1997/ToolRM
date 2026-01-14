export TEMPERATURE=0.6
export TOP_P=0.95
export TOP_K=20
export MAX_PROMPT_LEN=32768
export MAX_RESPONSE_LEN=8192

BASE_DATA_DIR="./data"
OUTPUT_DIR="./trbench_results"

# evaluate Qwen3-4B-Thinking-2507 on TRBench-BFCL
MODEL_NAME="Qwen3-4B-Thinking-2507"
MODEL_DIR="Qwen/Qwen3-4B-Thinking-2507"
python eval/eval_trbench_local.py \
    --local_llm_name ${MODEL_NAME} \
    --local_llm_dir ${MODEL_DIR} \
    --base_data_dir ${BASE_DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --llm_infer_mode think

# evaluate ToolRM-Gen on TRBench-BFCL
MODEL_CKPT_DIR="./model_ckpt"
MODEL_NAME="ToolRM-Gen-Qwen3-4B-Thinking-2507"
MODEL_DIR=${MODEL_CKPT_DIR}/${MODEL_NAME}
python eval/eval_trbench_local.py \
    --local_llm_name ${MODEL_NAME} \
    --local_llm_dir ${MODEL_DIR} \
    --base_data_dir ${BASE_DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --llm_infer_mode think