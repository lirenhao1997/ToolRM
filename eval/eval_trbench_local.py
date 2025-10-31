"""
Copyright (c) 2025, Alibaba Cloud and its affiliates;
Licensed under the CC BY-NC-SA 4.0 License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://creativecommons.org/licenses/by-nc-sa/4.0/

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset

import numpy as np
import jsonlines
import argparse
import os
import re

# Use additional parameters to determine the thinking mode of the model.
ADD_ENABLE_THINKING_TO_TOKENIZER = [
    'qwen3-4b',
    'qwen3-8b',
]

def jsonl_load(path: str):
    with jsonlines.open(path, "r") as reader:
        res = list(reader)
    return res


def jsonl_dump(data, path: str, mode: str="w"):
    with jsonlines.open(path, mode) as writer:
        for item in data:
            writer.write(item)


def extract_answer_from_text(text: str):
    """
    Extract the answer within '<choice>' and '</choice>' tags.
    The answer should be either '1' or '2', else None.
    """
    revise_success = -1
    answer_pattern = r'<choice>\n(.*?)\n</choice>'
    soft_answer_pattern = r'<choice>(.*?)</choice>'  # a pattern for softer answer matching

    match = re.search(answer_pattern, text, re.DOTALL)

    if not match:
        match = re.search(soft_answer_pattern, text, re.DOTALL)
        if not match:
            revise_success = 0
            return None, revise_success
        else:
            revise_success = 1

    answer_str = match.group(1).strip()
    if answer_str in ['1', '2']:
        return answer_str, revise_success
    else:
        return None, revise_success


def compute_score_pairwise(pred_answer, ground_truth) -> float:
    if pred_answer is None:
        rm_reward = 0
        pred_answer = ""
    else:
        if pred_answer == ground_truth:
            rm_reward = 1
        else:
            rm_reward = 0
    
    return rm_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_llm_name", default="qwen3-4b")
    parser.add_argument("--local_llm_dir", default="Qwen/Qwen3-4B")
    parser.add_argument("--base_data_dir", default="./data")
    parser.add_argument("--output_dir", default="./trbench_results")
    parser.add_argument("--llm_infer_mode", default="think")
    parser.add_argument("--load_exist_results", action="store_true")

    args = parser.parse_args()
    local_llm_name = args.local_llm_name
    local_llm_dir = args.local_llm_dir
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir
    llm_infer_mode = args.llm_infer_mode
    load_exist_results = args.load_exist_results

    backbone_model_name = local_llm_name.split('_')[0].lower()
    assert llm_infer_mode in ['think', 'no_think'], "The 'llm_infer_mode' parameter should be 'think' or 'no_think'."

    query_input_dir = f"{base_data_dir}/trbench_test_{llm_infer_mode}.parquet"
    query_output_dir = f"{output_dir}/{local_llm_name}.jsonl"

    dataset = load_dataset('parquet', data_files=query_input_dir)
    test_samples = dataset['train'].to_list()

    tokenizer = AutoTokenizer.from_pretrained(local_llm_dir)
    max_prompt_len = int(os.environ.get("MAX_PROMPT_LEN")) if os.environ.get("MAX_PROMPT_LEN") is not None else 32768

    texts = []
    test2raw = {}
    for index, sample in enumerate(test_samples):
        prompt = sample['prompt']
        prompt_len = len(tokenizer.encode(prompt[0]['content'], add_special_tokens=True))
        if prompt_len > max_prompt_len:
            continue
        if backbone_model_name in ADD_ENABLE_THINKING_TO_TOKENIZER:
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True if llm_infer_mode == 'think' else False,  # Switches between thinking and non-thinking modes for Qwen3 hybrid models. Default is True.
            )
        else:
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        test2raw[len(texts)] = index
        texts.append(text)
    print(f"Filter {len(test_samples)-len(texts)} samples with max_prompt_len={max_prompt_len}")

    # Generate outputs with local-deployed llm
    if load_exist_results:
        with jsonlines.open(query_output_dir, 'r') as reader:
            query_outputs = list(reader)

    else:
        temperature = float(os.environ.get("TEMPERATURE")) if os.environ.get("TEMPERATURE") is not None else 0
        top_p = float(os.environ.get("TOP_P")) if os.environ.get("TOP_P") is not None else 1.0
        top_k = int(os.environ.get("TOP_K")) if os.environ.get("TOP_K") is not None else -1
        repetition_penalty = float(os.environ.get("REPETITION_PENALTY")) if os.environ.get("REPETITION_PENALTY") is not None else 1.0
        presence_penalty = float(os.environ.get("PRESENCE_PENALTY")) if os.environ.get("PRESENCE_PENALTY") is not None else 0.0
        max_response_len = int(os.environ.get("MAX_RESPONSE_LEN")) if os.environ.get("MAX_RESPONSE_LEN") is not None else 32768

        inference_sampling_params = {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'repetition_penalty': repetition_penalty,
            'presence_penalty': presence_penalty,
            'max_tokens': max_response_len,
        }
        print(f"inference_sampling_params: {inference_sampling_params}")
        sampling_params = SamplingParams(**inference_sampling_params)

        llm = LLM(
            model=local_llm_dir,
            gpu_memory_utilization=0.95,
            max_model_len=32768,
            disable_cascade_attn=True,  # To deal with a performance degradation issue under large batch size. See https://github.com/vllm-project/vllm/issues/17652
        )
        outputs = llm.generate(texts, sampling_params)
        
        query_outputs = []
        for test_index, raw_index in test2raw.items():
            query_outputs.append(
                {
                    "index": test_samples[raw_index]['extra_info']['index'],
                    "original_index": test_samples[raw_index]['extra_info']['original_index'],
                    "input_text": texts[test_index],
                    "output_text": outputs[test_index].outputs[0].text
                }
            )

        if not os.path.exists(os.path.dirname(query_output_dir)):
            os.makedirs(os.path.dirname(query_output_dir))
        jsonl_dump(
            data=query_outputs,
            path=query_output_dir
        )

    success_parse_count = 0
    revised_success_parse_count = 0
    all_val_res = []
    val_res_by_task_source = {}

    for test_index, raw_index in test2raw.items():
        test_sample = test_samples[raw_index]
        model_output = query_outputs[test_index]['output_text'].strip()
        ground_truth = test_sample['reward_model']['ground_truth']
        task_source = test_sample['extra_info']['task_source']

        pred_answer, revise_success = extract_answer_from_text(text=model_output)
        score = compute_score_pairwise(pred_answer=pred_answer, ground_truth=ground_truth)

        if revise_success == -1:
            success_parse_count += 1
        if revise_success in [-1, 1]:
            revised_success_parse_count += 1

        all_val_res.append(score)

        if task_source not in val_res_by_task_source:
            val_res_by_task_source[task_source] = [score]
        else:
            val_res_by_task_source[task_source].append(score)

    print(f"======Evaluation result of {local_llm_name}======")
    print(f"Total number of samples in dataset: {len(test_samples)}")
    print(f"Total number of tested samples: {len(texts)}")
    print("-"*50)
    print(f"Number of successfully parsed samples: {success_parse_count}")
    print("-"*50)
    print(f"Number of successfully parsed samples after revision: {revised_success_parse_count}")
    print("-"*50)
    val_res_batch_1 = all_val_res[:len(all_val_res)//2]
    val_res_batch_2 = all_val_res[len(all_val_res)//2:]
    all_task_val_res = [res_1 * res_2 for res_1, res_2 in zip(val_res_batch_1, val_res_batch_2)]
    print(f"Average test score per sample: {np.mean(all_task_val_res)}")
    print("-"*50)
    for task_source, scores in val_res_by_task_source.items():
        task_val_res = [res_1 * res_2 for res_1, res_2 in zip(scores[:len(scores)//2], scores[len(scores)//2:])]
        print(f"Average test score in {task_source}: {np.mean(task_val_res)}")
