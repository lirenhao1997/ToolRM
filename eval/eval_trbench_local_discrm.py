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

import os
import torch
import copy
import argparse
import logging
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import jsonl_load, jsonl_dump

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_existing_results(output_path):
    if not os.path.exists(output_path):
        return 0, []
    
    existing_results = jsonl_load(output_path)

    processed_samples = len(existing_results) // 2
    return processed_samples, existing_results


def single_inference_with_retry(model, tokenizer, all_chats, output_path=None, max_retries=3, device="cuda"):
    all_scores = []
    
    if output_path:
        processed_samples, existing_results = load_existing_results(output_path)
        if processed_samples > 0:
            logger.info(f"Found existing results: skip {processed_samples} samples ({processed_samples * 2} records)")
            start_idx = processed_samples * 2
            all_scores = [result['score'] for result in existing_results]
        else:
            start_idx = 0
    else:
        start_idx = 0
    
    all_convs_formatted = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in all_chats]
    
    revised_convs = []
    for conv_formatted in all_convs_formatted:
        if tokenizer.bos_token is not None and conv_formatted.startswith(tokenizer.bos_token):
            conv_formatted = conv_formatted[len(tokenizer.bos_token):]
        revised_convs.append(conv_formatted)
    
    with torch.no_grad():
        pbar = tqdm(total=len(revised_convs), initial=start_idx, desc='Single Sample Inference')
        
        for i in range(start_idx, len(revised_convs)):
            conv = revised_convs[i]
            original_chat = all_chats[i]
            
            retry_count = 0
            while retry_count < max_retries:
                try:
                    conv_tokenized = tokenizer(conv, return_tensors="pt").to(device)
                    score = model(**conv_tokenized).logits[0][0].item()
                    all_scores.append(score)
                    
                    if output_path:
                        result = {
                            'chat': original_chat,
                            'score': float(score)
                        }
                        jsonl_dump(result, output_path, mode="a")
                    
                    del conv_tokenized
                    torch.cuda.empty_cache()
                    
                    pbar.update(1)
                    break
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                        retry_count += 1
                        logger.warning(f"Sample {i} OOM error. Retry: {retry_count}/{max_retries}")
                        
                        if 'conv_tokenized' in locals():
                            del conv_tokenized
                        torch.cuda.empty_cache()
                        
                        if retry_count >= max_retries:
                            logger.error(f"Sample {i} inference fail: reach retry limits")
                            raise e
                    else:
                        raise e
        
        pbar.close()
    
    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_llm_name", default="Skywork-Reward-V2-Qwen3-4B")
    parser.add_argument("--local_llm_dir", default="Skywork/Skywork-Reward-V2-Qwen3-4B")
    parser.add_argument("--base_data_dir", default="./data")
    parser.add_argument("--output_dir", default="./trbench_results")

    args = parser.parse_args()
    local_llm_name = args.local_llm_name
    local_llm_dir = args.local_llm_dir
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir

    query_input_dir = f"{base_data_dir}/trbench_test_think.parquet"
    query_output_dir = f"{output_dir}/{local_llm_name}.jsonl"
    
    os.makedirs(os.path.dirname(query_output_dir), exist_ok=True)

    dataset = load_dataset('parquet', data_files=query_input_dir)
    test_samples = dataset['train'].to_list()
    test_samples = test_samples[:len(test_samples)//2]

    device = "cuda:0"
    model = AutoModelForSequenceClassification.from_pretrained(
        local_llm_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_llm_dir)

    all_chats = []
    gts = []

    processed_samples, _ = load_existing_results(query_output_dir)
    start_idx = processed_samples

    for idx, sample in enumerate(test_samples[start_idx:], start_idx):
        chat_history = sample['chat_history']
        
        # concate the system message with the first user message for models which not support 'system' role
        if 'gemma' in local_llm_name.lower():
            system_msg = chat_history[0]
            first_user_msg = chat_history[1]
            assert system_msg['role'] == 'system' and first_user_msg['role'] == 'user'
            new_first_msg = {
                "role": "user",
                "content": f"<system message start>\n{system_msg['content']}\n<system message end>\n" + first_user_msg['content']
            }
            chat_history = chat_history[2:]
            chat_history.insert(0, new_first_msg)
        
        msg1 = {
            'role': 'assistant',
            'content': sample['chosen_response']
        }
        msg2 = {
            'role': 'assistant', 
            'content': sample['reject_response']
        }
        
        chat_1 = copy.deepcopy(chat_history)
        chat_1.append(msg1)
        chat_2 = copy.deepcopy(chat_history)
        chat_2.append(msg2)
        
        all_chats.append(chat_1)
        all_chats.append(chat_2)
        gts.append("1")

    if start_idx > 0:
        logger.info(f"Start from sample {start_idx}: {len(test_samples) - start_idx} samples wait for inference.")

    all_scores = single_inference_with_retry(
        model, tokenizer, all_chats,
        output_path=query_output_dir,
        device=device
    )

    if start_idx > 0:
        _, all_results = load_existing_results(query_output_dir)
        all_scores = [result['score'] for result in all_results]
        gts = ["1"] * len(test_samples)

    logger.info("All inference finished!")

    assert len(all_scores) == 2 * len(gts), f"Unmatched number of scores ({len(all_scores)}) with expected ({2 * len(gts)})"

    val_res = []
    val_res_by_task_source = {}
    task_sources = [sample['extra_info']['task_source'] for sample in test_samples]

    for index in range(len(gts)):
        score_1 = all_scores[2 * index]
        score_2 = all_scores[2 * index + 1]
        gt = gts[index]
        
        score = 1 if score_1 > score_2 else 0
        val_res.append(score)

        task_source = task_sources[index]

        if task_source not in val_res_by_task_source:
            val_res_by_task_source[task_source] = [score]
        else:
            val_res_by_task_source[task_source].append(score)

    print(f"======Evaluation result of {local_llm_name}======")
    print(f"Total number of samples in dataset: {len(test_samples)}")
    print(f"Total number of tested samples: {len(gts)}")
    print("-" * 50)
    print(f"Average test score per sample (W-Avg): {np.mean(val_res):.4f}")
    print("-" * 50)
    for task_source, scores in val_res_by_task_source.items():
        print(f"Average test score in {task_source}: {np.mean(scores):.4f}")
