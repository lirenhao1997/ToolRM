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

import json

def jsonl_load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        res = [json.loads(line) for line in f if line.strip()]
    return res

def jsonl_dump(data, path: str, mode: str = "w"):
    with open(path, mode, encoding="utf-8") as f:
        if isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
