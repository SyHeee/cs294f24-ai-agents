import os
import json
import toml
from collections import defaultdict
from typing import Any

# 全局 OpenAI 配置
GLOBAL_OPENAI_CONFIG = {
    "api_key": "YOUR-OPENAI_API",
    "rate_limit": -1,
    "model": "gpt-4-vision-preview",
    "temperature": 0
}

# 全局 Playwright 配置
GLOBAL_PLAYWRIGHT_CONFIG = {
    "save_video": False,
    "tracing": False,
    "locale": "en-US",
    "geolocation.longitude": 39.99542778404405,
    "geolocation.latitude": -83.0068669912263,
    "viewport.width": 1280,
    "viewport.height": 720,
    "trace.screenshots": True,
    "trace.snapshots": True,
    "trace.sources": True
}

# URL 映射规则
URL_MAPPING = {
    "__SHOPPING_ADMIN__": "http://128.105.144.15:7780/admin",
    "__SHOPPING__": "http://128.105.144.15:7770",
    "__REDDIT__": "http://128.105.144.15:9999",
    "__GITLAB__": "http://128.105.144.15:8023",
    "__MAP__": "http://128.105.144.15:3000",
    "__WIKIPEDIA__": "http://128.105.144.15:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
    "__HOMEPAGE__": "http://128.105.144.15:4399",
}

def convert_task_to_toml(task: dict, template_to_id: dict[str, int]) -> dict:
    """将任务转换为符合目标格式的 TOML 配置"""
    # 替换 start_url
    start_url = task["start_url"]
    for placeholder, actual_url in URL_MAPPING.items():
        if placeholder in start_url:
            start_url = start_url.replace(placeholder, actual_url)
            break

    # 生成任务的 TOML 配置
    task_toml = {
        "basic": {
            "is_demo": False,
            "save_file_dir": f"./results/{task['task_id']}",
            "default_task": task["intent"],
            "default_website": start_url
        },
        "experiment": {
            "task_file_path": "",
            "overwrite": False,
            "top_k": 50,
            "fixed_choice_batch_size": 17,
            "dynamic_choice_batch_size": 1600,
            "max_continuous_no_op": 2,
            "max_op": 20,
            "highlight": False,
            "monitor": True,
            "dev_mode": False
        },
        "openai": GLOBAL_OPENAI_CONFIG,
        "playwright": GLOBAL_PLAYWRIGHT_CONFIG
    }

    # 添加评估细节
    eval_types = task["eval"]["eval_types"]
    if "string_match" in eval_types:
        task_toml["extra"] = {
            "reference_answers": task["eval"].get("reference_answers", {}),
            "validation_type": "string_match"
        }
    elif "url_match" in eval_types:
        task_toml["extra"] = {
            "reference_url": task["eval"].get("reference_url", ""),
            "validation_type": "url_match"
        }
    elif "program_html" in eval_types:
        task_toml["extra"] = {
            "program_html": task["eval"].get("program_html", []),
            "validation_type": "program_html"
        }

    return task_toml

def main(input_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    template_to_id: dict[str, int] = defaultdict(lambda: len(template_to_id))

    with open(input_file, "r", encoding="utf-8") as f:
        raw_tasks = json.load(f)

    for task in raw_tasks:
        try:
            task_toml = convert_task_to_toml(task, template_to_id)
            task_id = task["task_id"]
            task_name = f"task_{task_id}.toml"

            output_path = os.path.join(output_dir, task_name)
            with open(output_path, "w", encoding="utf-8") as f_out:
                toml.dump(task_toml, f_out)

            print(f"Task {task_id} converted and saved to {output_path}")
        except Exception as e:
            print(f"Error processing task {task['task_id']}: {e}")

if __name__ == "__main__":
    input_path = "./test.raw.json"  # 输入 JSON 文件路径
    output_folder = "./converted_tasks"  # 输出文件夹
    main(input_path, output_folder)
