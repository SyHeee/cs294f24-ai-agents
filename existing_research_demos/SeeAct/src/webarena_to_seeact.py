import os
import json
import toml

# Configuration template
CONFIG_TEMPLATE = {
    "basic": {
        "is_demo": True,
        "save_file_dir": "../online_results",
        "default_task": "Find pdf of paper \"GPT-4V(ision) is a Generalist Web Agent, if Grounded\" from arXiv",
        "default_website": "https://www.google.com/"
    },
    "experiment": {
        "task_file_path": "./output/tasks.json",  # Path to generated tasks.json
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
    "openai": {
        "api_key": "yourapi",
        "rate_limit": -1,
        "model": "gpt-4-vision-preview",
        "temperature": 0
    },
    "oss_model": {},
    "playwright": {
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
}

def get_url_mapping():
    """Retrieve URL mapping from environment variables."""
    base_url = os.environ.get("MYWEBURL", "http://128.105.144.15")
    return {
        "__SHOPPING__": os.environ.get("SHOPPING", f"{base_url}:7770"),
        "__SHOPPING_ADMIN__": os.environ.get("SHOPPING_ADMIN", f"{base_url}:7780/admin"),
        "__REDDIT__": os.environ.get("REDDIT", f"{base_url}:9999"),
        "__GITLAB__": os.environ.get("GITLAB", f"{base_url}:8023"),
        "__MAP__": os.environ.get("MAP", f"{base_url}:3000"),
        "__WIKIPEDIA__": os.environ.get(
            "WIKIPEDIA", f"{base_url}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
        ),
        "__HOMEPAGE__": os.environ.get("HOMEPAGE", f"{base_url}:4399")
    }

def clean_task(task, url_mapping):
    """Simplify the task to match the expected output format."""
    start_url = task["start_url"]
    for placeholder, actual_url in url_mapping.items():
        if placeholder in start_url:
            start_url = start_url.replace(placeholder, actual_url)
            break

    return {
        "task_id": task["task_id"],
        "intent": task["intent"],
        "require_login": task.get("require_login", False),
        "storage_state": task.get("storage_state", ""),
        "start_url": start_url,
        "eval": {
            "eval_types": task["eval"]["eval_types"],
            "reference_answers": task["eval"].get("reference_answers", {})
        }
    }

def generate_output(input_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Write config.toml
    config_path = os.path.join(output_dir, "config.toml")
    with open(config_path, "w", encoding="utf-8") as config_file:
        toml.dump(CONFIG_TEMPLATE, config_file)

    # Load tasks and process them
    with open(input_file, "r", encoding="utf-8") as f:
        raw_tasks = json.load(f)

    # Retrieve URL mapping from environment variables
    url_mapping = get_url_mapping()

    # Simplify tasks
    simplified_tasks = [clean_task(task, url_mapping) for task in raw_tasks]

    # Save simplified tasks to tasks.json
    tasks_path = os.path.join(output_dir, "tasks.json")
    with open(tasks_path, "w", encoding="utf-8") as tasks_file:
        json.dump(simplified_tasks, tasks_file, indent=2)

    print(f"Generated config.toml at: {config_path}")
    print(f"Generated tasks.json at: {tasks_path}")

if __name__ == "__main__":
    input_path = "./test.raw.json"  # Input JSON file
    output_folder = "./output"  # Output directory
    generate_output(input_path, output_folder)
