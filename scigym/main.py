import argparse
import os
import time

import yaml
from dotenv import load_dotenv

from scigym.controller import Controller

load_dotenv()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run scientific hypothesis benchmark")

    # Add config file argument
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/config.yml",
        help="Path to configuration YAML file",
    )

    # Add other command line arguments that can override config
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        help="Model name (overrides config)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Dataset name (overrides config)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="Output directory (overrides config)",
    )

    args = parser.parse_args()

    # Load config from file
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    # Override config with command line arguments if provided
    cli_args = vars(args)
    for key, value in cli_args.items():
        if key != "config_path" and value is not None:
            if key in config:
                config[key] = value
            else:
                # Handle nested config structure if needed
                # This is a simple example, you might need to adjust based on your config structure
                config[key] = value

    return config


def find_file(root_dir, target_file="evaluation.json"):
    """
    Recursively search for target_file starting from root_dir.
    Returns the full path if found, None otherwise.
    """
    for root, dirs, files in os.walk(root_dir):
        if target_file in files:
            return os.path.join(root, target_file)
    return None


import json
import os
from datetime import datetime


def check_file_condition(path):
    # Check if the path exists
    if not os.path.exists(path):
        return False

    # Get all directories in the path that match timestamp format
    timestamp_dirs = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        # Check if it's a directory and follows timestamp format (20250508_121224)
        if os.path.isdir(item_path) and len(item) == 15 and item[8] == "_":
            # Validate it's actually a timestamp by parsing it
            datetime.strptime(item, "%Y%m%d_%H%M%S")
            timestamp_dirs.append(item)

    # If no timestamp directories found, return False
    if not timestamp_dirs:
        return False

    # Find the latest timestamp directory
    latest_dir = max(timestamp_dirs)
    latest_dir_path = os.path.join(path, latest_dir)

    # Check if evaluation.json exists in the latest directory
    txt_path = os.path.join(latest_dir_path, "chat_history_readable.txt")
    if not os.path.exists(txt_path):
        return False

    json_path = os.path.join(latest_dir_path, "evaluation.json")
    if not os.path.exists(json_path):
        return False

    # Read the json file and check the "success" field
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Return the value of "success" field (True or False)
        return data.get("success", False)
    except Exception:
        # Return False if there's any error reading or parsing the file
        return False


def main():
    """
    Main function to run the benchmark.
    """
    benchmark_config = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(benchmark_config["output_dir"], exist_ok=True)

    model_name = benchmark_config["model_name"]
    benchmark_name = os.path.basename(os.path.normpath(benchmark_config["benchmark_dir"]))
    output_dir = os.path.join(benchmark_config["output_dir"], benchmark_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join(output_dir, model_name)
    should_overwrite = benchmark_config.get("overwrite", False)

    if not should_overwrite and os.path.exists(output_dir) and check_file_condition(output_dir):
        print(f"Benchmark {benchmark_name} with {model_name} has been evaluated. Skipping...")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Starting benchmark {benchmark_name} with {model_name}")

    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(output_dir, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/config.yml", "w") as file:
        yaml.dump(benchmark_config, file, default_flow_style=False)

    # Create controller and run benchmark
    controller = Controller(
        path_to_sbml_cfg=benchmark_config["benchmark_dir"],
        model_name=benchmark_config["model_name"],
        api_key=os.environ[benchmark_config["api_key_env_name"]],
        max_iterations=benchmark_config["max_iterations"],
        test_memorize=benchmark_config["test_memorize"],
        output_directory=output_dir,
        # task_difficulty=benchmark_config["difficulty"],
        # anonymize=benchmark_config["anonymize"],
        experiment_actions_path=benchmark_config["experiment_actions_path"],
        customized_functions_path=benchmark_config["customized_functions_path"],
        eval_debug_rounds=benchmark_config["eval_debug_rounds"],
        temperature=benchmark_config["temperature"],
    )

    results = controller.run_benchmark()


if __name__ == "__main__":
    main()
