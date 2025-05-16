# SciGYM: Measuring Scientific Capabilities of Language Models with a Systems Biology Dry Lab

## Setup

To create the conda environment, run the following commands:

```bash
conda create --name scigym python=3.10.16 -y
conda activate scigym

pip install -e .

# Required for graph edit distance metric
conda install --channel conda-forge pygraphviz

# Optional development tools
pip install pre-commit
pre-commit install
```

## Download Benchmark Dataset

We host our full benchmark suite on HuggingFace and provide a script to download it. We provide two splits of the benchmark dataset:

- `small`: Consists of 137 models we evaluated in our paper
- `large`: Consists of an additional 213 models we did not evaluate

To download the splits, run the following commands:

```bash
python download.py --split small --save_dir <path_to_save_dir>
python download.py --split large --save_dir <path_to_save_dir>
```

## Setup an Agent

You can use one of our [supported agents](#currently-supported-agents) with your own API key. To set this up, please modify the `example.env` file in the root directory of the repository and rename it to `.env`. The file should contain the API key for the agent you want to use.

### Currently Supported Agents

- `gemini-2.5-pro-preview-03-25`
- `claude-3-5-haiku-20241022`
- `claude-3-7-sonnet-20250219`

## Running the Benchmark

To run the benchmark, you need to provide a configuration yaml that specifies the required parameters for the input, output, agent, and environment components of the run. We provide a default configuration file in `configs/default.yaml` which you can modify to suit your needs. The fields in this configuration file are detailed [below](#configuration-fields).

To run the benchmark, use the following command:

```bash
python scigym/main.py --config_path configs/config.yaml
```

### Configuration Fields

- `benchmark_dir`: The directory where the benchmark instance folder is located. You will have these folders after downloading the [benchmark dataset](#download-benchmark-dataset).
- `model_name`: One of the model names in the [supported agents list](#currently-supported-agents). This is the agent that will be used to run the benchmark.
- `api_key_env_name`: The environment variable name that contains the API key for the agent. This is required for agents that require an API key to run.
- `overwrite`: Whether to overwrite existing results for the current benchmark instance.
- `test_memorize`: Whether to test the agent's ability to memorize the model in a one-shot setting
- `eval_debug_rounds`: Number of rounds to allow the agent to re-submit its hypothesis SBML if there are errors in the previous submission
- `temperature`: The temperature parameter for the LLM
- `experiment_actions_path`: Path to system prompt explaining the experimental actions
- `customized_functions_path`: Path to system prompt explaining the routines that the agent can use
- `output_dir`: The directory where the output files will be saved
