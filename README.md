# Multi-Layered Self-Reflection with Auto-Prompting for LLMs

This repository provides a reference implementation of the **Multi-Layered Self-Reflection** approach with **Auto-Prompting** for improving multi-step reasoning in Large Language Models (LLMs). Initially demonstrated on mathematical reasoning benchmarks like GSM8K, the framework has been extended to evaluate performance on more complex domains, including the **MATH** (Hendrycks Mathematics), **AIME** (American Invitational Mathematics Examination), and **SWE-bench** (Software Engineering Benchmark) datasets.

The accompanying [paper](#citation) (details to be updated with new benchmark results) demonstrates how this method can significantly boost accuracy across various benchmarks and LLMs.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [General Dependencies](#general-dependencies)
  - [SWE-bench Environment Setup](#swe-bench-environment-setup)
- [Setup](#setup)
  - [API Keys (.env file)](#api-keys-env-file)
  - [Configuration (config.json)](#configuration-configjson)
- [Dataset Preparation](#dataset-preparation)
  - [General](#general)
  - [AIME Specifics](#aime-specifics)
  - [SWE-bench Specifics](#swe-bench-specifics)
- [Usage](#usage)
  - [1. Running Tests](#1-running-tests)
  - [2. Analyzing Results](#2-analyzing-results)
  - [3. Interactive Demo (Legacy)](#3-interactive-demo-legacy)
- [Method Details](#method-details)
- [Supported Benchmarks](#supported-benchmarks)
- [Citation](#citation)
- [License](#license)

---

## Overview

The core idea of **multi-layered Self-Reflection** is to allow the model to:
1. Generate an initial answer (often via Chain-of-Thought or a direct problem-solving attempt).
2. Reflect on incorrect or incomplete steps by **auto-generating** (meta-prompting) a new reflection prompt tailored to the specific mistakes and problem type.
3. Iteratively refine its solution over multiple layers until reaching a correct conclusion or hitting a maximum number of attempts.

This framework now supports evaluation on:
- **GSM8K & GSM-Symbolic**: Elementary mathematics.
- **MATH**: Challenging high-school and undergraduate mathematics problems.
- **AIME**: Prestigious and difficult math competition problems.
- **SWE-bench**: Real-world software engineering tasks involving bug fixing and feature implementation.

---

## Features

- **Automated Reflection Layers**: Iteratively prompt the model to produce refined solutions.
- **Auto-Prompt (Meta-Prompt) Generation**: The model adapts its reflection prompt based on the problem's structure, its prior errors, and the benchmark type.
- **Multiple LLM Support**: Easily switch between different models by updating the `config.json`.
- **Benchmark Integration**: Out-of-the-box support for GSM8K, GSM-Symbolic, MATH, AIME, and SWE-bench datasets.
- **Benchmark-Specific Prompts**: Tailored few-shot examples and meta-prompts for each benchmark.
- **Extensible Evaluation**: Modular design for adding new benchmarks and evaluation methods.

---

## Installation

### General Dependencies

1.  **Clone** this repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git # Replace with actual repo URL
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Linux or Mac
    # or
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install the required Python packages**:
    ```bash
    pip install -r requirements.txt
    ```
    Key packages include `datasets` (for Hugging Face datasets), `openai` (or other LLM provider SDKs), `pandas`, `numpy`, and `sympy` (for MATH evaluation).

### SWE-bench Environment Setup

Evaluating on SWE-bench requires a specific environment due to its reliance on Docker and local repositories.

1.  **Docker**: Ensure Docker is installed and running on your system. This is essential for the SWE-bench harness to create sandboxed environments for executing patches.
2.  **Disk Space & RAM**: SWE-bench can be resource-intensive.
    *   **Disk Space**: A minimum of 120GB is recommended, primarily for Docker images and cloned repositories.
    *   **RAM**: 16GB+ is recommended.
3.  **SWE-bench Repository**: You need to clone the official SWE-bench repository, as the evaluation harness interacts with its structure and testbeds.
    ```bash
    git clone https://github.com/princeton-nlp/SWE-bench.git
    ```
    You will need to set the path to this cloned repository in `config.json` (see `swe_bench_repo_path`).
4.  **Official Setup Instructions**: For detailed setup instructions for SWE-bench, refer to the [Official SWE-bench Setup Documentation](https://github.com/princeton-nlp/SWE-bench/blob/main/docs/setup.md). This includes information on installing necessary tools and building the evaluation environment.
    *   Note: Our framework calls the SWE-bench harness scripts; it does not reimplement the core evaluation environment.

---

## Setup

### API Keys (.env file)
Create a file named `.env` in the root directory with your LLM provider API keys:
```env
OPENROUTER_API_KEY=your_openrouter_key_here
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
# Add other provider keys as needed
```

### Configuration (config.json)
The script `run_test.py` is primarily driven by `config.json`. This file allows you to specify datasets, models, evaluation parameters, and test types.

**Key `config.json` sections:**

*   **`"datasets"`**: A list of dataset names to run experiments on.
    *   Examples: `"main"` (GSM8K-Symbolic main), `"p1"` (GSM8K-Symbolic p1), `"MATH"`, `"AIME"`, `"SWE-bench"`.
    ```json
    "datasets": ["main", "MATH", "AIME", "SWE-bench"],
    ```

*   **`"gsm_types"`**: Specific configurations for GSM-like datasets (e.g., `"gsm-symbolic"`, `"gsm8-std"`).

*   **`"models"`**: Dictionary defining the LLMs to use.
    ```json
    "models": {
        "gpt-4o-mini":{
            "name": "gpt-4o-mini-2024-07-18",
            "provider": "openai",
            "supports_sampling_params": true
        }
        // ... other models
    },
    ```

*   **Benchmark-Specific Parameters**:
    *   **`"MATH_params"`**:
        ```json
        "MATH_params": {
            "hf_id": "nlile/hendrycks-MATH-benchmark",
            "evaluation_type": "exact_match_latex_normalized" 
            // Note: Current eval uses sympy, numerical, and string normalization.
        },
        ```
    *   **`"AIME_params"`**:
        ```json
        "AIME_params": {
            "hf_id": "gneubig/aime-1983-2024",
            "evaluation_type": "exact_string_match", 
            "data_file": "AIME_Dataset_1983_2024.csv", // From the HF dataset
            "split_type": "chronological", // "chronological", "random", or "none"
            "test_years_start": 2018,      // For chronological split
            "max_test_samples": 200        // Max samples for test set
        },
        ```
        For AIME, `"split_type": "chronological"` creates a test set from years `>= test_years_start`.
    *   **`"SWE_bench_params"`**:
        ```json
        "SWE_bench_params": {
            "hf_id": "princeton-nlp/SWE-bench_Verified", // Or other SWE-bench subsets
            "evaluation_type": "swe_bench_harness",
            "docker_image": "swebench/swe-bench-runner:latest", // Default image
            "harness_script_path": "/path/to/your/SWE-bench/swebench/harness/run_evaluation.py", // IMPORTANT: User must set this
            "predictions_file_path": "./swe_bench_temp/model_predictions.jsonl", // Temporary file for predictions
            "swe_bench_repo_path": "/path/to/your/cloned/SWE-bench/", // IMPORTANT: User must set this
            "timeout": 900, // Timeout in seconds for harness execution per instance
            "conda_link": "/path/to/your/miniconda3/envs/swe-bench" // Optional: Path to conda env for SWE-bench
        },
        ```
        **Critical for SWE-bench**:
        - `harness_script_path`: Must point to the `run_evaluation.py` script within your cloned SWE-bench repository.
        - `swe_bench_repo_path`: Must point to the root of your cloned SWE-bench repository.

*   **`"max_reflection_layers"`**: Number of reflection layers (e.g., 3).
*   **`"auto_prompt_model"`**: Model used for meta-prompting (can be `"same"` to use the current evaluation model).
*   **`"run_test"` / `"run_analysis"`**: Booleans to control execution flow.
*   **`"test_types"`**: Booleans to enable/disable baseline, CoT, traditional reflection, and multi-layer reflection tests.

---

## Dataset Preparation

### General
- Datasets for **GSM8K, MATH, AIME, and SWE-bench** are primarily loaded from Hugging Face Datasets using their respective `hf_id` specified in `config.json`.
- Ensure you have internet access when running for the first time to download these datasets.

### AIME Specifics
- The AIME dataset (`gneubig/aime-1983-2024`) contains a "Year" column.
- If `split_type` in `AIME_params` is set to `"chronological"` (default), the `prepare_dataset` function will use problems from `test_years_start` onwards for the test set, up to `max_test_samples`. This allows testing on more recent problems.
- If set to `"random"`, it performs a random sample. If `"none"`, it uses all available data.

### SWE-bench Specifics
- The Hugging Face dataset (e.g., `princeton-nlp/SWE-bench_Verified`) provides task instances (problem statements, instance IDs, etc.).
- **Crucially**, the actual code repositories (testbeds) against which patches are evaluated are part of the cloned SWE-bench repository specified by `swe_bench_repo_path` in `config.json`.
- The SWE-bench harness, when executed by `run_test.py`, uses this local repository to set up the environment for each task instance. Ensure `swe_bench_repo_path` is correctly configured.

---

## Usage

### 1. Running Tests
To run experiments based on your `config.json`:
```bash
python run_test.py
```
- The script iterates through datasets and models specified in `config.json`.
- For each combination, it performs the enabled test types (baseline, CoT, reflection methods).
- Results (including LLM responses, extracted answers, scores, and reflection data) are saved as CSV files in the `results/` directory. Each CSV is typically named `results_{dataset_name}_{benchmark_name}_{model_name}.csv`.

**SWE-bench Execution Note**:
- The current implementation of `evaluate_response` for SWE-bench **simulates** the final harness execution step (`subprocess.run()`).
- It constructs the correct command, prepares prediction files, and simulates a successful harness run by creating a plausible `all_results.json` file.
- To perform **actual SWE-bench evaluation**, you would need to:
    1.  Ensure your environment meets all SWE-bench prerequisites (Docker, cloned repo, conda environment if used).
    2.  Correctly set `harness_script_path` and `swe_bench_repo_path` in `config.json`.
    3.  Uncomment the `subprocess.run()` call within the `evaluate_response` function in `run_test.py` and comment out the simulation block.
- The simulated step allows testing the rest of the pipeline without a full SWE-bench environment.

### 2. Analyzing Results
If `run_analysis` is `true` in `config.json`, `run_test.py` will automatically trigger `analyze_results()` after tests.
- This function reads all CSVs from the `results/` directory.
- It computes summary statistics (accuracy for different methods and reflection layers).
- The aggregated results are saved to `results/summary_results.xlsx`.

To run analysis independently on existing CSV results:
1.  Ensure `run_test` is `false` and `run_analysis` is `true` in `config.json`.
2.  Execute: `python run_test.py`

Results for MATH, AIME, and SWE-bench (based on simulated pass/fail for SWE-bench) will be included in this summary.

### 3. Interactive Demo (Legacy)
An interactive Streamlit demo (`chatbot.py`) is available for GSM8K-style problems. It has not yet been updated to fully support the structured inputs/outputs of MATH, AIME, or SWE-bench but can be used for general exploration of reflection with simpler math problems.
```bash
streamlit run chatbot.py
```

---

## Method Details

### Enhanced Reflection Process
1.  **Initial Attempt**: Generation of an initial solution using Chain-of-Thought (for math problems) or a direct attempt (for SWE-bench, generating a plan and patch).
2.  **Evaluation**:
    *   **MATH**: Complex evaluation using SymPy for symbolic equivalence, numerical comparison, and normalized string matching.
    *   **AIME**: Exact string match of numerical answers.
    *   **GSM-style**: Numerical matching.
    *   **SWE-bench**: (Simulated) execution of the generated patch against the testbed using the SWE-bench harness.
3.  **Adaptive Reflection (Auto-Prompting)**:
    *   If the initial attempt is incorrect, a benchmark-specific meta-prompt (`MATH_META_PROMPT` or `SWE_BENCH_META_PROMPT`) is used to generate a tailored reflection prompt. This guides the LLM to reflect on its previous mistakes and the problem's specifics.
    *   The traditional, static reflection prompt is also available as a fallback or alternative strategy.
4.  **Iterative Refinement**: The LLM attempts to re-solve the problem based on its reflection, up to `max_reflection_layers`.

---

## Supported Benchmarks

- **GSM8K / GSM-Symbolic**: Elementary mathematical reasoning.
  - Loaded via Hugging Face (`apple/GSM-Symbolic`).
  - Evaluation: Numerical matching.
- **MATH**: Advanced high-school and undergraduate mathematics.
  - Loaded via Hugging Face (`nlile/hendrycks-MATH-benchmark`).
  - Evaluation: Multi-stage (SymPy, numerical, string normalization).
- **AIME**: American Invitational Mathematics Examination problems.
  - Loaded via Hugging Face (`gneubig/aime-1983-2024`).
  - Evaluation: Exact numerical string matching.
  - Supports chronological splitting for test set creation.
- **SWE-bench**: Software engineering tasks (bug fixes, feature implementation).
  - Loaded via Hugging Face (`princeton-nlp/SWE-bench_Verified`).
  - Evaluation: (Simulated) execution via the official SWE-bench harness. Requires local SWE-bench repository and Docker for actual runs.

---

## Citation

If you find this repository or our research helpful, please cite:
*(Details of the original paper and any updates/new papers covering the extended benchmarks will be provided here.)*

```
@article{LoureiroSilva2025MultilayerSelfReflection,
  title={Enhancing Multi-Step Mathematical Reasoning in Large Language Models 
         with Multi-Layered Self-Reflection and Auto-Prompting},
  author={Silva, André de Souza Loureiro and Valverde Rebaza, Jorge Carlos},
  year={2025},
  # journal={To be submitted/Published in...},
  # url={Link to paper}
}
```

---

Thank you for your interest! For questions, suggestions, or contributions, please open an issue or submit a pull request.
