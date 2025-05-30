# Multi-Layered Self-Reflection with Auto-Prompting for LLMs

This repository provides a reference implementation of the **Multi-Layered Self-Reflection** approach with **Auto-Prompting** for improving multi-step reasoning in Large Language Models (LLMs). Initially demonstrated on mathematical reasoning benchmarks like GSM8K, the framework has been extended to evaluate performance on more complex domains, including the **MATH** (Hendrycks Mathematics) and **AIME** (American Invitational Mathematics Examination) datasets.

The accompanying [paper](#citation) (details to be updated with new benchmark results) demonstrates how this method can significantly boost accuracy across various benchmarks and LLMs.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [General Dependencies](#general-dependencies)
- [Setup](#setup)
  - [API Keys (.env file)](#api-keys-env-file)
  - [Configuration (config.json)](#configuration-configjson)
- [Dataset Preparation](#dataset-preparation)
  - [General](#general)
  - [AIME Specifics](#aime-specifics)
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

---

## Features

- **Automated Reflection Layers**: Iteratively prompt the model to produce refined solutions.
- **Auto-Prompt (Meta-Prompt) Generation**: The model adapts its reflection prompt based on the problem's structure, its prior errors, and the benchmark type.
- **Multiple LLM Support**: Easily switch between different models by updating the `config.json`.
- **Benchmark Integration**: Out-of-the-box support for GSM8K, GSM-Symbolic, MATH and AIME datasets.
- **Benchmark-Specific Prompts**: Tailored few-shot examples and meta-prompts for each benchmark.
- **Extensible Evaluation**: Modular design for adding new benchmarks and evaluation methods.

---

## Installation

### General Dependencies

1.  **Clone** this repository:
    ```bash
    git clone https://github.com/and270/maps_prompting.git
    cd maps_prompting
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
    *   Examples: `"main"` (GSM8K-Symbolic main), `"p1"` (GSM8K-Symbolic p1), `"MATH"`, `"AIME"`.
    ```json
    "datasets": ["main", "MATH", "AIME"],
    ```

*   **`"gsm_types"`**: Specific configurations for GSM-like datasets (e.g., `"gsm-symbolic"`, `"gsm8-std"`).

*   **`"models"`**: Dictionary defining the LLMs to use.
    ```json
    "models": {
        "gpt-4o-mini":{
            "name": "gpt-4o-mini-2024-07-18",
            "provider": "openai",
            "api_key_env": "OPENAI_API_KEY",
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


*   **`"max_reflection_layers"`**: Number of reflection layers (e.g., 3).
*   **`"auto_prompt_model"`**: Model used for meta-prompting (can be `"same"` to use the current evaluation model).
*   **`"run_test"` / `"run_analysis"`**: Booleans to control execution flow.
*   **`"test_types"`**: Booleans to enable/disable baseline, CoT, traditional reflection, and multi-layer reflection tests.

---

## Dataset Preparation

### General
- Datasets for **GSM8K, MATH, AIME ** are primarily loaded from Hugging Face Datasets using their respective `hf_id` specified in `config.json`.
- Ensure you have internet access when running for the first time to download these datasets.

### AIME Specifics
- The AIME dataset (`gneubig/aime-1983-2024`) contains a "Year" column.
- If `split_type` in `AIME_params` is set to `"chronological"` (default), the `prepare_dataset` function will use problems from `test_years_start` onwards for the test set, up to `max_test_samples`. This allows testing on more recent problems.
- If set to `"random"`, it performs a random sample. If `"none"`, it uses all available data.

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

### 2. Analyzing Results
If `run_analysis` is `true` in `config.json`, `run_test.py` will automatically trigger `analyze_results()` after tests.
- This function reads all CSVs from the `results/` directory.
- It computes summary statistics (accuracy for different methods and reflection layers).
- The aggregated results are saved to `results/summary_results.xlsx`.

To run analysis independently on existing CSV results:
1.  Ensure `run_test` is `false` and `run_analysis` is `true` in `config.json`.
2.  Execute: `python run_test.py`

Results for MATH and AIME  will be included in this summary.

### 3. Interactive Demo (Legacy)
An interactive Streamlit demo (`chatbot.py`) is available for GSM8K-style problems. It has not yet been updated to fully support the structured inputs/outputs of MATH or AIME but can be used for general exploration of reflection with simpler math problems.
```bash
streamlit run chatbot.py
```

---

## Method Details

### Enhanced Reflection Process
1.  **Initial Attempt**: Generation of an initial solution using Chain-of-Thought (for math problems) or a direct attempt.
2.  **Evaluation**:
    *   **MATH**: Complex evaluation using SymPy for symbolic equivalence, numerical comparison, and normalized string matching.
    *   **AIME**: Exact string match of numerical answers.
    *   **GSM-style**: Numerical matching.
3.  **Adaptive Reflection (Auto-Prompting)**:
    *   If the initial attempt is incorrect, a meta-prompt is used to generate a tailored reflection prompt. This guides the LLM to reflect on its previous mistakes and the problem's specifics.
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
