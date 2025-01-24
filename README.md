# Multi-Layered Self-Reflection with Auto-Prompting

This repository provides a reference implementation of the **Multi-Layered Self-Reflection** approach with **Auto-Prompting** for improving multi-step mathematical reasoning in Large Language Models (LLMs). The accompanying [paper](#citation) demonstrates how this method significantly boosts accuracy on both the GSM8K and GSM-Symbolic benchmarks across several LLMs (GPT-4o-mini, Llama 3.1–8B, Llama 3.1–70B), outperforming standard Chain-of-Thought (CoT) prompting and single-pass Self-Reflection methods.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Running Tests](#1-running-tests)
  - [2. Analyzing Results](#2-analyzing-results)
- [Method Details](#method-details)
- [Benchmarks](#benchmarks)
- [Citation](#citation)
- [License](#license)

---

## Overview

The core idea of **multi-layered Self-Reflection** is to allow the model to:
1. Generate an initial answer (often via Chain-of-Thought).
2. Reflect on incorrect or incomplete steps by **auto-generating** (meta-prompting) a new reflection prompt tailored to the specific mistakes and problem type.
3. Iteratively refine its solution over multiple layers until reaching a correct conclusion or hitting a maximum number of attempts.

This repository contains:
- **Python scripts** demonstrating how to query different LLMs through the [OpenRouter API](https://openrouter.ai/) (or any other API that follows a similar interface).
- **Configuration files** (`config.json`) that control dataset choice, reflection layers, model specifications, etc.
- **Utilities** for generating prompts, analyzing outputs, and saving results.

---

## Features

- **Automated Reflection Layers**: Iteratively prompt the model to produce refined solutions after detecting errors.
- **Auto-Prompt (Meta-Prompt) Generation**: The model adapts its reflection prompt based on the problem’s structure and its prior errors, rather than relying on a single, static reflection template.
- **Multiple LLM Support**: Easily switch between different models (GPT-4o-mini, Llama 3.1–8B, Llama 3.1–70B, etc.) by updating the config.
- **Benchmark Integration**: Out-of-the-box usage for GSM8K and GSM-Symbolic (main, p1, p2) datasets.

---

## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Linux or Mac
   # or
   venv\Scripts\activate      # On Windows
   ```

3. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

   > Note: This repository uses the `datasets` library, `openai` for API calls, `pandas` for data manipulation, etc.

4. **Obtain your OpenRouter API key** (or another valid LLM provider key). You’ll need this key to authenticate model calls.

---

## Setup

1. **.env file**  
   Create a file named `.env` in the root directory and set your OpenRouter API key:
   ```bash
   OPENROUTER_API_KEY=your_key_here
   ```
   Alternatively, you can set the environment variable `OPENROUTER_API_KEY` in your system.

2. **config.json**  
   The script reads parameters from `config.json`. A typical structure might look like:
   ```json
   {
     "run_test": true,
     "run_analysis": false,
     "datasets": ["main", "p1", "p2"],
     "gsm_types": ["gsm8-std", "gsm-symbolic"],
     "models": ["meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.1-70b-instruct"],
     "max_reflection_layers": 3,
     "auto_prompt_model": "same"
   }
   ```
   - `run_test`: whether to run the main experiment.
   - `run_analysis`: whether to parse and summarize results into an Excel file.
   - `datasets`: which GSM-Symbolic subsets to use (`main`, `p1`, `p2`).
   - `gsm_types`: which dataset format to load (`gsm8-std` or `gsm-symbolic`).
   - `models`: list of models to test.
   - `max_reflection_layers`: how many multi-layer reflections to allow.
   - `auto_prompt_model`: which model to use for generating auto-prompts. `"same"` means the same model does both tasks.

---

## Usage

### 1. Running Tests

To run the main experiment:
```bash
python main.py
```
- The script will load each dataset specified in `config.json`, sample data, and query each model in the `models` list.
- It will generate CSV logs inside a `results/` directory (one CSV per dataset/model combination).

### 2. Analyzing Results

If you have `run_analysis` set to `true` in `config.json`, the script will automatically:
- Load all CSV result files from `results/`
- Aggregate them into a single summary table
- Save an Excel file named `summary_results.xlsx`

Alternatively, you can run the analysis step independently after collecting data:
```bash
# Make sure "run_test" is false and "run_analysis" is true
python main.py
```
This will parse the existing CSVs in `results/` and produce the summary Excel.

---

## Method Details

**Multi-Layer Reflection**:
1. **CoT Answer**: Prompt the model with standard chain-of-thought to get an initial solution.
2. **Check**: If the solution is incorrect, generate a custom reflection prompt (auto-prompt) using meta-prompting.
3. **Reflection**: The model explains why the attempt might be wrong, identifies potential pitfalls, and re-solves the problem.
4. **Repeat**: Continue until the answer is correct or you hit the `max_reflection_layers` limit.

This approach is described in detail in our paper. It significantly mitigates the drop in performance on symbolically modified or extended problems (p1, p2 in GSM-Symbolic).

---

## Benchmarks

We use:

- **[GSM8K](https://github.com/openai/grade-school-math)**: 8.5k standard math word problems.
- **[GSM-Symbolic](https://arxiv.org/abs/2410.05229)**: A version of GSM8K with altered numerical values, extra clauses, and symbolic twists. Includes three variants:
  - `main`: Substitutes names/values.
  - `p1`: Adds 1 extra clause of complexity.
  - `p2`: Adds 2 extra clauses of complexity.

The code snippet demonstrates how to automatically load `main`, `p1`, or `p2` splits via the Hugging Face `datasets` library.

---

## Citation

If you find this repository or our paper helpful in your research, please cite it as:

```
@article{LoureiroSilva2025MultilayerSelfReflection,
  title={Enhancing Multi-Step Mathematical Reasoning in Large Language Models 
         with Multi-Layered Self-Reflection and Auto-Prompting},
  author={Silva, André de Souza Loureiro and Valverde Rebaza, Jorge Carlos},
  year={2025},
}
```

---



Thank you for your interest in **Multi-Layered Self-Reflection with Auto-Prompting**! If you have any questions, suggestions, or feedback, feel free to open an issue or submit a pull request.
