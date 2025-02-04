# Multi-Layered Self-Reflection with Auto-Prompting

This repository provides a reference implementation of the **Multi-Layered Self-Reflection** approach with **Auto-Prompting** for improving multi-step mathematical reasoning in Large Language Models (LLMs). The accompanying [paper](#citation) demonstrates how this method significantly boosts accuracy on both the GSM8K and GSM-Symbolic benchmarks across several LLMs (GPT-4o-mini, Llama 3.1–8B, Llama 3.1–70B), outperforming standard Chain-of-Thought (CoT) prompting and single-pass Self-Reflection methods.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Running the Interactive Demo](#1-running-the-interactive-demo)
  - [2. Running Tests](#2-running-tests)
  - [3. Analyzing Results](#3-analyzing-results)
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

---

## Features

- **Automated Reflection Layers**: Iteratively prompt the model to produce refined solutions after detecting errors.
- **Auto-Prompt (Meta-Prompt) Generation**: The model adapts its reflection prompt based on the problem's structure and its prior errors, rather than relying on a single, static reflection template.
- **Multiple LLM Support**: Easily switch between different models (GPT-4o-mini, Llama 3.1–8B, Llama 3.1–70B, etc.) by updating the config.
- **Benchmark Integration**: Out-of-the-box usage for GSM8K and GSM-Symbolic (main, p1, p2) datasets.
- **Interactive Demo Interface**: A Streamlit-based web interface for testing the method with different models and reflection layers.

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

4. **Obtain your OpenRouter API key** (or another valid LLM provider key). You'll need this key to authenticate model calls.

---

## Setup

1. **.env file**  
   Create a file named `.env` in the root directory with API keys:
   ```bash
   OPENROUTER_API_KEY=your_key_here
   OPENAI_API_KEY=your_openai_key
   DEEPSEEK_API_KEY=your_deepseek_key
   ```

2. **config.json**  
   The script reads parameters from `config.json`. A typical structure might look like:
   ```json
   {
    "datasets": ["main", "p1", "p2"],
    "gsm_types": ["gsm-symbolic", "gsm8-std"],
    "models": {
         "gpt-4o-2024-11-20": {
            "input_token_cost": 0.0000025,
            "output_token_cost": 0.00001
         },
         "llama-3.1-70b-instruct":{
            "name": "meta-llama/llama-3.1-70b-instruct",
            "provider": "openrouter",
            "supports_sampling_params": true
        },
        ...
    },
    "max_reflection_layers": 3,
    "auto_prompt_model": "same",
    "run_test": true,
    "run_analysis": true,
    "test_types": {
        "run_base": true,
        "run_cot": true,
        "run_traditional_self_reflection": true,
        "run_multi_layer_self_reflection": true

    }
   }
   ```

---

## Usage

### 1. Running the Interactive Demo

To launch the web interface:
```bash
streamlit run chatbot.py
```

The interface provides:
- Model selection from your configured providers
- Toggle between GSM8K format (numerical answers) and free-form responses
- Real-time visualization of:
  - Chain-of-Thought reasoning
  - Multiple reflection layers
  - Answer extraction
- Interactive reflection layer generation

### 2. Running Tests

To reproduce the paper's results and run the main experiment:
```bash
python run_test.py
```
- The script will load each dataset specified in `config.json`, sample data, and query each model in the `models` list.
- It will generate CSV logs inside a `results/` directory (one CSV per dataset/model combination).
- This will replicate the experimental setup and evaluation methodology used in the paper.

### 3. Analyzing Results

If you have `run_analysis` set to `true` in `config.json`, the script will automatically:
- Load all CSV result files from `results/`
- Aggregate them into a single summary table
- Save an Excel file named `summary_results.xlsx`

Alternatively, you can run the analysis step independently after collecting data:
```bash
# Make sure "run_test" is false and "run_analysis" is true
python run_test.py
```
This will parse the existing CSVs in `results/` and produce the summary Excel.

### Cost Calculation

We provide detailed cost tracking through `cost_calculator.py`:
```bash
python cost_calculator.py
```
- Token counting using TikToken
- Model-specific pricing from `model_pricing.json`
- Separates costs for:
  - Initial CoT prompts
  - Reflection generations
  - Re-answer attempts
- Produces Excel reports with per-model breakdowns

---

## Method Details

### Enhanced Reflection Process
1. **Chain-of-Thought (CoT) Baseline**: Initial reasoning with 8-shot prompting
2. **Error Detection**: Automatic answer extraction and validation
3. **Adaptive Reflection**:
   - *Traditional Method*: Fixed reflection template
   - *Multi-layer Method*: Auto-generated prompts via meta-prompting
4. **Iterative Refinement**: Up to 3 reflection layers with:
   - Customized instructions
   - Error type classification
   - Step-by-step solution regeneration

## Benchmarks

Supported models in `model_pricing.json`:
```json
{
    "gpt-4o-2024-11-20": {
        "input_token_cost": 0.0000025,
        "output_token_cost": 0.00001
    },
    // ... other models ...
}
```

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
