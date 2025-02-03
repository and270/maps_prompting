import json
import tiktoken
import pandas as pd
from pathlib import Path

def count_tokens(text, model_name):
    # Handle NaN/None values and ensure string type
    if pd.isna(text):
        return 0
        
    text = str(text)
    
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Allow all special tokens and treat them as normal text
    return len(encoding.encode(text, disallowed_special=()))

def calculate_costs(results_dir="results/multilayer_auto_prompt_self_reflection", pricing_file="model_pricing.json"):
    # Load pricing data
    with open(pricing_file) as f:
        pricing = json.load(f)
    
    # Process all CSV files in run folders
    cost_data = []
    for csv_path in Path(results_dir).rglob("run-*/results_*.csv"):
        df = pd.read_csv(csv_path)
        
        # Extract metadata from filename
        filename_parts = csv_path.stem.split("_")
        dataset = filename_parts[1]
        gsm_type = filename_parts[2]
        model = filename_parts[-1]
        run_number = csv_path.parent.name.split('-')[-1]

        # Process each row
        for _, row in df.iterrows():
            model_pricing = pricing.get(model, {})
            input_cost = model_pricing.get('input_token_cost', 0)
            output_cost = model_pricing.get('output_token_cost', 0)

            # CoT costs
            cot_input = count_tokens(generate_cot_prompt(row['question']), model)
            cot_output = count_tokens(row['cot_full_response'], model)
            
            # Reflection costs (if any)
            refl_input = count_tokens(row.get('reflection_prompt', ''), model)
            refl_output = count_tokens(row.get('reflection_full_response', ''), model)
            
            # Re-answer costs (if any)
            reanswer_input = 0
            if pd.notna(row.get('reflection_full_response')) and pd.notna(row.get('cot_response')):
                reanswer_prompt = generate_reanswer_prompt(
                    row['question'],
                    row['cot_response'],
                    row['reflection_full_response']
                )
                reanswer_input = count_tokens(reanswer_prompt, model)
            reanswer_output = count_tokens(row.get('reflection_response', ''), model)

            cost_data.append({
                "run": run_number,
                "model": model,
                "dataset": dataset,
                "gsm_type": gsm_type,
                "cot_input": cot_input,
                "cot_output": cot_output,
                "refl_input": refl_input,
                "refl_output": refl_output,
                "reanswer_input": reanswer_input,
                "reanswer_output": reanswer_output,
                "total_cost": (
                    (cot_input + refl_input + reanswer_input) * input_cost +
                    (cot_output + refl_output + reanswer_output) * output_cost
                )
            })

    # Aggregate and calculate means
    if cost_data:
        cost_df = pd.DataFrame(cost_data)
        
        # Aggregate per run
        run_totals = cost_df.groupby(['run', 'model', 'dataset', 'gsm_type']).agg({
            'cot_input': 'sum',
            'cot_output': 'sum',
            'refl_input': 'sum',
            'refl_output': 'sum',
            'reanswer_input': 'sum',
            'reanswer_output': 'sum',
            'total_cost': 'sum'
        }).reset_index()

        # Calculate means across runs
        summary = run_totals.groupby(['model', 'dataset', 'gsm_type']).agg({
            'cot_input': 'mean',
            'cot_output': 'mean',
            'refl_input': 'mean',
            'refl_output': 'mean',
            'reanswer_input': 'mean',
            'reanswer_output': 'mean',
            'total_cost': 'mean'
        }).reset_index()

        # Save results
        summary.to_excel("model_costs_summary.xlsx", index=False)
        return summary

def calculate_reasoning_model_costs(results_dir="results/multilayer_auto_prompt_self_reflection", pricing_file="model_pricing.json"):
    """Calculate costs for reasoning model experiments (CoT only)"""
    # Load pricing data
    with open(pricing_file) as f:
        pricing = json.load(f)
    
    cost_data = []
    for csv_path in Path(results_dir).rglob("run-*/o3-mini-measured/*.csv"):
        df = pd.read_csv(csv_path)
        
        # Extract metadata from path
        path_parts = csv_path.parts
        run_number = [p for p in path_parts if p.startswith('run-')][0].split('-')[-1]
        model = path_parts[-2]  # Get model from parent folder
        
        # Extract dataset and GSM type from filename
        filename_parts = csv_path.stem.split("_")
        dataset = filename_parts[1]
        gsm_type = filename_parts[2]

        # Process each row
        for _, row in df.iterrows():
            model_pricing = pricing.get(model, {})
            input_cost = model_pricing.get('input_token_cost', 0)
            output_cost = model_pricing.get('output_token_cost', 0)

            # CoT costs only
            cot_input = count_tokens(generate_cot_prompt(row['question']), model)
            cot_output = count_tokens(row['cot_full_response'], model)
            
            total_cost = (cot_input * input_cost) + (cot_output * output_cost)
            
            cost_data.append({
                "run": run_number,
                "model": model,
                "dataset": dataset,
                "gsm_type": gsm_type,
                "cot_input": cot_input,
                "cot_output": cot_output,
                "total_cost": total_cost
            })

    # Aggregate and calculate means
    if cost_data:
        cost_df = pd.DataFrame(cost_data)
        
        # Aggregate per run
        run_totals = cost_df.groupby(['run', 'model', 'dataset', 'gsm_type']).agg({
            'cot_input': 'sum',
            'cot_output': 'sum',
            'total_cost': 'sum'
        }).reset_index()

        # Calculate means across runs
        summary = run_totals.groupby(['model', 'dataset', 'gsm_type']).agg({
            'cot_input': 'mean',
            'cot_output': 'mean',
            'total_cost': 'mean'
        }).reset_index()

        # Save results
        summary.to_excel("reasoning_model_costs_summary.xlsx", index=False)
        return summary
    return pd.DataFrame()

# Add these helper functions
def generate_cot_prompt(question):
    return f"""{EIGHT_SHOT_EXAMPLES}
Now, look at this question:
Q: {question}
A: Let's think step by step..."""

def generate_reanswer_prompt(question, prev_answer, reflection):
    return f"""{EIGHT_SHOT_EXAMPLES}

Now, look at this question:
Question: {question}
Your initial answer was: {prev_answer}
You previously answered this question incorrectly.
Then you reflected on the problem, your solution, and the correct answer::
Reflection: {reflection}

Provide your corrected reasoning and answer in the examples format.
"""

# Add EIGHT_SHOT_EXAMPLES constant (copy from run_test.py)
EIGHT_SHOT_EXAMPLES = """[COPY THE EXACT CONTENT FROM run_test.py HERE]"""

if __name__ == "__main__":
    # Run both calculations
    main_summary = calculate_costs()
    reasoning_summary = calculate_reasoning_model_costs() 