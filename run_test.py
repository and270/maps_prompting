# Import necessary libraries
import pandas as pd
from datasets import load_dataset
import os
import json
from concurrent.futures import ThreadPoolExecutor

from llms_api import query_model
from methods import extract_answer_gsm_format, generate_auto_reflection_auto_adapt_prompt, generate_auto_reflection_traditional_prompt, generate_cot_prompt, generate_reanswer_prompt

def prepare_dataset(dataset_type='main'):
    print(f"Loading the {dataset_type} dataset...")
    ds = load_dataset("apple/GSM-Symbolic", dataset_type)
    df = pd.DataFrame(ds['test'])
    # Select a random sample by "original_id"
    sample = df.groupby("original_id").sample(n=1, random_state=42)
    print(f"Dataset {dataset_type} loaded. Size: {len(sample)}.")
    return sample

def evaluate_response(response, expected_answer):
    try:
        if response is None:
            return 0
        return int(float(response) == float(expected_answer))
    except Exception as e:
        print(f"Error in evaluate_response: {e}")
        return 0

# Main function to run experiments
def run_gsm8(sample, api_key, config, model, dataset_name, gsm_type, api_provider, supports_sampling_params):
    results = []
    for idx, row in sample.iterrows():
        question = row["original_question"] if gsm_type == "gsm8-std" else row["question"]
        expected_answer = extract_answer_gsm_format(row["original_answer"]) if gsm_type == "gsm8-std" else extract_answer_gsm_format(row["answer"])
        print("Expected answer: ", expected_answer)
        
        # Initialize variables with defaults
        base_full_response = None
        base_response = None
        base_score = 0
        cot_full_response = None
        cot_response = None
        cot_score = 0

        # Baseline
        if config["test_types"]["run_base"]:
            base_prompt = question
            base_full_response = query_model(api_key, base_prompt, model, supports_sampling_params, api_provider)
            base_response = extract_answer_gsm_format(base_full_response) if base_full_response else None
            base_score = evaluate_response(base_response, expected_answer)
            print(f"Base response: {base_response} - Score: {base_score}")

        # CoT (will run if either CoT is enabled or any reflection method is enabled)
        if (config["test_types"]["run_cot"] or 
            config["test_types"]["run_traditional_self_reflection"] or 
            config["test_types"]["run_multi_layer_self_reflection"]):
            cot_prompt = generate_cot_prompt(question)
            cot_full_response = query_model(api_key, cot_prompt, model, supports_sampling_params, api_provider)
            cot_response = extract_answer_gsm_format(cot_full_response) if cot_full_response else None
            cot_score = evaluate_response(cot_response, expected_answer)
            print(f"COT response: {cot_response} - Score: {cot_score}")

        # Traditional Self-Reflection
        reflection_data_traditional_method = []
        if config["test_types"]["run_traditional_self_reflection"]:
            if cot_score == 0:
                traditional_reflection_prompt = generate_auto_reflection_traditional_prompt(question, cot_full_response)
                traditional_reflection_response = query_model(api_key, traditional_reflection_prompt, model, supports_sampling_params, api_provider)
                traditional_reanswer_prompt = generate_reanswer_prompt(question, cot_response, traditional_reflection_response)
                traditional_reanswer_response = query_model(api_key, traditional_reanswer_prompt, model, supports_sampling_params, api_provider)
                traditional_answer = extract_answer_gsm_format(traditional_reanswer_response)
                traditional_score = evaluate_response(traditional_answer, expected_answer)
                reflection_data_traditional_method.append({
                    "layer": None,
                    "score": traditional_score,
                    "response": traditional_answer,
                    "reflection_prompt": traditional_reflection_prompt,
                    "full_response": traditional_reflection_response,
                    "auto_prompt_used": "traditional"
                })
            else:
                reflection_data_traditional_method.append({
                    "layer": None,
                    "score": cot_score,
                    "response": None,
                    "reflection_prompt": None,
                    "full_response": None,
                    "auto_prompt_used": None
                })

        # Multi-layer Self-Reflection
        reflection_data_multi_layer__method = []
        if config["test_types"]["run_multi_layer_self_reflection"]:
            # If the CoT answer is already correct, record it without further reflection.
            if cot_score == 1:
                reflection_data_multi_layer__method.append({
                    "layer": 0,
                    "score": cot_score,
                    "response": cot_response,
                    "reflection_prompt": None,
                    "full_response": None,
                    "auto_prompt_used": None
                })
            else:
                previous_incorrect_answers = []
                max_layers = config["max_reflection_layers"]
                current_answer = cot_response
                current_score = cot_score
                auto_prompt_model = config["auto_prompt_model"]

                for layer in range(max_layers):
                    previous_incorrect_answers.append(current_answer)
                    auto_prompt_used = generate_auto_reflection_auto_adapt_prompt(
                        question, previous_incorrect_answers, auto_prompt_model, api_key, api_provider
                    )
                    reflection_prompt = auto_prompt_used
                    reflection_full_response = query_model(api_key, reflection_prompt, model, supports_sampling_params, api_provider)
                    reanswer_prompt = generate_reanswer_prompt(question, current_answer, reflection_full_response)
                    reanswer_full_response = query_model(api_key, reanswer_prompt, model, supports_sampling_params, api_provider)
                    current_answer = extract_answer_gsm_format(reanswer_full_response)
                    current_score = evaluate_response(current_answer, expected_answer)
                    reflection_data_multi_layer__method.append({
                        "layer": layer + 1,
                        "score": current_score,
                        "response": current_answer,
                        "reflection_prompt": reflection_prompt,
                        "full_response": reflection_full_response,
                        "auto_prompt_used": auto_prompt_used
                    })
                    print(f"Reflection (layer {layer + 1}) response: {current_answer} - Score: {current_score}")
                    if current_score == 1:
                        break

        results.append({
            "dataset": dataset_name,
            "gsm_type": gsm_type,
            "model": model,
            "question": question,
            "expected_answer": expected_answer,
            "base_full_response": base_full_response,
            "base_response": base_response,
            "base_score": base_score,
            "cot_full_response": cot_full_response,
            "cot_response": cot_response,
            "cot_score": cot_score,
            'traditional_reflection_data': reflection_data_traditional_method,
            "reflection_data": reflection_data_multi_layer__method
        })
    return pd.DataFrame(results)

def save_results(results_df, filename="experiment_results.csv"):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    expanded_results = []
    for _, row in results_df.iterrows():
        base_row = {
            'dataset': row['dataset'],
            'gsm_type': row['gsm_type'],
            'model': row['model'],
            'question': row['question'],
            'expected_answer': row['expected_answer'],
            'base_full_response': row.get('base_full_response', None),
            'base_response': row.get('base_response', None),
            'base_score': row.get('base_score', 0),
            'cot_full_response': row.get('cot_full_response', None),
            'cot_response': row.get('cot_response', None),
            'cot_score': row.get('cot_score', 0),
            'reflection_layer': None,
            'reflection_score': None,
            'reflection_response': None,
            'reflection_prompt': None,
            'reflection_full_response': None,
            'auto_prompt_used': None,
            'traditional_reflection_score': None,
            'traditional_reflection_response': None,
            'traditional_reflection_prompt': None,
            'traditional_reflection_full_response': None,
            'traditional_auto_prompt_used': None
        }
        # Add base row
        expanded_results.append(base_row.copy())
        # Add traditional reflection data if it exists
        if row.get('traditional_reflection_data') and len(row['traditional_reflection_data']) > 0:
            traditional_data = row['traditional_reflection_data'][0]
            trad_row = base_row.copy()
            if traditional_data:
                trad_row.update({
                    'traditional_reflection_score': traditional_data.get('score', None),
                    'traditional_reflection_response': traditional_data.get('response', None),
                    'traditional_reflection_prompt': traditional_data.get('reflection_prompt', None),
                    'traditional_reflection_full_response': traditional_data.get('full_response', None),
                    'traditional_auto_prompt_used': traditional_data.get('auto_prompt_used', None)
                })
                expanded_results.append(trad_row)
        # Add multi-layer reflection data if it exists
        if row.get('reflection_data'):
            for layer_data in row['reflection_data']:
                if layer_data:
                    reflection_row = base_row.copy()
                    reflection_row.update({
                        'reflection_layer': layer_data.get('layer', None),
                        'reflection_score': layer_data.get('score', None),
                        'reflection_response': layer_data.get('response', None),
                        'reflection_prompt': layer_data.get('reflection_prompt', None),
                        'reflection_full_response': layer_data.get('full_response', None),
                        'auto_prompt_used': layer_data.get('auto_prompt_used', None)
                    })
                    expanded_results.append(reflection_row)
    try:
        expanded_df = pd.DataFrame(expanded_results)
        expanded_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        fallback_filename = filename.replace('.csv', '.json')
        with open(fallback_filename, 'w') as f:
            json.dump(expanded_results, f, indent=2)
        print(f"Results saved to fallback JSON file: {fallback_filename}")

# Function to load configuration from config.json
def load_config():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    return config

def worker(args):
    dataset_name, gsm_type, model_info, sample, API_KEY, config = args
    try:
        local_config = config.copy()
        # Extract model info
        model_name = model_info['name']
        api_provider = model_info['provider']
        supports_sampling_params = model_info.get('supports_sampling_params', True)
        # If auto_prompt_model is set to "same", use the current model also for the auto prompt
        if local_config['auto_prompt_model'] == "same":
            local_config['auto_prompt_model'] = model_name
        print(f"\nExecuting test with:")
        print(f"Dataset: {dataset_name}")
        print(f"GSM type: {gsm_type}")
        print(f"Model: {model_name}")
        print(f"Provider: {api_provider}")
        print(f"Supports sampling params: {supports_sampling_params}")
        print(f"Max Reflection Layers: {local_config['max_reflection_layers']}")
        print(f"Auto Prompt Model: {local_config['auto_prompt_model']}")
        # Get the correct API key based on the provider specified in config
        api_key_map = {
            'openrouter': os.getenv("OPENROUTER_API_KEY"),
            'openai': os.getenv("OPENAI_API_KEY"),
            'deepseek': os.getenv("DEEPSEEK_API_KEY")
        }
        API_KEY = api_key_map.get(api_provider)
        if not API_KEY:
            raise ValueError(f"No API key found for provider: {api_provider}")
        safe_model_name = model_name.replace("/", "_")
        results_df = run_gsm8(sample, API_KEY, local_config, model_name, dataset_name, gsm_type, api_provider, supports_sampling_params)
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        safe_dataset_name = dataset_name.replace("/", "_")
        safe_gsm_type = gsm_type.replace("-", "_")
        save_results(results_df, f"{results_dir}/results_{safe_dataset_name}_{safe_gsm_type}_{safe_model_name}.csv")
    except Exception as e:
        print(f"Error in worker thread: {e}")

def analyze_results(results_dir="results"):
    import pandas as pd
    import os

    all_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
    if not all_files:
        print("No CSV results files found in the directory.")
        return

    # Read all CSVs and combine them
    df_list = []
    for file in all_files:
        file_path = os.path.join(results_dir, file)
        temp_df = pd.read_csv(file_path)
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)

    # Convert key columns to numeric
    numeric_cols = [
        "base_score", "cot_score",
        "traditional_reflection_score",
        "reflection_layer", "reflection_score"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Group-level aggregator for each question
    def combine_question_data(group):
        # Base score + CoT score are typically 0 or 1 per question.
        # We use .iloc[0] because base_score/cot_score are repeated on every row for that question anyway.
        base_score = group["base_score"].iloc[0]
        cot_score = group["cot_score"].iloc[0]

        # Traditional reflection
        trad_scores = group["traditional_reflection_score"].dropna()
        trad_attempted = len(trad_scores) > 0
        trad_success = trad_scores.max() if trad_attempted else 0

        # For multi-layer reflection, figure out if each layer was correct (0 or 1).
        # Use .max() so if there's more than one row for a layer, we pick up if any row was correct.
        reflection_layer_1_score = group.loc[group["reflection_layer"] == 1, "reflection_score"].max() if (group["reflection_layer"] == 1).any() else 0
        reflection_layer_2_score = group.loc[group["reflection_layer"] == 2, "reflection_score"].max() if (group["reflection_layer"] == 2).any() else 0
        reflection_layer_3_score = group.loc[group["reflection_layer"] == 3, "reflection_score"].max() if (group["reflection_layer"] == 3).any() else 0

        # Now define two new columns:
        # 1. Score if we only allow reflection layer 1 (i.e. CoT or layer 1).
        score_up_to_layer_1 = max(cot_score, reflection_layer_1_score)

        # 2. Score if we allow up to layer 3 (i.e. CoT or any reflection layer).
        score_up_to_layer_3 = max(
            cot_score,
            reflection_layer_1_score,
            reflection_layer_2_score,
            reflection_layer_3_score
        )

        return pd.Series({
            "base_score": base_score,
            "cot_score": cot_score,

            "trad_attempted": int(trad_attempted),
            "trad_success": trad_success,

            "reflection_layer_1_score": reflection_layer_1_score,
            "reflection_layer_2_score": reflection_layer_2_score,
            "reflection_layer_3_score": reflection_layer_3_score,

            # New columns:
            "score_up_to_layer_1": score_up_to_layer_1,
            "score_up_to_layer_3": score_up_to_layer_3
        })

    # Aggregate at the question level
    pivot_df = (
        df
        .groupby(["dataset", "gsm_type", "model", "question"], as_index=False)
        .apply(combine_question_data)
    )

    # Summarize at dataset/gsm_type/model level
    summary_df = (
        pivot_df
        .groupby(["dataset", "gsm_type", "model"], as_index=False)
        .agg({
            "base_score": "sum",
            "cot_score": "sum",
            "trad_attempted": "sum",
            "trad_success": "sum",
            "reflection_layer_1_score": "sum",
            "reflection_layer_2_score": "sum",
            "reflection_layer_3_score": "sum",
            "score_up_to_layer_1": "sum",  # sum across questions
            "score_up_to_layer_3": "sum",  # sum across questions
            "question": "count"
        })
        .rename(columns={"question": "total_questions"})
    )

    # Basic accuracies
    summary_df["base_accuracy"] = summary_df["base_score"] / summary_df["total_questions"]
    summary_df["cot_accuracy"] = summary_df["cot_score"] / summary_df["total_questions"]

    # Traditional reflection accuracy (depending on your definition)
    summary_df["traditional_reflection_accuracy"] = (
        summary_df["trad_success"] / summary_df["total_questions"]
    )

    # Reflection layers
    summary_df["reflection_layer_1_accuracy"] = (
        summary_df["reflection_layer_1_score"] / summary_df["total_questions"]
    )
    summary_df["reflection_layer_2_accuracy"] = (
        summary_df["reflection_layer_2_score"] / summary_df["total_questions"]
    )
    summary_df["reflection_layer_3_accuracy"] = (
        summary_df["reflection_layer_3_score"] / summary_df["total_questions"]
    )

    # Now the two columns you requested as sums: 
    #  "score_up_to_layer_1" (the number of questions correct by CoT or layer1)
    #  "score_up_to_layer_3" (the number of questions correct by CoT or layer1/2/3)
    # If you also want them as *accuracies*, do:
    summary_df["accuracy_up_to_layer_1"] = (
        summary_df["score_up_to_layer_1"] / summary_df["total_questions"]
    )
    summary_df["accuracy_up_to_layer_3"] = (
        summary_df["score_up_to_layer_3"] / summary_df["total_questions"]
    )

    # Save results to Excel
    output_path = os.path.join(results_dir, "summary_results.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

    print("Saved summarized results to:", output_path)
    return summary_df




def main():
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    config = load_config()
    run_test = config.get('run_test', False)
    run_analysis = config.get('run_analysis', False)
    if run_test:
        datasets = config.get('datasets', ['main'])
        gsm_types = config.get('gsm_types', ['gsm8-std'])
        models = config.get('models', {})
        valid_pairs = []
        for dataset_name in datasets:
            for gsm_type in gsm_types:
                if gsm_type == "gsm8-std" and dataset_name != "main":
                    continue
                valid_pairs.append((dataset_name, gsm_type))
        dataset_samples = {dataset_name: prepare_dataset(dataset_name) for dataset_name in datasets}
        tasks = [
            (dataset_name, gsm_type, model_info, dataset_samples[dataset_name],
             OPENROUTER_API_KEY if model_info['provider'] == 'openrouter'
             else OPENAI_API_KEY if model_info['provider'] == 'openai'
             else DEEPSEEK_API_KEY if model_info['provider'] == 'deepseek'
             else None,
             config)
            for dataset_name, gsm_type in valid_pairs
            for model_info in models.values()
        ]
        max_workers = min(len(tasks), 10)
        print(f"Starting {len(tasks)} tasks with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(worker, tasks))
    if run_analysis:
        analyze_results()

if __name__ == "__main__":
    main()
