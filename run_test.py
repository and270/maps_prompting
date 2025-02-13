# run_test.py

import os
import json
import pandas as pd
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import query_model and your prompt-generation helpers
from llms_api import query_model
from methods import (
    extract_answer_gsm_format,
    generate_auto_reflection_auto_adapt_prompt,
    generate_auto_reflection_traditional_prompt,
    generate_cot_prompt,
    generate_reanswer_prompt,
)


def prepare_dataset(dataset_type="main"):
    """
    Loads the specified dataset from "apple/GSM-Symbolic" and returns a sample
    (one random row per original_id).
    """
    print(f"Loading the {dataset_type} dataset...")
    ds = load_dataset("apple/GSM-Symbolic", dataset_type)
    df = pd.DataFrame(ds["test"])
    # Select a random sample by "original_id"
    sample = df.groupby("original_id").sample(n=1, random_state=42)
    print(f"Dataset {dataset_type} loaded. Size: {len(sample)}.")
    return sample


def evaluate_response(response, expected_answer):
    """
    Compare the response to the expected answer for correctness.
    Returns 1 if correct, 0 if incorrect, and logs any parsing errors.
    """
    try:
        if response is None:
            return 0
        return int(float(response) == float(expected_answer))
    except Exception as e:
        print(f"[Warning] Error in evaluate_response: {e}")
        return 0


def run_gsm8(sample, api_key, config, model, dataset_name, gsm_type, api_provider, supports_sampling_params):
    """
    Main function to run experiments on the given sample of GSM-like data.
    - Adds retry logic for each question so that one failing question does not kill the entire process.
    """
    results = []

    for idx, row in sample.iterrows():
        question = (
            row["original_question"] if gsm_type == "gsm8-std" else row["question"]
        )
        expected_answer = (
            extract_answer_gsm_format(row["original_answer"])
            if gsm_type == "gsm8-std"
            else extract_answer_gsm_format(row["answer"])
        )

        # We'll try each question up to 3 times in case of transient errors
        max_retries = 3
        row_result = None  # We store the final result here if successful

        for attempt in range(max_retries):
            try:
                # ---------------------------
                # Perform all queries for a single question
                # ---------------------------
                print(f"\n--- Processing Question idx={idx}, Attempt={attempt+1}/{max_retries} ---")
                print("Expected answer: ", expected_answer)

                # Initialize variables with defaults
                base_full_response = None
                base_response = None
                base_score = 0
                cot_full_response = None
                cot_response = None
                cot_score = 0

                # 1) Baseline
                if config["test_types"]["run_base"]:
                    base_prompt = question
                    base_full_response = query_model(
                        api_key=api_key,
                        prompt=base_prompt,
                        model=model,
                        supports_sampling_params=supports_sampling_params
                    )
                    base_response = (
                        extract_answer_gsm_format(base_full_response)
                        if base_full_response
                        else None
                    )
                    base_score = evaluate_response(base_response, expected_answer)
                    print(f"Base response: {base_response} - Score: {base_score}")

                # 2) Chain-of-Thought (CoT)
                #    Run if either CoT or any reflection method is enabled
                if (
                    config["test_types"]["run_cot"]
                    or config["test_types"]["run_traditional_self_reflection"]
                    or config["test_types"]["run_multi_layer_self_reflection"]
                ):
                    cot_prompt = generate_cot_prompt(question)
                    cot_full_response = query_model(
                        api_key=api_key,
                        prompt=cot_prompt,
                        model=model,
                        supports_sampling_params=supports_sampling_params
                    )
                    cot_response = (
                        extract_answer_gsm_format(cot_full_response)
                        if cot_full_response
                        else None
                    )
                    cot_score = evaluate_response(cot_response, expected_answer)
                    print(f"COT response: {cot_response} - Score: {cot_score}")

                # 3) Traditional Self-Reflection
                reflection_data_traditional_method = []
                if config["test_types"]["run_traditional_self_reflection"]:
                    # Only do reflection if COT is wrong
                    if cot_score == 0:
                        traditional_reflection_prompt = generate_auto_reflection_traditional_prompt(
                            question, cot_full_response
                        )
                        traditional_reflection_response = query_model(
                            api_key=api_key,
                            prompt=traditional_reflection_prompt,
                            model=model,
                            supports_sampling_params=supports_sampling_params
                        )
                        traditional_reanswer_prompt = generate_reanswer_prompt(
                            question, cot_response, traditional_reflection_response
                        )
                        traditional_reanswer_response = query_model(
                            api_key=api_key,
                            prompt=traditional_reanswer_prompt,
                            model=model,
                            supports_sampling_params=supports_sampling_params
                        )
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
                        # If COT was already correct, just store that
                        reflection_data_traditional_method.append({
                            "layer": None,
                            "score": cot_score,
                            "response": None,
                            "reflection_prompt": None,
                            "full_response": None,
                            "auto_prompt_used": None
                        })

                # 4) Multi-layer Self-Reflection
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
                                question,
                                previous_incorrect_answers,
                                auto_prompt_model,
                                api_key,
                                api_provider  # <--- If needed by your auto-prompt generator, keep it here
                            )
                            reflection_prompt = auto_prompt_used
                            reflection_full_response = query_model(
                                api_key=api_key,
                                prompt=reflection_prompt,
                                model=model,
                                supports_sampling_params=supports_sampling_params
                            )
                            reanswer_prompt = generate_reanswer_prompt(
                                question, current_answer, reflection_full_response
                            )
                            reanswer_full_response = query_model(
                                api_key=api_key,
                                prompt=reanswer_prompt,
                                model=model,
                                supports_sampling_params=supports_sampling_params
                            )
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

                # If we get here without raising an exception, build row_result:
                row_result = {
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
                    "traditional_reflection_data": reflection_data_traditional_method,
                    "reflection_data": reflection_data_multi_layer__method
                }

                # Break out of the retry loop (success)
                break

            except Exception as e:
                print(f"[ERROR] Attempt {attempt+1}/{max_retries} failed for question idx={idx}: {e}")
                # If we are on our last attempt, record an error row and skip the question
                if attempt == max_retries - 1:
                    row_result = {
                        "dataset": dataset_name,
                        "gsm_type": gsm_type,
                        "model": model,
                        "question": question,
                        "expected_answer": expected_answer,
                        "error": str(e),
                        # You can put placeholders for the rest or just omit them
                        "base_full_response": None,
                        "base_response": None,
                        "base_score": 0,
                        "cot_full_response": None,
                        "cot_response": None,
                        "cot_score": 0,
                        "traditional_reflection_data": [],
                        "reflection_data": []
                    }

        # After max_retries attempts, if row_result is not None, we append it to results
        if row_result is not None:
            results.append(row_result)

    # Return the results as a DataFrame
    return pd.DataFrame(results)


def save_results(results_df, filename="experiment_results.csv"):
    """
    Save results to CSV. If there's an error writing CSV, fallback to JSON.
    """
    import json
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    expanded_results = []
    for _, row in results_df.iterrows():
        base_row = {
            "dataset": row["dataset"],
            "gsm_type": row["gsm_type"],
            "model": row["model"],
            "question": row["question"],
            "expected_answer": row.get("expected_answer", None),
            "base_full_response": row.get("base_full_response", None),
            "base_response": row.get("base_response", None),
            "base_score": row.get("base_score", 0),
            "cot_full_response": row.get("cot_full_response", None),
            "cot_response": row.get("cot_response", None),
            "cot_score": row.get("cot_score", 0),
            "reflection_layer": None,
            "reflection_score": None,
            "reflection_response": None,
            "reflection_prompt": None,
            "reflection_full_response": None,
            "auto_prompt_used": None,
            "traditional_reflection_score": None,
            "traditional_reflection_response": None,
            "traditional_reflection_prompt": None,
            "traditional_reflection_full_response": None,
            "traditional_auto_prompt_used": None,
            "error": row.get("error", None),
        }
        # Add base row
        expanded_results.append(base_row.copy())

        # Add traditional reflection data if it exists
        trad_data_list = row.get("traditional_reflection_data", [])
        if trad_data_list and len(trad_data_list) > 0:
            # Typically it's only one item, but let's handle multiple
            for trad_item in trad_data_list:
                if trad_item:
                    trad_row = base_row.copy()
                    trad_row.update({
                        "traditional_reflection_score": trad_item.get("score", None),
                        "traditional_reflection_response": trad_item.get("response", None),
                        "traditional_reflection_prompt": trad_item.get("reflection_prompt", None),
                        "traditional_reflection_full_response": trad_item.get("full_response", None),
                        "traditional_auto_prompt_used": trad_item.get("auto_prompt_used", None)
                    })
                    expanded_results.append(trad_row)

        # Add multi-layer reflection data if it exists
        reflection_data_list = row.get("reflection_data", [])
        for layer_data in reflection_data_list:
            if layer_data:
                reflection_row = base_row.copy()
                reflection_row.update({
                    "reflection_layer": layer_data.get("layer", None),
                    "reflection_score": layer_data.get("score", None),
                    "reflection_response": layer_data.get("response", None),
                    "reflection_prompt": layer_data.get("reflection_prompt", None),
                    "reflection_full_response": layer_data.get("full_response", None),
                    "auto_prompt_used": layer_data.get("auto_prompt_used", None),
                })
                expanded_results.append(reflection_row)

    try:
        expanded_df = pd.DataFrame(expanded_results)
        expanded_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        fallback_filename = filename.replace(".csv", ".json")
        with open(fallback_filename, "w") as f:
            json.dump(expanded_results, f, indent=2)
        print(f"Results saved to fallback JSON file: {fallback_filename}")


def load_config():
    """
    Loads configuration from config.json
    """
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    return config


def worker(args):
    """
    Worker function to process a single (dataset_name, gsm_type, model_info) tuple.
    """
    dataset_name, gsm_type, model_info, sample, _, config = args
    try:
        local_config = config.copy()
        model_name = model_info["name"]
        api_provider = model_info["provider"]
        supports_sampling_params = model_info.get("supports_sampling_params", True)

        # If auto_prompt_model is set to "same", use the current model as auto-prompt model
        if local_config["auto_prompt_model"] == "same":
            local_config["auto_prompt_model"] = model_name

        print(f"\n--- Executing test with ---")
        print(f"Dataset: {dataset_name}")
        print(f"GSM type: {gsm_type}")
        print(f"Model: {model_name}")
        print(f"Provider: {api_provider}")
        print(f"Supports sampling params: {supports_sampling_params}")
        print(f"Max Reflection Layers: {local_config['max_reflection_layers']}")
        print(f"Auto Prompt Model: {local_config['auto_prompt_model']}")

        # Determine correct API key from environment variables
        api_key_map = {
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
        }
        API_KEY = api_key_map.get(api_provider)
        if not API_KEY:
            raise ValueError(f"No API key found for provider: {api_provider}")

        safe_model_name = model_name.replace("/", "_")
        results_df = run_gsm8(
            sample=sample,
            api_key=API_KEY,
            config=local_config,
            model=model_name,
            dataset_name=dataset_name,
            gsm_type=gsm_type,
            api_provider=api_provider,  # We still pass it so run_gsm8 can pass to other methods if needed
            supports_sampling_params=supports_sampling_params
        )

        # Save results
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        safe_dataset_name = dataset_name.replace("/", "_")
        safe_gsm_type = gsm_type.replace("-", "_")
        filename = f"{results_dir}/results_{safe_dataset_name}_{safe_gsm_type}_{safe_model_name}.csv"
        save_results(results_df, filename)

    except Exception as e:
        print(f"[ERROR] Unhandled exception in worker thread: {e}")


def analyze_results(results_dir="results"):
    """
    Reads all CSVs in the given directory, combines them into a DataFrame,
    and computes summary statistics. Saves the summary to an Excel file.
    """
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
        "base_score",
        "cot_score",
        "traditional_reflection_score",
        "reflection_layer",
        "reflection_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Group-level aggregator for each question
    def combine_question_data(group):
        # We can rely on .iloc[0] for columns that do not vary by reflection row
        base_score = group["base_score"].iloc[0]
        cot_score = group["cot_score"].iloc[0]

        # Traditional reflection
        trad_scores = group["traditional_reflection_score"].dropna()
        trad_attempted = len(trad_scores) > 0
        trad_success = trad_scores.max() if trad_attempted else 0

        # For multi-layer reflection, check layer 1, 2, 3
        layer_1_rows = group["reflection_layer"] == 1
        layer_2_rows = group["reflection_layer"] == 2
        layer_3_rows = group["reflection_layer"] == 3

        reflection_layer_1_score = group.loc[layer_1_rows, "reflection_score"].max() if layer_1_rows.any() else 0
        reflection_layer_2_score = group.loc[layer_2_rows, "reflection_score"].max() if layer_2_rows.any() else 0
        reflection_layer_3_score = group.loc[layer_3_rows, "reflection_score"].max() if layer_3_rows.any() else 0

        # Score if we only allow reflection layer 1 (i.e., COT or layer 1)
        score_up_to_layer_1 = max(cot_score, reflection_layer_1_score)

        # Score if we allow up to layer 3 (COT or reflection up to layer 3)
        score_up_to_layer_3 = max(
            cot_score,
            reflection_layer_1_score,
            reflection_layer_2_score,
            reflection_layer_3_score,
        )

        return pd.Series({
            "base_score": base_score,
            "cot_score": cot_score,
            "trad_attempted": int(trad_attempted),
            "trad_success": trad_success,
            "reflection_layer_1_score": reflection_layer_1_score,
            "reflection_layer_2_score": reflection_layer_2_score,
            "reflection_layer_3_score": reflection_layer_3_score,
            "score_up_to_layer_1": score_up_to_layer_1,
            "score_up_to_layer_3": score_up_to_layer_3,
        })

    pivot_df = (
        df.groupby(["dataset", "gsm_type", "model", "question"], as_index=False)
        .apply(combine_question_data)
    )

    summary_df = (
        pivot_df.groupby(["dataset", "gsm_type", "model"], as_index=False)
        .agg({
            "base_score": "sum",
            "cot_score": "sum",
            "trad_attempted": "sum",
            "trad_success": "sum",
            "reflection_layer_1_score": "sum",
            "reflection_layer_2_score": "sum",
            "reflection_layer_3_score": "sum",
            "score_up_to_layer_1": "sum",
            "score_up_to_layer_3": "sum",
            "question": "count",  # 'question' is the grouping key, so .count() = number of unique questions
        })
        .rename(columns={"question": "total_questions"})
    )

    # Basic accuracies
    summary_df["base_accuracy"] = summary_df["base_score"] / summary_df["total_questions"]
    summary_df["cot_accuracy"] = summary_df["cot_score"] / summary_df["total_questions"]

    # Traditional reflection accuracy
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

    # Additional columns for partial-layer reflection results
    summary_df["accuracy_up_to_layer_1"] = (
        summary_df["score_up_to_layer_1"] / summary_df["total_questions"]
    )
    summary_df["accuracy_up_to_layer_3"] = (
        summary_df["score_up_to_layer_3"] / summary_df["total_questions"]
    )

    # Save to Excel
    output_path = os.path.join(results_dir, "summary_results.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

    print("Saved summarized results to:", output_path)
    return summary_df


def main():
    """
    Main entrypoint:
    1) Loads config.
    2) Depending on config, runs tests (with threading) and/or runs analysis.
    """
    from dotenv import load_dotenv

    # Load environment variables (OPENROUTER_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, etc.)
    load_dotenv()

    config = load_config()
    run_test = config.get("run_test", False)
    run_analysis = config.get("run_analysis", False)

    if run_test:
        # Gather tasks
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

        datasets = config.get("datasets", ["main"])
        gsm_types = config.get("gsm_types", ["gsm8-std"])
        models = config.get("models", {})

        valid_pairs = []
        for dataset_name in datasets:
            for gsm_type in gsm_types:
                # If you want to skip certain combos, you can do it here:
                if gsm_type == "gsm8-std" and dataset_name != "main":
                    continue
                valid_pairs.append((dataset_name, gsm_type))

        # Pre-load samples for each dataset once
        dataset_samples = {
            dataset_name: prepare_dataset(dataset_name) for dataset_name in datasets
        }

        tasks = []
        for dataset_name, gsm_type in valid_pairs:
            for model_info in models.values():
                tasks.append((
                    dataset_name,
                    gsm_type,
                    model_info,
                    dataset_samples[dataset_name],
                    None,  # not used, but included for structure
                    config
                ))

        max_workers = min(len(tasks), 10)
        print(f"Starting {len(tasks)} tasks with {max_workers} workers...")

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for t in tasks:
                futures.append(executor.submit(worker, t))

            for f in as_completed(futures):
                try:
                    f.result()  # If a worker raised an exception, it will be re-raised here
                except Exception as e:
                    print(f"[ERROR] Worker failed with: {e}")
                    # We continue; this does not kill the entire job

    # Run analysis if requested
    if run_analysis:
        analyze_results()


if __name__ == "__main__":
    main()
