# run_test.py

import os
import json
import re
import pandas as pd
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

import math # Added for MATH evaluation
import sympy # Added for MATH evaluation

# Import query_model and your prompt-generation helpers
from llms_api import query_model
from methods import (
    extract_answer_gsm_format,
    extract_answer_math, 
    extract_answer_aime,
    generate_auto_reflection_auto_adapt_prompt,
    generate_auto_reflection_traditional_prompt,
    generate_cot_prompt,
    generate_reanswer_prompt,
)


def prepare_dataset(dataset_name, config):
    """
    Loads the specified dataset and returns a sample.
    Handles GSM-Symbolic, MATH, AIME  datasets based on dataset_name and config.
    """
    print(f"Loading the {dataset_name} dataset...")

    if dataset_name == "MATH":
        math_params = config.get("MATH_params", {})
        hf_id_math = math_params.get("hf_id", "nlile/hendrycks-MATH-benchmark")
        ds = load_dataset(hf_id_math)
        df = pd.DataFrame(ds["test"]) 
        if "level" not in df.columns or "type" not in df.columns:
            print("[Warning] MATH dataset is missing 'level' or 'type' columns. Disaggregated analysis might be affected.")
        # Use max_test_samples from MATH_params or default to 200
        sample_size = min(len(df), math_params.get("max_test_samples", 200))
        sample = df.sample(n=sample_size, random_state=42)
        print(f"Dataset {dataset_name} loaded from {hf_id_math}. Sample size: {len(sample)}.")

    elif dataset_name == "AIME":
        aime_params = config.get("AIME_params", {})
        hf_id_aime = aime_params.get("hf_id", "gneubig/aime-1983-2024")
        ds = load_dataset(hf_id_aime)
        if "train" not in ds:
            raise ValueError(f"AIME dataset '{hf_id_aime}' does not contain a 'train' split as expected.")
        
        df = pd.DataFrame(ds["train"])

        if "Year" not in df.columns:
            raise ValueError("AIME DataFrame is missing the 'Year' column, required for chronological splitting.")
        df["Year"] = pd.to_numeric(df["Year"], errors='coerce')
        df.dropna(subset=["Year"], inplace=True) 
        df["Year"] = df["Year"].astype(int)
        
        split_type = aime_params.get("split_type", "chronological") 
        max_test_samples = aime_params.get("max_test_samples", 200)
        test_year_exact = aime_params.get("test_year_exact") 
        
        if split_type == "chronological":
            if test_year_exact:
                test_df = df[df["Year"] == test_year_exact].copy()
                print(f"AIME dataset: Total instances = {len(df)}, Filtered for exact year {test_year_exact}, Test instances before sampling = {len(test_df)}")
            else:
                test_years_start = aime_params.get("test_years_start", 2018)
                test_df = df[df["Year"] >= test_years_start].copy()
                print(f"AIME dataset: Total instances = {len(df)}, Chronological split from {test_years_start}, Test instances before sampling = {len(test_df)}")

            if len(test_df) == 0:
                fallback_message = f"year {test_year_exact}" if test_year_exact else f"{test_years_start}+"
                print(f"[Warning] AIME: No instances from {fallback_message}. Using last {max_test_samples} instances from all years.")
                test_df = df.nlargest(max_test_samples, 'Year') 
                if len(test_df) == 0: raise ValueError("AIME dataset empty after fallback.")
            
            sample = test_df.sample(n=min(len(test_df), max_test_samples), random_state=42)
            print(f"AIME dataset: Sampled {len(sample)} test instances. Year range: {sample['Year'].min()}-{sample['Year'].max() if not sample.empty else 'N/A'}")
        elif split_type == "random":
            sample = df.sample(n=min(len(df), max_test_samples), random_state=42)
            print(f"Dataset {dataset_name} random sample size: {len(sample)}.")
        else: 
            sample = df.copy() 
            print(f"Dataset {dataset_name} using all {len(sample)} instances (no split).")

    elif dataset_name in config.get("gsm_types", []) or dataset_name == "main": # GSM-like
        ds = load_dataset("apple/GSM-Symbolic", dataset_name)
        df = pd.DataFrame(ds["test"])
        sample = df.groupby("original_id").sample(n=1, random_state=42)
        print(f"Dataset {dataset_name} (GSM-Symbolic type) loaded. Size: {len(sample)}.")

    else:
        raise ValueError(f"Unsupported dataset_name: '{dataset_name}'.")
    return sample

# --- Helper functions for MATH evaluation (remain unchanged) ---
def _normalize_for_sympy(text_expr):
    if text_expr is None: return None
    norm_expr = text_expr.replace(r"\pi", "pi").replace(r"\cdot", "*").replace(r"\times", "*")
    norm_expr = re.sub(r"\\frac\s*{\s*(.*?)\s*}\s*{\s*(.*?)\s*}", r"(\1)/(\2)", norm_expr)
    norm_expr = re.sub(r"\\sqrt\s*{\s*(.*?)\s*}", r"sqrt(\1)", norm_expr)
    norm_expr = re.sub(r"\\text{\s*(.*?)\s*}", r"\1", norm_expr)
    norm_expr = re.sub(r"\\mathrm{\s*(.*?)\s*}", r"\1", norm_expr)
    if norm_expr.startswith("$") and norm_expr.endswith("$"): norm_expr = norm_expr[1:-1]
    return norm_expr.strip()

def _try_convert_to_float(expr_str):
    if expr_str is None: return None
    try:
        expr_str_pi = expr_str.lower().replace("pi", str(math.pi))
        return float(sympy.sympify(expr_str_pi).evalf())
    except Exception: # Broad exception for various parsing/conversion failures
        fraction_match_parens = re.match(r"^\s*\(\s*([\d\.]+)\s*\)\s*/\s*\(\s*([\d\.]+)\s*\)\s*$", expr_str)
        fraction_match_no_parens = re.match(r"^\s*([\d\.]+)\s*/\s*([\d\.]+)\s*$", expr_str)
        fraction_match = fraction_match_parens or fraction_match_no_parens
        if fraction_match:
            try:
                num = float(fraction_match.group(1))
                den = float(fraction_match.group(2))
                return num / den if den != 0 else None
            except ValueError: return None
        return None

def _normalize_for_string_compare(text_expr):
    if text_expr is None: return None
    norm_expr = text_expr.lower().replace(" ", "") # Remove all spaces
    norm_expr = norm_expr.replace("^","**").replace("\\","") # Basic normalization
    return norm_expr
# --- End of MATH Helper functions ---

def evaluate_response(response_text, expected_answer_text, benchmark_name, config, instance_id=None):
    if response_text is None: return 0
    try:
        if benchmark_name == "MATH":
            actual_extracted = extract_answer_math(response_text)
            expected_extracted = extract_answer_math(str(expected_answer_text)) 
            log_prefix = f"[Info] MATH eval (Instance: {instance_id if instance_id else 'Unknown'}):"
            if actual_extracted is None or expected_extracted is None:
                print(f"[Warning] MATH eval (Instance: {instance_id if instance_id else 'Unknown'}): Extraction failed. Actual: '{actual_extracted}', Expected from: '{str(expected_answer_text)[:100]}...'")
                return 0
            norm_actual_sympy = _normalize_for_sympy(actual_extracted); norm_expected_sympy = _normalize_for_sympy(expected_extracted)
            if norm_actual_sympy is not None and norm_expected_sympy is not None:
                try:
                    parsed_actual = sympy.sympify(norm_actual_sympy); parsed_expected = sympy.sympify(norm_expected_sympy)
                    if sympy.simplify(parsed_actual - parsed_expected) == 0: print(f"{log_prefix} SymPy Exact Match."); return 1
                    if hasattr(parsed_actual, 'evalf') and hasattr(parsed_expected, 'evalf'):
                        if abs(sympy.N(parsed_actual - parsed_expected)) < 1e-9: print(f"{log_prefix} SymPy Numerical Match."); return 1
                except (sympy.SympifyError, TypeError, AttributeError, RecursionError) as e: print(f"[Warning] MATH eval SymPy error '{e}'. Falling back.")
            float_actual = _try_convert_to_float(actual_extracted); float_expected = _try_convert_to_float(expected_extracted)
            if float_actual is not None and float_expected is not None and abs(float_actual - float_expected) < 1e-5: print(f"{log_prefix} Numerical Float Match."); return 1
            norm_actual_str = _normalize_for_string_compare(actual_extracted); norm_expected_str = _normalize_for_string_compare(expected_extracted)
            if norm_actual_str == norm_expected_str: print(f"{log_prefix} Normalized String Match."); return 1
            print(f"{log_prefix} All comparisons failed."); return 0
        elif benchmark_name == "AIME":
            actual_extracted_answer = extract_answer_aime(response_text)
            if actual_extracted_answer is None: return 0
            return 1 if actual_extracted_answer.strip() == str(expected_answer_text).strip() else 0
        elif benchmark_name in ["gsm-symbolic", "gsm8-std", "main", "p1", "p2"]:
            extracted_response_val = extract_answer_gsm_format(response_text)
            if extracted_response_val is None: return 0
            try: expected_float = float(expected_answer_text)
            except: return 0
            return 1 if abs(extracted_response_val - expected_float) < 1e-5 else 0
        else: print(f"[Error] Unknown benchmark_name '{benchmark_name}' in evaluate_response."); return 0
    except Exception as e:
        print(f"[Error] Exc in evaluate_response for '{benchmark_name}' (Inst: {instance_id}): {e}. Resp: '{str(response_text)[:100]}...' Exp: '{str(expected_answer_text)[:100]}...'")
        return 0


def run_benchmark(sample, api_key, config, model, dataset_name, benchmark_name, api_provider, supports_sampling_params):
    results = []
    for idx, row in sample.iterrows():
        question, expected_answer_val = "", None
        math_level, math_type = None, None 

        if benchmark_name == "gsm8-std":
            question, expected_answer_val = row["original_question"], extract_answer_gsm_format(row["original_answer"])
        elif benchmark_name == "gsm-symbolic":
            question, expected_answer_val = row["question"], extract_answer_gsm_format(row["answer"])
        elif benchmark_name == "MATH":
            question, expected_answer_val = row["problem"], row["solution"]
            math_level = row.get("level") 
            math_type = row.get("type")   
        elif benchmark_name == "AIME":
            question, expected_answer_val = row["Problem"], str(row["Answer"])
        else:
            print(f"[Warning] Unknown benchmark '{benchmark_name}' for row {idx}.")
            question, expected_answer_val = row.get("question", "N/A"), None
        
        current_instance_id = row.get("instance_id", str(idx)) # Use actual instance_id or fallback to DataFrame index
        max_retries = 3
        row_result_dict = None

        for attempt in range(max_retries):
            try:
                print(f"\n--- Q_idx={idx} (Instance: {current_instance_id}), Attempt={attempt+1}/{max_retries}, Bench: {benchmark_name} ---")
                # For brevity, actual expected answer not printed here, but available in expected_answer_val

                base_full_response, base_response, base_score = None, None, 0
                cot_full_response, cot_response, cot_score = None, None, 0
                
                if config["test_types"]["run_base"]:
                    base_full_response = query_model(api_key, question, model, supports_sampling_params)
                    if benchmark_name == "MATH": base_response = extract_answer_math(base_full_response)
                    else: base_response = extract_answer_gsm_format(base_full_response)
                    base_score = evaluate_response(base_full_response, expected_answer_val, benchmark_name, config, current_instance_id)
                    print(f"Base Extracted: {str(base_response)[:100]}... Score: {base_score}")

                if any(config["test_types"].get(k) for k in ["run_cot", "run_traditional_self_reflection", "run_multi_layer_self_reflection"]):
                    cot_prompt = generate_cot_prompt(question, benchmark_name)
                    cot_full_response = query_model(api_key, cot_prompt, model, supports_sampling_params)
                    if benchmark_name == "MATH": cot_response = extract_answer_math(cot_full_response)
                    else: cot_response = extract_answer_gsm_format(cot_full_response)
                    cot_score = evaluate_response(cot_full_response, expected_answer_val, benchmark_name, config, current_instance_id)
                    print(f"CoT Extracted: {str(cot_response)[:100]}... Score: {cot_score}")

                reflection_data_traditional_method = []
                if config["test_types"]["run_traditional_self_reflection"] and cot_score == 0:
                    trad_reflect_prompt = generate_auto_reflection_traditional_prompt(question, cot_full_response)
                    trad_reflect_resp = query_model(api_key, trad_reflect_prompt, model, supports_sampling_params)
                    trad_reanswer_prompt = generate_reanswer_prompt(question, cot_response, trad_reflect_resp)
                    trad_reanswer_resp = query_model(api_key, trad_reanswer_prompt, model, supports_sampling_params)
                    trad_score = evaluate_response(trad_reanswer_resp, expected_answer_val, benchmark_name, config, current_instance_id)
                    trad_extracted = extract_answer_math(trad_reanswer_resp) if benchmark_name == "MATH" else extract_answer_gsm_format(trad_reanswer_resp)
                    reflection_data_traditional_method.append({"layer": None, "score": trad_score, "response": trad_extracted, "reflection_prompt": trad_reflect_prompt, "full_response": trad_reflect_resp, "auto_prompt_used": "traditional"})
                elif config["test_types"]["run_traditional_self_reflection"] and cot_score == 1:
                     reflection_data_traditional_method.append({"layer": None, "score": cot_score, "response": cot_response, "reflection_prompt": None, "full_response": None, "auto_prompt_used": None})

                reflection_data_multi_layer__method = []
                if config["test_types"]["run_multi_layer_self_reflection"]:
                    if cot_score == 1:
                        reflection_data_multi_layer__method.append({"layer": 0, "score": cot_score, "response": cot_response, "reflection_prompt": None, "full_response": None, "auto_prompt_used": None})
                    else:
                        cur_ans_ext, cur_full_resp, cur_score = cot_response, cot_full_response, cot_score
                        for layer in range(config["max_reflection_layers"]):
                            adapt_prompt = generate_auto_reflection_auto_adapt_prompt(question, cur_full_resp, config["auto_prompt_model"], api_key, api_provider, benchmark_name)
                            reflect_resp = query_model(api_key, adapt_prompt, model, supports_sampling_params)
                            reanswer_prompt = generate_reanswer_prompt(question, cur_ans_ext, reflect_resp)
                            reanswer_full_resp = query_model(api_key, reanswer_prompt, model, supports_sampling_params)
                            cur_score = evaluate_response(reanswer_full_resp, expected_answer_val, benchmark_name, config, current_instance_id)
                            cur_ans_ext = extract_answer_math(reanswer_full_resp) if benchmark_name == "MATH" else extract_answer_gsm_format(reanswer_full_resp)
                            reflection_data_multi_layer__method.append({
                                "layer": layer + 1, "score": cur_score, "response": cur_ans_ext, 
                                "reflection_prompt": adapt_prompt, "full_response": reflect_resp, 
                                "reanswer_full_response": reanswer_full_resp, "auto_prompt_used": "auto_adapt"})
                            print(f"Reflect L{layer+1} Extracted: {str(cur_ans_ext)[:100]}... Score: {cur_score}")
                            if cur_score == 1: break
                            cur_full_resp = reanswer_full_resp
                
                row_result_dict = {
                    "dataset": dataset_name, "benchmark_name": benchmark_name, "model": model, 
                    "question": question, "expected_answer": expected_answer_val, 
                    "instance_id": current_instance_id, # Ensure instance_id is in results
                    "math_level": math_level, "math_type": math_type,
                    "base_full_response": base_full_response, "base_response": base_response, "base_score": base_score,
                    "cot_full_response": cot_full_response, "cot_response": cot_response, "cot_score": cot_score,
                    "traditional_reflection_data": reflection_data_traditional_method, 
                    "reflection_data": reflection_data_multi_layer__method
                }
                break 
            except Exception as e:
                print(f"[ERROR] Attempt {attempt+1}/{max_retries} failed for Q_idx={idx} (Inst: {current_instance_id}): {e}")
                if attempt == max_retries - 1:
                    row_result_dict = {"dataset": dataset_name, "benchmark_name": benchmark_name, "model": model, 
                                  "question": question, "expected_answer": expected_answer_val, "error": str(e),
                                  "instance_id": current_instance_id, "math_level": math_level, "math_type": math_type}
        if row_result_dict: results.append(row_result_dict)
    return pd.DataFrame(results)

def save_results(results_df, filename="experiment_results.csv"):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory): os.makedirs(directory)
    expanded_results = []
    for _, row in results_df.iterrows():
        base_row = {
            "dataset": row.get("dataset"), "benchmark_name": row.get("benchmark_name"), "model": row.get("model"),
            "question": row.get("question"), "expected_answer": row.get("expected_answer"),
            "instance_id": row.get("instance_id"), # Added instance_id
            "math_level": row.get("math_level"), "math_type": row.get("math_type"),
            "base_full_response": row.get("base_full_response"), "base_response": row.get("base_response"), "base_score": row.get("base_score", 0),
            "cot_full_response": row.get("cot_full_response"), "cot_response": row.get("cot_response"), "cot_score": row.get("cot_score", 0),
            "error": row.get("error"), "reflection_layer": None, "reflection_score": None, "reflection_response": None,
            "reflection_prompt": None, "reflection_full_response": None, "reanswer_full_response": None, "auto_prompt_used": None,
            "traditional_reflection_score": None, "traditional_reflection_response": None,
            "traditional_reflection_prompt": None, "traditional_reflection_full_response": None, "traditional_auto_prompt_used": None
        }
        expanded_results.append(base_row.copy())
        for trad_item in row.get("traditional_reflection_data", []):
            if trad_item: expanded_results.append({**base_row, **trad_item, "reflection_layer": "trad"}) # Mark layer for trad
        for layer_data in row.get("reflection_data", []):
            if layer_data: expanded_results.append({**base_row, **layer_data})
    try:
        pd.DataFrame(expanded_results).to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}. Fallback to JSON.")
        try:
            with open(filename.replace(".csv", ".json"), "w") as f: json.dump(expanded_results, f, indent=2)
            print(f"Results saved to JSON fallback: {filename.replace('.csv', '.json')}")
        except Exception as json_e: print(f"Error saving to JSON: {json_e}")

def analyze_results(results_dir="results"):
    all_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".csv")]
    if not all_files: print("No CSV results found."); return
    df_list = []
    for file in all_files:
        try: df_list.append(pd.read_csv(file))
        except Exception as e: print(f"[Warning] Error reading {file}: {e}")
    if not df_list: print("No valid CSV data found."); return
    
    df = pd.concat(df_list, ignore_index=True)
    numeric_cols = ["base_score", "cot_score", "traditional_reflection_score", "reflection_layer", "reflection_score"]
    for col in numeric_cols: 
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(0)

    def combine_question_data(group): # Aggregates scores for a single question
        res = { "base_score": group["base_score"].iloc[0], "cot_score": group["cot_score"].iloc[0],
                "trad_attempted": 0, "trad_success": 0,
                "reflection_layer_1_score": 0, "reflection_layer_2_score": 0, "reflection_layer_3_score": 0 }
        if "traditional_auto_prompt_used" in group.columns and "traditional_reflection_score" in group.columns:
            trad_scores = group.loc[group["traditional_auto_prompt_used"].notna(), "traditional_reflection_score"]
            if not trad_scores.empty: res["trad_attempted"], res["trad_success"] = 1, trad_scores.max()
        if "reflection_layer" in group.columns and "reflection_score" in group.columns:
            for i in [1, 2, 3]:
                layer_rows = group[group["reflection_layer"] == i]
                if not layer_rows.empty: res[f"reflection_layer_{i}_score"] = layer_rows["reflection_score"].max()
        res["score_up_to_layer_1"] = max(res["cot_score"], res["reflection_layer_1_score"])
        res["score_up_to_layer_3"] = max(res["cot_score"], res["reflection_layer_1_score"], res["reflection_layer_2_score"], res["reflection_layer_3_score"])
        return pd.Series(res)

    main_group_keys = ["dataset", "benchmark_name", "model", "question"] # Use "question" as unique ID per instance
    if not all(key in df.columns for key in main_group_keys):
        print(f"[Error] Main summary: Missing grouping keys. Needed: {main_group_keys}. Available: {list(df.columns)}"); return
    
    pivot_df = df.groupby(main_group_keys, as_index=False, dropna=False).apply(combine_question_data)
    if isinstance(pivot_df.index, pd.MultiIndex): pivot_df = pivot_df.reset_index()

    summary_agg_config = { "base_score": "sum", "cot_score": "sum", "trad_attempted": "sum", "trad_success": "sum",
                           "reflection_layer_1_score": "sum", "reflection_layer_2_score": "sum", "reflection_layer_3_score": "sum",
                           "score_up_to_layer_1": "sum", "score_up_to_layer_3": "sum", "question": "count" }
    valid_summary_agg_cols = {k: v for k,v in summary_agg_config.items() if k in pivot_df.columns}
    summary_df = pivot_df.groupby(["dataset", "benchmark_name", "model"], as_index=False, dropna=False).agg(valid_summary_agg_cols).rename(columns={"question": "total_questions"})
    
    output_path = os.path.join(results_dir, "summary_results.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if not summary_df.empty and summary_df["total_questions"].sum() > 0:
            for metric_base in ["base", "cot", "traditional_reflection", "reflection_layer_1", "reflection_layer_2", "reflection_layer_3", "score_up_to_layer_1", "score_up_to_layer_3"]:
                score_col_map = {"base": "base_score", "cot": "cot_score", "traditional_reflection": "trad_success", **{f"reflection_layer_{i}": f"reflection_layer_{i}_score" for i in [1,2,3]}, **{f"score_up_to_layer_{i}": f"score_up_to_layer_{i}" for i in [1,3]}}
                score_col, acc_col = score_col_map.get(metric_base), f"{metric_base}_accuracy"
                if score_col and score_col in summary_df.columns: summary_df[acc_col] = summary_df[score_col] / summary_df["total_questions"]
                else: summary_df[acc_col] = 0.0
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
            print(f"Saved main summary to: {output_path}, sheet 'Summary'")
        else: print("[Warning] Main summary empty or no questions. Skipping 'Summary' sheet.")

        # MATH Disaggregated Analysis
        if "math_level" in df.columns and "math_type" in df.columns:
            math_df = df[(df["benchmark_name"] == "MATH") & df["math_level"].notna() & df["math_type"].notna()].copy()
            if not math_df.empty:
                math_group_keys = ["model", "math_level", "math_type", "question"]
                if not all(key in math_df.columns for key in math_group_keys): print(f"[Error] MATH Disagg.: Missing keys. Need: {math_group_keys}")
                else:
                    math_pivot_df = math_df.groupby(math_group_keys, as_index=False, dropna=False).apply(combine_question_data)
                    if isinstance(math_pivot_df.index, pd.MultiIndex): math_pivot_df = math_pivot_df.reset_index()
                    
                    valid_math_agg_cols = {k:v for k,v in summary_agg_config.items() if k in math_pivot_df.columns}
                    math_summary_df = math_pivot_df.groupby(["model", "math_level", "math_type"], as_index=False, dropna=False).agg(valid_math_agg_cols).rename(columns={"question": "total_questions"})

                    if not math_summary_df.empty and math_summary_df["total_questions"].sum() > 0:
                        for metric_base in ["base", "cot", "traditional_reflection", "reflection_layer_1", "reflection_layer_2", "reflection_layer_3", "score_up_to_layer_1", "score_up_to_layer_3"]:
                            score_col_map = {"base": "base_score", "cot": "cot_score", "traditional_reflection": "trad_success", **{f"reflection_layer_{i}": f"reflection_layer_{i}_score" for i in [1,2,3]}, **{f"score_up_to_layer_{i}": f"score_up_to_layer_{i}" for i in [1,3]}}
                            score_col, acc_col = score_col_map.get(metric_base), f"{metric_base}_accuracy"
                            if score_col and score_col in math_summary_df.columns: math_summary_df[acc_col] = math_summary_df[score_col] / math_summary_df["total_questions"]
                            else: math_summary_df[acc_col] = 0.0
                        math_summary_df.to_excel(writer, index=False, sheet_name="MATH_Disaggregated")
                        print(f"Saved MATH disaggregated summary to sheet 'MATH_Disaggregated'")
                    else: print("[Warning] MATH disagg. summary empty/no questions. Skipping sheet.")
            else: print("[Info] No MATH data with 'math_level'/'math_type' for disaggregated analysis.")
        else: print("[Info] 'math_level' or 'math_type' cols not in main df. Skipping MATH disagg. analysis.")
    return summary_df

def main():
    from dotenv import load_dotenv
    load_dotenv()
    config = load_config()
    run_test_flag, run_analysis_flag = config.get("run_test", False), config.get("run_analysis", False)

    if run_test_flag:
        datasets_cfg, gsm_types_cfg, models_cfg = config.get("datasets", []), config.get("gsm_types", []), config.get("models", {})
        dataset_samples = {name: prepare_dataset(name, config) for name in datasets_cfg}
        tasks = []
        for ds_name in datasets_cfg:
            if ds_name in ["MATH", "AIME"]:
                for model_info in models_cfg.values(): tasks.append((ds_name, ds_name, model_info, dataset_samples[ds_name], None, config))
            elif ds_name in gsm_types_cfg or ds_name == "main": 
                valid_gsm_benchmarks = [b for b in (gsm_types_cfg if ds_name == "main" else [ds_name]) if b in ["gsm-symbolic", "gsm8-std"]]
                for benchmark_name_for_task in list(set(valid_gsm_benchmarks)):
                    if benchmark_name_for_task == "gsm8-std" and ds_name != "main": continue 
                    for model_info in models_cfg.values(): tasks.append((ds_name, benchmark_name_for_task, model_info, dataset_samples[ds_name], None, config))
            else: 
                 print(f"[Warning] Dataset '{ds_name}' setup unclear. Assuming direct benchmark.")
                 for model_info in models_cfg.values(): tasks.append((ds_name, ds_name, model_info, dataset_samples[ds_name], None, config))
        
        if tasks:
            max_workers = min(len(tasks), (os.cpu_count() or 1) * 2, 10) 
            print(f"Starting {len(tasks)} tasks with {max_workers} workers...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(worker, t) for t in tasks]
                for f in as_completed(futures):
                    try: f.result()
                    except Exception as e: print(f"[ERROR] Worker task failed: {e}")
        else: print("No tasks generated to run.")
    if run_analysis_flag: analyze_results()

if __name__ == "__main__": main
