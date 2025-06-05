# run_test.py

import os
import json
import re
import pandas as pd
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

import math 
import sympy

from chatbot import load_config
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
        sample = df.sample(n=sample_size)
        print(f"Dataset {dataset_name} loaded from {hf_id_math}. Sample size: {len(sample)}.")

    elif dataset_name == "AIME":
        aime_params = config.get("AIME_params", {})
        hf_id_aime = aime_params.get("hf_id", "opencompass/AIME2025") # Default to new ID
        subset_names_aime = aime_params.get("subset_names", ["AIME2025-I"]) # Default to new subset list
        split_name_aime = aime_params.get("split_name", "test") # Default to 'test' split
        
        print(f"Loading AIME dataset from Hugging Face: ID='{hf_id_aime}', Subsets='{subset_names_aime}', Split='{split_name_aime}'")
        
        all_samples = []
        for subset_name in subset_names_aime:
            try:
                # Load the specific subset and split directly
                ds = load_dataset(hf_id_aime, name=subset_name, split=split_name_aime)
                subset_sample_df = pd.DataFrame(ds) # Use all data from the split
                print(f"Dataset {dataset_name} (Subset: {subset_name}, Split: {split_name_aime}) loaded. Size: {len(subset_sample_df)}.")
                if subset_sample_df.empty:
                    print(f"[Warning] Loaded AIME dataset '{hf_id_aime}' (Subset: {subset_name}, Split: {split_name_aime}) is empty.")
                    # Optionally, continue to the next subset or raise an error
                    # continue 
                # Verify expected columns based on the image: 'question' and 'answer'
                if "question" not in subset_sample_df.columns or "answer" not in subset_sample_df.columns:
                    print(f"[Warning] AIME dataset loaded from {hf_id_aime} (Subset: {subset_name}) is missing 'question' or 'answer' columns. Please check column names. Found columns: {subset_sample_df.columns.tolist()}")
                all_samples.append(subset_sample_df)
            except Exception as e:
                # Decide if one subset failing should halt everything or just be a warning
                print(f"[Error] Failed to load AIME dataset '{hf_id_aime}' (Subset: {subset_name}, Split: {split_name_aime}). Original error: {e}. Skipping this subset.")
                # raise ValueError(f"Failed to load AIME dataset '{hf_id_aime}' (Subset: {subset_name}, Split: {split_name_aime}). Original error: {e}")
        
        if not all_samples:
            raise ValueError(f"Failed to load any AIME subsets for '{hf_id_aime}' with subsets '{subset_names_aime}' and split '{split_name_aime}'.")

        sample = pd.concat(all_samples, ignore_index=True)
        print(f"Combined AIME dataset loaded. Total size: {len(sample)}.")
        if sample.empty:
             raise ValueError(f"Combined AIME dataset from '{hf_id_aime}' (Subsets: {subset_names_aime}, Split: {split_name_aime}) is empty after attempting to load all specified subsets.")

    elif dataset_name in config.get("gsm_types", []) or dataset_name == "main": # GSM-like
        ds = load_dataset("apple/GSM-Symbolic", dataset_name)
        df = pd.DataFrame(ds["test"])
        sample = df.groupby("original_id").sample(n=1)
        print(f"Dataset {dataset_name} (GSM-Symbolic type) loaded. Size: {len(sample)}.")

    else:  # Default to GSM-like from "apple/GSM-Symbolic" if not MATH or AIME
        print(f"Assuming '{dataset_name}' is a GSM-Symbolic configuration (e.g., main, p1, p2). Loading from 'apple/GSM-Symbolic'.")
        try:
            # dataset_name must be a valid configuration for "apple/GSM-Symbolic"
            # e.g., "main", "p1", "p2"
            ds = load_dataset("apple/GSM-Symbolic", dataset_name)
            df = pd.DataFrame(ds["test"])
            # GSM-Symbolic datasets ("main", "p1", "p2") are expected to have "original_id"
            sample = df.groupby("original_id").sample(n=1)
            print(f"Dataset {dataset_name} (GSM-Symbolic type) loaded. Sample size: {len(sample)}.")
        except Exception as e:
            raise ValueError(f"Failed to load '{dataset_name}' as a GSM-Symbolic configuration from 'apple/GSM-Symbolic'. Ensure '{dataset_name}' is a valid configuration (e.g., 'main', 'p1', 'p2') and the dataset is accessible. Original error: {e}")
    return sample

# --- Helper functions for MATH evaluation (remain unchanged) ---
def _normalize_for_sympy(text_expr):
    if text_expr is None: return None
    # Standardize minus signs
    norm_expr = text_expr.replace("–", "-").replace("−", "-") # Added to handle en-dash and mathematical minus
    norm_expr = norm_expr.replace(r"\pi", "pi").replace(r"\cdot", "*").replace(r"\times", "*")
    # Handle \frac
    norm_expr = re.sub(r"\\frac\s*{\s*(.*?)\s*}\s*{\s*(.*?)\s*}", r"(\1)/(\2)", norm_expr)
    # Handle \tfrac (added)
    norm_expr = re.sub(r"\\tfrac\s*{\s*(.*?)\s*}\s*{\s*(.*?)\s*}", r"(\1)/(\2)", norm_expr)
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
    log_prefix_main = f"[EvaluateResponse] (Instance: {instance_id if instance_id else 'Unknown'}, Benchmark: {benchmark_name}):"
    print(f"{log_prefix_main} Raw Response Text (first 100 chars): '{str(response_text)[:100]}...' Expected Answer Text: '{str(expected_answer_text)[:100]}...'")

    try:
        if benchmark_name == "MATH":
            actual_extracted = extract_answer_math(response_text)
            expected_extracted = extract_answer_math(str(expected_answer_text)) 
            log_prefix = f"[Info] MATH eval (Instance: {instance_id if instance_id else 'Unknown'}):"
            
            print(f"{log_prefix} Comparing Actual Extracted: '{actual_extracted}' vs Expected Extracted: '{expected_extracted}'")

            if actual_extracted is None or expected_extracted is None:
                print(f"[Warning] MATH eval (Instance: {instance_id if instance_id else 'Unknown'}): Extraction failed. Actual: '{actual_extracted}', Expected from: '{str(expected_answer_text)[:100]}...' Outputting 0.")
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
            print(f"{log_prefix} All comparisons failed. Outputting 0."); return 0
        elif benchmark_name == "AIME":
            actual_extracted_answer = extract_answer_aime(response_text)
            log_prefix = f"[Info] AIME eval (Instance: {instance_id if instance_id else 'Unknown'}):"
            
            print(f"{log_prefix} Comparing Actual Extracted: '{actual_extracted_answer}' vs Expected: '{str(expected_answer_text).strip()}'")

            if actual_extracted_answer is None: 
                print(f"{log_prefix} Actual extracted is None. Outputting 0.")
                return 0
            is_correct = 1 if actual_extracted_answer.strip() == str(expected_answer_text).strip() else 0
            print(f"{log_prefix} Match result: {is_correct}")
            return is_correct
        elif benchmark_name in ["gsm-symbolic", "gsm8-std", "main", "p1", "p2"]:
            extracted_response_val = extract_answer_gsm_format(response_text)
            log_prefix = f"[Info] GSM-like eval (Instance: {instance_id if instance_id else 'Unknown'}):"

            if extracted_response_val is None: 
                print(f"{log_prefix} Extracted response value is None. Outputting 0.")
                return 0
            try: 
                expected_float = float(expected_answer_text)
            except: 
                print(f"{log_prefix} Could not convert expected_answer_text ('{expected_answer_text}') to float. Outputting 0.")
                return 0
            
            print(f"{log_prefix} Comparing Actual Extracted (float): {extracted_response_val} vs Expected (float): {expected_float}")
            is_correct = 1 if abs(extracted_response_val - expected_float) < 1e-5 else 0
            print(f"{log_prefix} Match result (abs diff < 1e-5): {is_correct}")
            return is_correct
        else: 
            print(f"[Error] Unknown benchmark_name '{benchmark_name}' in evaluate_response. Outputting 0.")
            return 0
    except Exception as e:
        print(f"[Error] Exc in evaluate_response for '{benchmark_name}' (Inst: {instance_id}): {e}. Resp: '{str(response_text)[:100]}...' Exp: '{str(expected_answer_text)[:100]}...'")
        return 0


def run_benchmark(sample, api_key, config, model, dataset_name, benchmark_name, api_provider, supports_sampling_params, thinking_effort_support=False, reasoning_effort="medium"):
    results = []
    total_questions_in_sample = len(sample)
    for i, (idx, row) in enumerate(sample.iterrows()):
        question, expected_answer_val = "", None
        math_level, math_type = None, None 
        processed_golden_answer_for_log = None # For logging next to extracted answers

        if benchmark_name == "gsm8-std":
            question, expected_answer_val = row["original_question"], extract_answer_gsm_format(row["original_answer"])
            processed_golden_answer_for_log = str(expected_answer_val) # Already processed
        elif benchmark_name == "gsm-symbolic":
            question, expected_answer_val = row["question"], extract_answer_gsm_format(row["answer"])
            processed_golden_answer_for_log = str(expected_answer_val) # Already processed
        elif benchmark_name == "MATH":
            question, expected_answer_val = row["problem"], row["solution"]
            # expected_answer_val is raw, extract it for logging consistency with evaluate_response
            processed_golden_answer_for_log = extract_answer_math(str(expected_answer_val))
            math_level = row.get("level") 
            math_type = row.get("type")   
        elif benchmark_name == "AIME":
            question, expected_answer_val = row["question"], str(row["answer"])
            # expected_answer_val is already the clean answer string for AIME
            processed_golden_answer_for_log = str(expected_answer_val).strip()
        else:
            print(f"[Warning] Unknown benchmark '{benchmark_name}' for row {idx}.")
            question, expected_answer_val = row.get("question", "N/A"), None
            processed_golden_answer_for_log = str(expected_answer_val)
        
        current_instance_id = row.get("instance_id", str(idx)) # Use actual instance_id or fallback to DataFrame index
        max_retries = 3
        row_result_dict = None

        for attempt in range(max_retries):
            try:
                print(f"\n--- Question {i+1}/{total_questions_in_sample} (Orig_idx: {idx}, Instance: {current_instance_id}), Attempt={attempt+1}/{max_retries}, Bench: {benchmark_name} ---")
                print(f"Question: {question}")
                print(f"Expected Golden Solution (raw): {expected_answer_val}")

                # Helper to get last N characters, handling None
                get_last_n = lambda text, n: str(text)[-n:] if text and isinstance(text, str) else str(text)

                base_full_response, base_response, base_score = None, None, 0
                cot_full_response, cot_response, cot_score = None, None, 0
                
                if config["test_types"]["run_base"]:
                    print("\n--- Base Model ---")
                    base_full_response = query_model(api_key, question, model, 
                                                     supports_sampling_params=supports_sampling_params, 
                                                     api_provider=api_provider,
                                                     thinking_effort_support=thinking_effort_support,
                                                     reasoning_effort=reasoning_effort,
                                                     temperature=config["answer_temperature"],
                                                     top_p=config["answer_top_p"])
                    if benchmark_name == "MATH": base_response = extract_answer_math(base_full_response)
                    else: base_response = extract_answer_gsm_format(base_full_response)
                    base_score = evaluate_response(base_full_response, expected_answer_val, benchmark_name, config, current_instance_id)
                    print(f"Full Response (last 300 chars): {get_last_n(base_full_response, 300)}")
                    print(f"Extracted Answer: {str(base_response)}")
                    print(f"Ideal Correct Answer (for comparison): {processed_golden_answer_for_log}")
                    print(f"Score: {base_score}")

                if any(config["test_types"].get(k) for k in ["run_cot", "run_traditional_self_reflection", "run_multi_layer_self_reflection"]):
                    print("\n--- Chain-of-Thought (CoT) ---")
                    cot_prompt = generate_cot_prompt(question, benchmark_name)
                    cot_full_response = query_model(api_key, cot_prompt, model, 
                                                    supports_sampling_params=supports_sampling_params, 
                                                    api_provider=api_provider,
                                                    thinking_effort_support=thinking_effort_support,
                                                    reasoning_effort=reasoning_effort,
                                                    temperature=config["answer_temperature"],
                                                    top_p=config["answer_top_p"])
                    if benchmark_name == "MATH": cot_response = extract_answer_math(cot_full_response)
                    else: cot_response = extract_answer_gsm_format(cot_full_response)
                    cot_score = evaluate_response(cot_full_response, expected_answer_val, benchmark_name, config, current_instance_id)
                    print(f"Full Response (last 300 chars): {get_last_n(cot_full_response, 300)}")
                    print(f"Extracted Answer: {str(cot_response)}")
                    print(f"Ideal Correct Answer (for comparison): {processed_golden_answer_for_log}")
                    print(f"Score: {cot_score}")

                reflection_data_traditional_method = []
                if config["test_types"]["run_traditional_self_reflection"]:
                    if cot_score == 0:
                        print("\n--- Traditional Self-Reflection ---")
                        trad_reflect_prompt = generate_auto_reflection_traditional_prompt(question, cot_full_response)
                        trad_reflect_resp = query_model(api_key, trad_reflect_prompt, model, 
                                                        supports_sampling_params=supports_sampling_params, 
                                                        api_provider=api_provider,
                                                        thinking_effort_support=thinking_effort_support,
                                                        reasoning_effort=reasoning_effort,
                                                        temperature=config["answer_temperature"],
                                                        top_p=config["answer_top_p"])
                        print(f"Reflection Response (last 300 chars): {get_last_n(trad_reflect_resp, 300)}")
                        
                        trad_reanswer_prompt = generate_reanswer_prompt(question, cot_response, trad_reflect_resp)
                        trad_reanswer_resp = query_model(api_key, trad_reanswer_prompt, model, 
                                                       supports_sampling_params=supports_sampling_params, 
                                                       api_provider=api_provider,
                                                       thinking_effort_support=thinking_effort_support,
                                                       reasoning_effort=reasoning_effort,
                                                       temperature=config["answer_temperature"],
                                                       top_p=config["answer_top_p"])
                        trad_score = evaluate_response(trad_reanswer_resp, expected_answer_val, benchmark_name, config, current_instance_id)
                        trad_extracted = extract_answer_math(trad_reanswer_resp) if benchmark_name == "MATH" else extract_answer_gsm_format(trad_reanswer_resp)
                        print(f"Re-answer Full Response (last 300 chars): {get_last_n(trad_reanswer_resp, 300)}")
                        print(f"Re-answer Extracted: {str(trad_extracted)}")
                        print(f"Ideal Correct Answer (for comparison): {processed_golden_answer_for_log}")
                        print(f"Score: {trad_score}")
                        reflection_data_traditional_method.append({"layer": None, "score": trad_score, "response": trad_extracted, "reflection_prompt": trad_reflect_prompt, "full_response": trad_reflect_resp, "auto_prompt_used": "traditional"})
                    else: # cot_score == 1
                         print("\n--- Traditional Self-Reflection (Skipped, CoT Correct) ---")
                         print(f"Score (from CoT): {cot_score}")
                         reflection_data_traditional_method.append({"layer": None, "score": cot_score, "response": cot_response, "reflection_prompt": None, "full_response": None, "auto_prompt_used": "cot_correct_skipped_reflection"})

                reflection_data_multi_layer__method = []
                if config["test_types"]["run_multi_layer_self_reflection"]:
                    if cot_score == 1:
                        print("\n--- Multi-Layer Self-Reflection (Skipped, CoT Correct) ---")
                        print(f"Score (from CoT): {cot_score}")
                        reflection_data_multi_layer__method.append({"layer": 0, "score": cot_score, "response": cot_response, "reflection_prompt": None, "full_response": cot_full_response, "auto_prompt_used": None})
                    else:
                        print("\n--- Multi-Layer Self-Reflection ---")
                        current_ans_ext = cot_response # Extracted answer from the last attempt (initially CoT)
                        previous_extracted_incorrect_answers = [cot_response] # List of EXTRACTED incorrect answers

                        past_layer_reflection_prompt=None
                        for layer in range(config["max_reflection_layers"]):
                            print(f"\n--- Layer {layer+1} ---")
                            auto_prompt_model_config_key = config.get("auto_prompt_model", "same")
                            
                            # Determine the model and API details for the meta-prompting (auto_prompt_model)
                            auto_prompt_model_name_to_use: str
                            auto_prompt_api_key_to_use: str
                            auto_prompt_api_provider_to_use: str
                            auto_prompt_supports_sampling_to_use = True
                            auto_prompt_thinking_effort_support = False
                            auto_prompt_reasoning_effort = "medium"

                            if auto_prompt_model_config_key == "same" or auto_prompt_model_config_key not in config.get("models", {}):
                                if auto_prompt_model_config_key != "same":
                                    print(f"[Warning] Auto-prompt model key '{auto_prompt_model_config_key}' not found in config.json. Falling back to main model: {model}")
                                auto_prompt_model_name_to_use = model # Main model name
                                auto_prompt_api_key_to_use = api_key # Main model API key
                                auto_prompt_api_provider_to_use = api_provider # Main model API provider
                                auto_prompt_thinking_effort_support = thinking_effort_support
                                auto_prompt_reasoning_effort = reasoning_effort
                                # Find the main model in config to get its supports_sampling_params
                                for model_details_val in config.get("models", {}).values():
                                    if model_details_val.get("name") == model:
                                        auto_prompt_supports_sampling_to_use = model_details_val.get("supports_sampling_params", True)
                                        break
                            else:
                                specific_model_details = config["models"][auto_prompt_model_config_key]
                                auto_prompt_model_name_to_use = specific_model_details["name"]
                                auto_prompt_api_provider_to_use = specific_model_details["provider"]
                                auto_prompt_supports_sampling_to_use = specific_model_details.get("supports_sampling_params", True)
                                auto_prompt_thinking_effort_support = specific_model_details.get("thinking_effort_support", False)
                                auto_prompt_reasoning_effort = specific_model_details.get("reasoning_effort", "medium")
                                api_key_env_var = specific_model_details.get("api_key_env")
                                if not api_key_env_var:
                                    print(f"[Error] 'api_key_env' not defined for auto-prompt model '{auto_prompt_model_config_key}'. Cannot use for meta-prompting. Halting reflection for this instance.")
                                    break # Break from reflection layers loop for this instance
                                fetched_api_key = os.getenv(api_key_env_var)
                                if not fetched_api_key:
                                    print(f"[Error] API key not found via env var '{api_key_env_var}' for auto-prompt model '{auto_prompt_model_config_key}'. Cannot use for meta-prompting. Halting reflection for this instance.")
                                    break # Break from reflection layers loop for this instance
                                auto_prompt_api_key_to_use = fetched_api_key

                            print(f"[DEBUG] Multi-Layer Reflection Layer {layer+1}: Generating reflection prompt with {len(previous_extracted_incorrect_answers)} previous incorrect extracted answer(s).")

                            if not config["use_same_meta_prompt_for_all_layers"] or past_layer_reflection_prompt is None:
                                reflection_instructions_prompt = generate_auto_reflection_auto_adapt_prompt(
                                    question,
                                    previous_extracted_incorrect_answers, # Pass extracted answers
                                    auto_prompt_model_name_to_use,          
                                    auto_prompt_api_key_to_use,         
                                    auto_prompt_api_provider_to_use,    
                                    auto_prompt_supports_sampling_to_use,
                                    auto_prompt_thinking_effort_support,
                                    auto_prompt_reasoning_effort,
                                    auto_prompt_temperature=config["meta_prompt_generation_temperature"],
                                    auto_prompt_top_p=config["meta_prompt_generation_top_p"]
                                )
                            else:
                                reflection_instructions_prompt = past_layer_reflection_prompt

                            if not reflection_instructions_prompt:
                                print(f"[Warning] Failed to generate reflection instructions for layer {layer+1}. Skipping layer.")
                                break

                            # Log the generated instructions for reflection
                            print(f"[DEBUG] Reflection Instructions Prompt for Layer {layer+1} (last 700 chars):\n'''{get_last_n(reflection_instructions_prompt, 700)}'''")

                            reflection_actual_text = query_model(
                                api_key,
                                reflection_instructions_prompt,
                                model,
                                supports_sampling_params=supports_sampling_params,
                                api_provider=api_provider,
                                thinking_effort_support=thinking_effort_support,
                                reasoning_effort=reasoning_effort,
                                temperature=config["answer_temperature"],
                                top_p=config["answer_top_p"]
                            )
                            print(f"Reflection Actual Text (last 300 chars): {get_last_n(reflection_actual_text, 300)}")

                            if not reflection_actual_text:
                                print(f"[Warning] Main model failed to generate reflection text for layer {layer+1}. Skipping layer.")
                                break

                            reanswer_prompt = generate_reanswer_prompt(question, current_ans_ext, reflection_actual_text)
                            reanswer_full_response = query_model(
                                api_key,
                                reanswer_prompt,
                                model,
                                supports_sampling_params=supports_sampling_params,
                                api_provider=api_provider,
                                thinking_effort_support=thinking_effort_support,
                                reasoning_effort=reasoning_effort,
                                temperature=config["answer_temperature"],
                                top_p=config["answer_top_p"]
                            )

                            if not reanswer_full_response:
                                print(f"[Warning] Main model failed to generate re-answer for layer {layer+1}. Skipping layer.")
                                break

                            current_score = evaluate_response(reanswer_full_response, expected_answer_val, benchmark_name, config, current_instance_id)
                            # Use a new variable for the newly extracted answer to avoid confusion with current_ans_ext from the *previous* step
                            new_extracted_ans = extract_answer_math(reanswer_full_response) if benchmark_name == "MATH" else extract_answer_gsm_format(reanswer_full_response)

                            print(f"Re-answer Full Response (last 300 chars): {get_last_n(reanswer_full_response, 300)}")
                            print(f"Re-answer Extracted: {str(new_extracted_ans)}")
                            print(f"Ideal Correct Answer (for comparison): {processed_golden_answer_for_log}")
                            print(f"Score: {current_score}")

                            reflection_data_multi_layer__method.append({
                                "layer": layer + 1,
                                "score": current_score,
                                "response": new_extracted_ans, # Store the new extracted answer
                                "reflection_prompt": reflection_instructions_prompt,
                                "full_response": reflection_actual_text, # The reflection itself
                                "reanswer_full_response": reanswer_full_response, # The new answer attempt
                                "auto_prompt_used": "auto_adapt"
                            })

                            if current_score == 1:
                                print(f"Correct answer achieved at Multi-Layer Reflection Layer {layer+1}. Stopping reflection for this instance.")
                                break # Exit the reflection loop for this instance

                            # Prepare for the next layer if this one was incorrect and it's not the last layer
                            if layer < config["max_reflection_layers"] - 1:
                                previous_extracted_incorrect_answers.append(new_extracted_ans) # Add the EXTRACTED result of the failed re-answer
                                current_ans_ext = new_extracted_ans # Update current_ans_ext for the next iteration's reanswer_prompt
                            elif current_score == 0: # Still incorrect and it was the last layer
                                print(f"Multi-Layer Reflection: Max layers ({config['max_reflection_layers']}) reached, answer still incorrect for instance {current_instance_id}.")
                
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
        # Always add the base row which contains base and CoT scores
        expanded_results.append(base_row.copy())

        # Add traditional reflection data as a separate effective "row" if it exists
        for trad_item in row.get("traditional_reflection_data", []):
            if trad_item:
                trad_specific_row = base_row.copy() # Start with common data
                trad_specific_row.update({
                    "reflection_layer": "trad", # Special marker for traditional reflection
                    "traditional_reflection_score": trad_item.get("score"),
                    "traditional_reflection_response": trad_item.get("response"),
                    "traditional_reflection_prompt": trad_item.get("reflection_prompt"),
                    "traditional_reflection_full_response": trad_item.get("full_response"),
                    "traditional_auto_prompt_used": trad_item.get("auto_prompt_used")
                })
                expanded_results.append(trad_specific_row)

        # Add multi-layer reflection data, each layer as an effective "row"
        for layer_data in row.get("reflection_data", []):
            if layer_data:
                layer_specific_row = base_row.copy() # Start with common data
                layer_specific_row.update({
                    "reflection_layer": layer_data.get("layer"),
                    "reflection_score": layer_data.get("score"),
                    "reflection_response": layer_data.get("response"),
                    "reflection_prompt": layer_data.get("reflection_prompt"),
                    "reflection_full_response": layer_data.get("full_response"), # This was the reflection itself
                    "reanswer_full_response": layer_data.get("reanswer_full_response"), # This was the re-answer attempt
                    "auto_prompt_used": layer_data.get("auto_prompt_used")
                })
                expanded_results.append(layer_specific_row)
    try:
        # Filter out rows that are purely for base/CoT if traditional or multi-layer data exists for that question
        # This is to avoid double counting in the groupby later if we are not careful\
        final_df = pd.DataFrame(expanded_results)
        final_df.to_csv(filename, index=False)
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

def worker(task_args):
    """
    Worker function to be executed by the ThreadPoolExecutor.
    Unpacks task arguments, runs the benchmark, and saves the results.
    """
    dataset_name, benchmark_name_for_task, model_info, sample, _, config_from_task = task_args
    
    api_key = os.getenv(model_info["api_key_env"])
    if not api_key:
        print(f"[Error] API key not found for model {model_info['name']} using env var {model_info['api_key_env']}. Skipping.")
        return

    # Extract thinking/reasoning parameters from model config
    thinking_effort_support = model_info.get("thinking_effort_support", False)
    reasoning_effort = model_info.get("reasoning_effort", "medium")

    results_df = run_benchmark(
        sample=sample,
        api_key=api_key,
        config=config_from_task,
        model=model_info["name"],
        dataset_name=dataset_name,
        benchmark_name=benchmark_name_for_task,
        api_provider=model_info["provider"],
        supports_sampling_params=model_info.get("supports_sampling_params", True),
        thinking_effort_support=thinking_effort_support,
        reasoning_effort=reasoning_effort
    )
    
    if not results_df.empty:
        results_dir = config_from_task.get("results_dir", "results")
        model_name_sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', model_info["name"]) # Sanitize model name for filename
        filename = os.path.join(results_dir, f"{dataset_name}_{benchmark_name_for_task}_{model_name_sanitized}_results.csv")
        save_results(results_df, filename)
    else:
        print(f"No results to save for {dataset_name} - {benchmark_name_for_task} - {model_info['name']}")

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

if __name__ == "__main__": main()
