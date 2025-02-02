# Import necessary libraries
import pandas as pd
from datasets import load_dataset
import openai
import re
import os
from dotenv import load_dotenv
import json
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import time

load_dotenv()

EIGHT_SHOT_EXAMPLES = """
See the examples below to guide you on how your answer format should be:

Q: If John has 3 apples and buys 2 more, how many apples does he have now?
A: Let's think step by step.
John starts with 3 apples.
He buys 2 more apples.
So, the total number of apples is 3 + 2 = 5.
The answer is 5.

Q: Sarah had 10 candies. She gave 4 to her friend. How many candies does she have left?
A: Let's think step by step.
Sarah starts with 10 candies.
She gives 4 candies to her friend.
So, the number of candies left is 10 - 4 = 6.
The answer is 6.

Q: There are 5 baskets, each with 3 oranges. How many oranges are there in total?
A: Let's think step by step.
Each basket has 3 oranges.
There are 5 baskets.
So, the total number of oranges is 5 * 3 = 15.
The answer is 15.

Q: A baker has 24 cupcakes and packs them equally into 6 boxes. How many cupcakes are in each box?
A: Let's think step by step.
The baker has 24 cupcakes.
He packs them into 6 boxes equally.
So, each box contains 24 / 6 = 4 cupcakes.
The answer is 4.

Q: Emily read 7 pages of a book on Monday and 9 pages on Tuesday. How many pages did she read in total?
A: Let's think step by step.
Emily read 7 pages on Monday.
She read 9 pages on Tuesday.
So, the total number of pages read is 7 + 9 = 16.
The answer is 16.

Q: Tom had 15 marbles. He lost 7 of them. How many marbles does he have now?
A: Let's think step by step.
Tom starts with 15 marbles.
He loses 7 marbles.
So, the number of marbles left is 15 - 7 = 8.
The answer is 8.

Q: There are 4 packs of pencils, each containing 6 pencils. How many pencils are there in total?
A: Let's think step by step.
Each pack contains 6 pencils.
There are 4 packs.
So, the total number of pencils is 4 * 6 = 24.
The answer is 24.

Q: A gardener has 36 flowers and plants them equally in 9 rows. How many flowers are in each row?
A: Let's think step by step.
The gardener has 36 flowers.
He plants them equally in 9 rows.
So, each row contains 36 / 9 = 4 flowers.
The answer is 4.
"""

def prepare_dataset(dataset_type='main'):
    print(f"Loading the {dataset_type} dataset...")

    ds = load_dataset("apple/GSM-Symbolic", dataset_type)
    df = pd.DataFrame(ds['test'])

    # Select a random sample by "original_id"
    sample = df.groupby("original_id").sample(n=1, random_state=42)

    print(f"Dataset {dataset_type} loaded. Size: {len(sample)}.")
    return sample


def generate_cot_prompt(question):
    return f"""{EIGHT_SHOT_EXAMPLES}
Now, look at this question:
Q: {question}
A: Let's think step by step..."""

def generate_auto_reflection_traditional_prompt(question, answer):
    return f"""You are an expert in math.
You have incorrectly answered the following question.
Your task is to reflect on the problem, your solution, and the correct answer.
You will then use this information help you answer the same question in the future.
First, explain why you answered the question incorrectly.
Second, list the keywords that describe the type of your errors from most general to most specific.
Third, solve the problem again, step-by-step, based on your knowledge of the correct answer.
Fourth, create a list of detailed instructions to help you correctly solve this problem in the future.
Finally, create a list of general advice to help you solve similar types of problems in the future.
Be concise in your response; however, capture all of the essential information.
For guidance, I will provide you with a single generic example problem and reflection (below).
[Example Input]
Question: What is the product of the number of letters contained in the name of the city
where Iowa State University is located multiplied by the number of letters
contained in the name of the state?
Answer:
Iowa State University is located in the city of Ames
ISU is located in the state of Iowa.
The answer is 32
---
[Example Output]
Explanation:
I miscalculated the product of the number of letters in the city and state names.
The gap in my knowledge was not in geography but in basic arithmetic.
I knew the correct city and state but made a calculation error.
Error Keywords:
- Calculation error
- Arithmetic error
- Multiplication error
Instructions:
1. Identify the city where the university is located.
2. Identify the state where the university is located.
3. Count the number of letters in the name of the city.
4. Count the number of letters in the name of the state.
5. Multiply the number of letters in the city by the number of letters in the state.
6. Work step-by-step through your mathematical calculations.
7. Double-check your calculations to ensure accuracy.
8. Choose the answer that matches your calculated result.
Advice:
- Always read the question carefully and understand the problem.
- Always decompose complex problems into multiple simple steps.
- Always think through each subproblem step-by-step.
- Never skip any steps; be explicit in each step of your reasoning.
- Always double-check your calculations and final answer.
- Remember that the product of two numbers is the result of multiplying them together,
not adding them.
Solution:
Iowa State University is located in the city of Ames
Iowa State University is located in the state of Iowa.
The city name "Ames" contains 4 letters.
The state name "Iowa" contains 4 letters.
The product of 4*4 is 16.
The answer is 16
Now, look at this question:
Question: {question}
Your initial answer was: {answer}
You previously answered this question incorrectly. Reflect on why your answer was incorrect and identify the type of error. Then, solve the problem again step-by-step with corrections.
"""

def generate_auto_reflection_auto_adapt_prompt(question, previous_incorrect_answers, auto_prompt_model, api_key):
    meta_prompt = f"""You are an expert in adapting instructions for language models. Your task is to create a personalized Self-Reflection prompt for a model that is trying to solve a mathematical problem. You will receive the original question and should adapt the prompt based on it.

Your task is to modify the Self-Reflection template so that it is as specific and helpful as possible for the problem. Focus on aspects such as:

*   **Type of problem:** The Self-Reflection prompt should guide the model to solve the specific type of problem presented in the question.
*   **Common mistakes:** The Self-Reflection prompt should guide the model to identify the common mistakes that are made when solving this type of problem.
*   **Complexity of the problem:** The Self-Reflection prompt should guide the model to try to understand the complexity of the problem, if more steps arte needed to solve it.


Here is the original Self-Reflection template that you should adapt:

--- Beginning of the template ---
You are an expert in <PROBLEM_AREA>.
You have incorrectly answered the following question.
Your task is to reflect on the problem, your solution, and the correct answer.
You will then use this information help you answer the same question in the future.
First, explain why you answered the question incorrectly.
Second, list the keywords that describe the type of your errors from most general to most specific.
Third, solve the problem again, step-by-step, based on your knowledge of the correct answer.
Fourth, create a list of detailed instructions to help you correctly solve this problem in the future.
Finally, create a list of general advice to help you solve similar types of problems in the future.
Be concise in your response; however, capture all of the essential information.
For guidance, I will provide you with a single generic example problem and reflection (below).
[Example Input]
Question: <an example question similar on complexity to the question received>
Wrong answer: <the wrong reasoning and answer to the example question, with a specific mistake made along the way that resulted in the wrong answer>
---
[Example Output]
Explanation:
I miscalculated the <explanation of the mistake>
Error Keywords:
- <keywords of the mistake>
Instructions:
<list of instructions to solve the problem>
Advice:
<list of general advice to solve similar types of problems>
Solution:
<the correct reasoning and answer to the example question>
--- End of the template ---

Now, adapt the above template for the following question:

Question: {question}

Generate the adapted Self-Reflection prompt (remember, you need to create a similar example question on complexity to the question received (NOT THE SAME ONE), a wrong answer to it and the correct answer):
"""

    adapted_reflection_prompt = query_model(api_key, meta_prompt, auto_prompt_model)

    adapted_reflection_prompt += f"""\n\nNow, look at this question:
Question: {question}
Your initial Chain-of-Thought answer was: {previous_incorrect_answers[0] if previous_incorrect_answers else 'None'}"""

    if len(previous_incorrect_answers) > 1:
        adapted_reflection_prompt += "\n\nYour previous reflection answers were:"
        for i, answer in enumerate(previous_incorrect_answers[1:], 1):
            adapted_reflection_prompt += f"\nReflection {i}: {answer}"
    
    adapted_reflection_prompt += """\nYou previously answered this question incorrectly. Reflect on why your answer was incorrect and identify the type of error. Then, solve the problem again step-by-step with corrections. Your new answer MUST be different from your previous answers cause they were all incorrect."""

    return adapted_reflection_prompt


def generate_reanswer_prompt(question, answer, reflection):
    return f"""{EIGHT_SHOT_EXAMPLES}

Now, look at this question:
Question: {question}
Your initial answer was: {answer}
You previously answered this question incorrectly.
Then you reflected on the problem, your solution, and the correct answer::
Reflection: {reflection}

Provide your corrected reasoning and answer in the examples format.
"""

# Function to extract the numerical answer from the GSM format
def extract_answer_gsm_format(response):
    try:
        if not response:
            return None
        response = response.replace(",", "")
        numbers = re.findall(r"-?\d+\.?\d*", response)
        if not numbers:
            return None
        return float(numbers[-1])
    except Exception as e:
        print(f"Error in extract_answer_gsm_format: {e}")
        return None


# Function to interact with models using OpenRouter API
def query_model(api_key, prompt, model, supports_sampling_params=True, api_provider="openrouter", max_retries=3):
    for attempt in range(max_retries):
        try:
            if api_provider == "openrouter":
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    timeout=180.0
                )
            elif api_provider == "openai":
                client = openai.OpenAI(
                    api_key=api_key,
                )
            elif api_provider == "deepseek":
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com/v1",
                )
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
            
            # Base parameters for the API call
            api_params = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can solve math problems step by step.",
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            
            # Add sampling parameters only if supported
            if supports_sampling_params:
                api_params.update({
                    "temperature": 0,
                    "top_p": 1,
                })
            
            chat_completion = client.chat.completions.create(**api_params)
            response = chat_completion.choices[0].message.content.strip()
            return response if response else None
            
        except Exception as e:
            error_details = {
                "attempt": attempt + 1,
                "api_provider": api_provider,
                "model": model,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            
            print("\nError Details:")
            for key, value in error_details.items():
                print(f"{key}: {value}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds... ({attempt + 1}/{max_retries})")
                time.sleep(5)
            else:
                print(f"Failed after {max_retries} attempts.")
                return None


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

        # Initialize variables with None/0
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
            if cot_score == 0:
                reflection_data_multi_layer__method.append({
                    "layer": 0,
                    "score": current_score,
                    "response": current_answer,
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
                    auto_prompt_used = generate_auto_reflection_auto_adapt_prompt(question, previous_incorrect_answers, auto_prompt_model, api_key)

                    reflection_prompt = auto_prompt_used
                    reflection_full_response = query_model(api_key, reflection_prompt, model, supports_sampling_params, api_provider)

                    reanswer_prompt = generate_reanswer_prompt(question, current_answer, reflection_full_response)
                    reanswer_full_response = query_model(api_key, reanswer_prompt, model, supports_sampling_params, api_provider)

                    current_answer = extract_answer_gsm_format(reanswer_full_response)
                    current_score = evaluate_response(current_answer, expected_answer)
                    
                    reflection_data_multi_layer__method.append({
                            "layer": layer+1,
                            "score": current_score,
                            "response": current_answer,
                            "reflection_prompt": reflection_prompt,
                            "full_response": reflection_full_response,
                            "auto_prompt_used": auto_prompt_used
                    })

                    print(f"Reflection (layer {layer+1}) response: {current_answer} - Score: {current_score}")

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
            'base_full_response': row['base_full_response'],
            'base_response': row['base_response'],
            'base_score': row['base_score'],
            'cot_full_response': row['cot_full_response'],
            'cot_response': row['cot_response'],
            'cot_score': row['cot_score']
        }

        # Always add the base row (contains CoT results) even without reflections
        expanded_results.append(base_row.copy())

        # Add traditional reflection data (only once) if exists
        if row['traditional_reflection_data']:
            traditional_data = row['traditional_reflection_data'][0]
            trad_row = base_row.copy()
            trad_row.update({
                'traditional_reflection_score': traditional_data['score'],
                'traditional_reflection_response': traditional_data['response'],
                'traditional_reflection_prompt': traditional_data['reflection_prompt'],
                'traditional_reflection_full_response': traditional_data['full_response'],
                'traditional_auto_prompt_used': traditional_data['auto_prompt_used']
            })
            expanded_results.append(trad_row)

        # Add multi-layer reflection data if exists
        for layer_data in row['reflection_data']:
            reflection_row = base_row.copy()
            reflection_row.update({
                'reflection_layer': layer_data['layer'],
                'reflection_score': layer_data['score'],
                'reflection_response': layer_data['response'],
                'reflection_prompt': layer_data['reflection_prompt'],
                'reflection_full_response': layer_data['full_response'],
                'auto_prompt_used': layer_data['auto_prompt_used']
            })
            expanded_results.append(reflection_row)

    expanded_df = pd.DataFrame(expanded_results)
    expanded_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

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

        safe_model_name = model_name.replace("/", "_")
        results_df = run_gsm8(sample, API_KEY, local_config, model_name, dataset_name, gsm_type, api_provider, supports_sampling_params)
        
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Change filename generation to use validated parameters
        safe_dataset_name = dataset_name.replace("/", "_")
        safe_gsm_type = gsm_type.replace("-", "_")
        save_results(results_df, f"{results_dir}/results_{safe_dataset_name}_{safe_gsm_type}_{safe_model_name}.csv")
    except Exception as e:
        print(f"Error in worker thread: {e}")

def analyze_results(results_dir="results"):
    """
    Analyze all expanded results CSV files (produced by `save_results`) in `results_dir`
    and create a single Excel file summarizing the accuracy at each method/level.
    """
    all_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
    if not all_files:
        print("No CSV results files found in the directory.")
        return

    df_list = []
    for file in all_files:
        file_path = os.path.join(results_dir, file)
        temp_df = pd.read_csv(file_path)
        df_list.append(temp_df)

    if not df_list:
        print("No data found in the results directory.")
        return

    df = pd.concat(df_list, ignore_index=True)

    def combine_question_data(group):
        """Handle missing columns dynamically"""
        scores = {
            "base_score": 0,
            "cot_score": 0,
            "traditional_reflection_score": 0,
            "reflection_layer_1_score": 0,
            "reflection_layer_2_score": 0,
            "reflection_layer_3_score": 0,
        }

        # Update only existing scores
        for score in scores.keys():
            if score in group.columns:
                score_series = group[score].dropna()
                scores[score] = score_series.iloc[0] if not score_series.empty else 0

        return pd.Series({
            "base_score": scores["base_score"],
            "cot_score": scores["cot_score"],
            "traditional_reflection_score": scores["traditional_reflection_score"],
            "reflection_layer_1_score": scores["reflection_layer_1_score"],
            "reflection_layer_2_score": scores["reflection_layer_2_score"],
            "reflection_layer_3_score": scores["reflection_layer_3_score"],
            "question": group["question"].iloc[0]  # Preserve original question
        })

    # Apply the aggregator to get a single row per question
    pivot_df = (
        df
        .groupby(["dataset", "gsm_type", "model", "question"], as_index=False)
        .apply(combine_question_data)
    )

    # Calculate summary statistics
    summary_df = (
        pivot_df
        .groupby(["dataset", "gsm_type", "model"], as_index=False)
        .agg({
            "base_score": "sum",
            "cot_score": "sum",
            "traditional_reflection_score": "sum",
            "reflection_layer_1_score": "sum",
            "reflection_layer_2_score": "sum",
            "reflection_layer_3_score": "sum",
            "question": "count"
        })
    )

    # Rename question->total_questions for clarity
    summary_df.rename(columns={"question": "total_questions"}, inplace=True)

    # Dynamic accuracy calculation
    accuracy_columns = []
    if "base_score" in summary_df.columns:
        summary_df["base_accuracy"] = summary_df["base_score"] / summary_df["total_questions"]
        accuracy_columns.append("base_accuracy")
    
    if "cot_score" in summary_df.columns:
        summary_df["cot_accuracy"] = summary_df["cot_score"] / summary_df["total_questions"]
        accuracy_columns.append("cot_accuracy")
    
    # Similar conditional checks for other scores...
    
    # Finally, keep only relevant columns
    keep_columns = ["dataset", "gsm_type", "model", "total_questions"] + accuracy_columns
    summary_df = summary_df[keep_columns]

    output_path = os.path.join(results_dir, "summary_results.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

    print(f"Summary results saved to {output_path}")

def main():
    # Load different API keys from environment
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
        
        # Create valid dataset/gsm_type pairs
        valid_pairs = []
        for dataset_name in datasets:
            for gsm_type in gsm_types:
                # Enforce that gsm8-std only runs with main dataset
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

        # Use ThreadPoolExecutor to run tasks in parallel
        max_workers = min(len(tasks), 10)
        print(f"Starting {len(tasks)} tasks with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(worker, tasks))

    if run_analysis:
        analyze_results()

if __name__ == "__main__":
    main()