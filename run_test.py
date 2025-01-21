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

# Load environment variables from .env file
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

# Function to load and prepare the dataset
def prepare_dataset(dataset_type='main'):
    print(f"Loading the {dataset_type} dataset...")
    # Load only the selected dataset variant
    ds = load_dataset("apple/GSM-Symbolic", dataset_type)

    # Convert to DataFrame using the 'test' split
    df = pd.DataFrame(ds['test'])

    # Select a random sample by "original_id"
    sample = df.groupby("original_id").sample(n=1, random_state=42)

    print(f"Dataset {dataset_type} loaded. Size: {len(sample)}.")
    return sample

# Function to generate prompt using Chain of Thought (CoT)
def generate_cot_prompt(question):
    return f"""{EIGHT_SHOT_EXAMPLES}
Now, look at this question:
Q: {question}
A: Let's think step by step..."""

# Function to generate initial prompt for Self-Reflection (Original Composite)
def generate_auto_reflection_prompt(question, previous_incorrect_answers, auto_prompt_model, api_key):
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

# Function to generate re-answer prompt based on reflection
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
def query_model(api_key, prompt, model):
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=180.0
        )

        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can solve math problems step by step.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            top_p=1,
        )
        response = chat_completion.choices[0].message.content.strip()
        return response if response else None
    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None

# Function to evaluate responses
def evaluate_response(response, expected_answer):
    try:
        if response is None:
            return 0
        return int(float(response) == float(expected_answer))
    except Exception as e:
        print(f"Error in evaluate_response: {e}")
        return 0

# Main function to run experiments
def run_gsm8(sample, api_key, config, model, dataset_name, gsm_type):
    results = []
    for idx, row in sample.iterrows():
        question = row["original_question"] if gsm_type == "gsm8-std" else row["question"]
        expected_answer = extract_answer_gsm_format(row["original_answer"]) if gsm_type == "gsm8-std" else extract_answer_gsm_format(row["answer"])

        print("Expected answer: ", expected_answer)

        # Baseline
        base_prompt = question
        base_full_response = query_model(api_key, base_prompt, model)
        base_response = extract_answer_gsm_format(base_full_response) if base_full_response else None
        base_score = evaluate_response(base_response, expected_answer)

        print(f"Base response: {base_response} - Score: {base_score}")

        # CoT
        cot_prompt = generate_cot_prompt(question)
        cot_full_response = query_model(api_key, cot_prompt, model)
        cot_response = extract_answer_gsm_format(cot_full_response) if cot_full_response else None
        cot_score = evaluate_response(cot_response, expected_answer)

        print(f"COT response: {cot_response} - Score: {cot_score}")

        # Self-Reflection
        reflection_data = []
        previous_incorrect_answers = []

        max_layers = config["max_reflection_layers"]
        current_answer = cot_response
        current_score = cot_score
        auto_prompt_model = config["auto_prompt_model"]

        if current_score == 1:
            reflection_data.append({
                "layer": 0,
                "score": current_score,
                "response": current_answer,
                "reflection_prompt": None,
                "full_response": None,
                "auto_prompt_used": None
            })

        else:
            for layer in range(max_layers):# Correct answer, no need for further reflection
                    
                previous_incorrect_answers.append(current_answer)
                auto_prompt_used = generate_auto_reflection_prompt(question, previous_incorrect_answers, auto_prompt_model, api_key)

                reflection_prompt = auto_prompt_used
                reflection_full_response = query_model(api_key, reflection_prompt, model)

                reanswer_prompt = generate_reanswer_prompt(question, current_answer, reflection_full_response)
                reanswer_full_response = query_model(api_key, reanswer_prompt, model)

                current_answer = extract_answer_gsm_format(reanswer_full_response)
                current_score = evaluate_response(current_answer, expected_answer)
                    
                reflection_data.append({
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
            "reflection_data": reflection_data
        })
    return pd.DataFrame(results)

# Function to save results
def save_results(results_df, filename="experiment_results.csv"):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Expand the list of dictionaries in reflection_data
    expanded_results = []
    for _, row in results_df.iterrows():
        for i, layer_data in enumerate(row['reflection_data']):
            new_row = row.to_dict()
            # Only keep base and cot data for the first reflection layer
            if i > 0:
                new_row.update({
                    'base_full_response': '',
                    'base_response': '',
                    'base_score': '',
                    'cot_full_response': '',
                    'cot_response': '',
                    'cot_score': ''
                })
            new_row.update({
                'reflection_layer': layer_data['layer'],
                'reflection_score': layer_data['score'],
                'reflection_response': layer_data['response'],
                'reflection_prompt': layer_data['reflection_prompt'],
                'reflection_full_response': layer_data['full_response'],
                'auto_prompt_used': layer_data['auto_prompt_used']
            })
            del new_row['reflection_data']  # Remove the original column
            expanded_results.append(new_row)

    expanded_df = pd.DataFrame(expanded_results)
    expanded_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# Function to load configuration from config.json
def load_config():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    return config

def worker(args):
    dataset_name, gsm_type, model, sample, API_KEY, config = args
    try:
        # Create a local copy of config to modify for this worker
        local_config = config.copy()
        
        # If auto_prompt_model is set to "same", use the current model
        if local_config['auto_prompt_model'] == "same":
            local_config['auto_prompt_model'] = model

        print(f"\nExecuting test with:")
        print(f"Dataset: {dataset_name}")
        print(f"GSM type: {gsm_type}")
        print(f"Model: {model}")
        print(f"Max Reflection Layers: {local_config['max_reflection_layers']}")
        print(f"Auto Prompt Model: {local_config['auto_prompt_model']}")

        safe_model_name = model.replace("/", "_")
        results_df = run_gsm8(sample, API_KEY, local_config, model, dataset_name, gsm_type)
        
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        save_results(results_df, f"{results_dir}/results_dataset_{dataset_name}_{gsm_type}_{safe_model_name}.csv")
    except Exception as e:
        print(f"Error in worker thread: {e}")

def analyze_results(results_dir="results"):
    """Analyze all results files and create a summary."""
    all_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    summary_data = []

    for file in all_files:
        df = pd.read_csv(os.path.join(results_dir, file))
        
        # Extract metadata from filename
        parts = file.replace('results_dataset_', '').replace('.csv', '').split('_')
        dataset = parts[0]
        gsm_type = parts[1]
        model = '_'.join(parts[2:])

        # Calculate metrics
        total_questions = len(df[df['reflection_layer'] == 0])  # Count unique questions
        
        # Base method results
        base_correct = df[df['reflection_layer'] == 0]['base_score'].sum()
        base_accuracy = base_correct / total_questions if total_questions > 0 else 0

        # CoT method results
        cot_correct = df[df['reflection_layer'] == 0]['cot_score'].sum()
        cot_accuracy = cot_correct / total_questions if total_questions > 0 else 0

        # Self-reflection results by layer
        max_layer = df['reflection_layer'].max()
        reflection_accuracies = []
        
        for layer in range(max_layer + 1):
            layer_correct = df[df['reflection_layer'] == layer]['reflection_score'].sum()
            layer_accuracy = layer_correct / total_questions if total_questions > 0 else 0
            reflection_accuracies.append(layer_accuracy)

        # Create summary row
        summary_row = {
            'dataset': dataset,
            'gsm_type': gsm_type,
            'model': model,
            'total_questions': total_questions,
            'base_accuracy': base_accuracy,
            'cot_accuracy': cot_accuracy,
        }
        
        # Add reflection layers accuracies
        for layer, accuracy in enumerate(reflection_accuracies):
            summary_row[f'reflection_layer_{layer}_accuracy'] = accuracy

        summary_data.append(summary_row)

    # Create and save summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, 'summary_results.csv'), index=False)
    print(f"Summary results saved to {os.path.join(results_dir, 'summary_results.csv')}")

def main():
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    config = load_config()

    datasets = config.get('datasets', ['main'])
    gsm_types = config.get('gsm_types', ['gsm8-std'])
    models = config.get('models', ['meta-llama/llama-3.1-8b-instruct'])
    
    # Filter out empty model strings
    models = [model for model in models if model]

    # Prepare all dataset samples upfront
    dataset_samples = {dataset_name: prepare_dataset(dataset_name) for dataset_name in datasets}

    # Create all possible combinations of parameters
    tasks = [
        (dataset_name, gsm_type, model, dataset_samples[dataset_name], API_KEY, config)
        for dataset_name in datasets
        for gsm_type in gsm_types
        for model in models
    ]

    # Use ThreadPoolExecutor to run tasks in parallel
    max_workers = min(len(tasks), 10)  # Limit max concurrent threads
    print(f"Starting {len(tasks)} tasks with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(worker, tasks))

    # After all tasks are completed, analyze results
    analyze_results()

if __name__ == "__main__":
    main()