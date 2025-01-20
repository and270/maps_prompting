# Import necessary libraries
import pandas as pd
from datasets import load_dataset
import openai
import re
import os
from dotenv import load_dotenv
import json

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
def generate_initial_reflection_prompt(question, answer):
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

# Function to generate Self-Reflection prompt using auto-prompting
def generate_auto_reflection_prompt(question, cot_answer, auto_prompt_model, api_key):
    meta_prompt = """You are an expert in adapting instructions for language models. Your task is to create a personalized Self-Reflection prompt for a model that is trying to solve a mathematical problem. You will receive the original question, the model's generated answer (including the step-by-step reasoning), and a template for a Self-Reflection prompt.

Your task is to modify the Self-Reflection template so that it is as specific and helpful as possible for the problem and the generated answer in question. Focus on aspects such as:

*   **Type of error (if evident):** If the CoT answer is wrong, the Self-Reflection prompt should guide the model in identifying the type of error (calculation, logical, interpretation, etc.).
*   **Specific steps of CoT:** The Self-Reflection prompt can refer to specific steps of the CoT reasoning where the error may have occurred.
*   **Numbers and Operations:** If relevant, highlight the numbers and operations involved in the question, so that the model pays special attention to them.
*   **Expected answer format:** Reinforce the expected numerical format, without additional information, such as "The answer is X."

Here is the original Self-Reflection template that you should adapt:

---
[Original Self-Reflection Template (Composite)]

You are an expert in math.
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

---

Now, adapt the above template for the following question and CoT answer:

Question: [Original question]
CoT Answer: [Model-generated CoT answer]

Generate the adapted Self-Reflection prompt:
"""

    prompt_for_auto_prompting = f"{meta_prompt}\n\nQuestion: {question}\nCoT Answer: {cot_answer}\n\nGenerate the adapted Self-Reflection prompt:"

    adapted_reflection_prompt = query_model(api_key, prompt_for_auto_prompting, auto_prompt_model)

    if adapted_reflection_prompt is None:
        print("Error generating auto-reflection prompt. Using default prompt.")
        adapted_reflection_prompt = generate_initial_reflection_prompt(question, cot_answer)

    return adapted_reflection_prompt

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
        if config["use_auto_prompt"]:
            max_layers = config["max_reflection_layers"]
            current_answer = cot_response
            current_score = cot_score
            auto_prompt_model = config["auto_prompt_model"]

            for layer in range(max_layers):
                if current_score == 1:
                    reflection_data.append({
                        "layer": layer,
                        "score": current_score,
                        "response": current_answer,
                        "reflection_prompt": None,
                        "full_response": None,
                        "auto_prompt_used": None
                    })
                    break  # Correct answer, no need for further reflection
                
                auto_prompt_used = generate_auto_reflection_prompt(question, current_answer, auto_prompt_model, api_key) if layer > 0 else generate_initial_reflection_prompt(question, current_answer)
                reflection_prompt = auto_prompt_used
                reflection_full_response = query_model(api_key, reflection_prompt, model)

                if reflection_full_response is None:
                    print(f"Timeout or error in reflection (layer {layer+1})")
                    reflection_data.append({
                        "layer": layer,
                        "score": 0,
                        "response": None,
                        "reflection_prompt": reflection_prompt,
                        "full_response": None,
                        "auto_prompt_used": auto_prompt_used
                    })
                    break

                reanswer_prompt = generate_reanswer_prompt(question, current_answer, reflection_full_response)
                reanswer_full_response = query_model(api_key, reanswer_prompt, model)

                if reanswer_full_response is None:
                    print(f"Timeout or error in reanswer (layer {layer+1})")
                    reflection_data.append({
                        "layer": layer,
                        "score": 0,
                        "response": None,
                        "reflection_prompt": reflection_prompt,
                        "full_response": reflection_full_response,
                        "auto_prompt_used": auto_prompt_used
                    })
                    break

                current_answer = extract_answer_gsm_format(reanswer_full_response)
                current_score = evaluate_response(current_answer, expected_answer)
                
                reflection_data.append({
                    "layer": layer,
                    "score": current_score,
                    "response": current_answer,
                    "reflection_prompt": reflection_prompt,
                    "full_response": reflection_full_response,
                    "auto_prompt_used": auto_prompt_used
                })

                print(f"Reflection (layer {layer+1}) response: {current_answer} - Score: {current_score}")
        else:
            reflection_prompt = generate_initial_reflection_prompt(question, cot_response)
            reflection_full_response = query_model(api_key, reflection_prompt, model)
            reflection_response = extract_answer_gsm_format(reflection_full_response) if reflection_full_response else None
            reflection_score = evaluate_response(reflection_response, expected_answer)
            reflection_data.append({
                "layer": 0,
                "score": reflection_score,
                "response": reflection_response,
                "reflection_prompt": reflection_prompt,
                "full_response": reflection_full_response,
                "auto_prompt_used": None
            })

            print(f"Reflection response: {reflection_response} - Score: {reflection_score}")
            
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
        for layer_data in row['reflection_data']:
            new_row = row.to_dict()
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

def main():
    API_KEY = os.getenv("OPENROUTER_API_KEY")

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    config = load_config()

    datasets = config.get('datasets', ['main'])
    gsm_types = config.get('gsm_types', ['gsm8-std'])
    models = config.get('models', ['meta-llama/llama-3.1-8b-instruct'])

    for dataset_name in datasets:
        sample = prepare_dataset(dataset_name)
        for gsm_type in gsm_types:
            for model in models:
                print(f"\nExecuting test with:")
                print(f"Dataset: {dataset_name}")
                print(f"GSM type: {gsm_type}")
                print(f"Model: {model}")
                print(f"Using Auto-Prompt: {config['use_auto_prompt']}")
                if config["use_auto_prompt"]:
                    print(f"Max Reflection Layers: {config['max_reflection_layers']}")
                    print(f"Auto Prompt Model: {config['auto_prompt_model']}")

                safe_model_name = model.replace("/", "_")
                results_df = run_gsm8(sample, API_KEY, config, model, dataset_name, gsm_type)
                save_results(results_df, f"{results_dir}/results_dataset_{dataset_name}_{gsm_type}_{safe_model_name}.csv")

if __name__ == "__main__":
    main()