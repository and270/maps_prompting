# Importar bibliotecas necessárias
import pandas as pd
from datasets import load_dataset
import openai
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

EIGHT_SHOT_EXAMPLES = """
See the examples bellow to guide you on how your answer format should be:

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

# Função para carregar e preparar o dataset
def prepare_dataset():
    print("Carregando o dataset...")
    # Carregar as variantes do GSM-Symbolic
    ds_main = load_dataset("apple/GSM-Symbolic", "main")
    ds_p1 = load_dataset("apple/GSM-Symbolic", "p1")
    ds_p2 = load_dataset("apple/GSM-Symbolic", "p2")

    # Convert to DataFrame using the 'test' split
    df_main = pd.DataFrame(ds_main['test'])  # Specify the 'test' split
    df_p1 = pd.DataFrame(ds_p1['test'])
    df_p2 = pd.DataFrame(ds_p2['test'])

    # Selecionar uma amostra aleatória por "original_id" para cada variante
    sample_main = df_main.groupby("original_id").sample(n=1, random_state=42)
    sample_p1 = df_p1.groupby("original_id").sample(n=1, random_state=42)
    sample_p2 = df_p2.groupby("original_id").sample(n=1, random_state=42)

    print(f"Datasets carregados. Tamanhos: main={len(sample_main)}, p1={len(sample_p1)}, p2={len(sample_p2)}.")
    return sample_main, sample_p1, sample_p2

# Função para gerar prompt usando Chain of Thought (CoT)
def generate_cot_prompt(question):
    #Conforme paper do CoT, a tpecnica envolve o use de few shot com respostas em cadeia de pensamento e estímulo para desenvolver a resposta passo a passo:
    return f"""{EIGHT_SHOT_EXAMPLES}
Now, look at this question:
Q: {question}
A: Let's think step by step..."""

# Função para gerar prompt inicial para Self-Reflection. Utiliza o agente "Composite", com a junção de todas as técnicas (Retry, Keywords, Advice, Explanation, Instructions, Solution)
# Para melhor encaixe no formato de resposta do GSM8, o exemplo deixa por último a técnica de Solution, que irá imprimir como último número a solução.
# Além disso, o exemplo foi adaptado, trocando-se a resposta de uma questão de múltipla escolha para a resposta direta do número, conforme formato GSM8
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

# Função para gerar prompt de re-resposta baseado na reflexão
def generate_reanswer_prompt(question, answer, reflection):
    #Conforme paper do Self-Reflection, mas ajustado para o formato GSM8
    return f"""{EIGHT_SHOT_EXAMPLES}

Now, look at this question:
Question: {question}
Your initial answer was: {answer}
You previously answered this question incorrectly.
Then you reflected on the problem, your solution, and the correct answer::
Reflection: {reflection}

Provide your corrected reasoning and answer in the examples format.
"""

# Conforme instrução dataset gsm-symbolic
def extract_answer_gsm_format(response):
    try:
        if not response:  # Handle empty responses
            return None
        
        # Remove commas so for example 5,000 becomes 5000
        response = response.replace(",", "")
        # Find all numbers in the response
        numbers = re.findall(r"-?\d+\.?\d*", response)
        
        if not numbers:  # If no numbers found
            return None
            
        # Return the last number found
        return float(numbers[-1])
    except Exception as e:
        print(f"Error in extract_answer_gsm_format: {e}")
        return None

# Função para interagir com os modelos usando OpenRouter API
def query_model(api_key, prompt, model="meta-llama/llama-3.1-8b-instruct"):
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0  # 120 seconds timeout
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
            temperature=0, #conforme instrução de reprodução do GSM Symbolic.
            top_p=1,  #conforme instrução de reprodução do GSM Symbolic.
        )
        response = chat_completion.choices[0].message.content.strip()
        return response if response else None
    except Exception as e:
        print(f"Erro ao consultar o modelo {model}: {e}")
        return None

# Função para avaliar respostas
def evaluate_response(response, expected_answer):
    try:
        # If response is None, it means there was an error in extraction
        if response is None:
            return 0
        # Convert both to float for comparison
        return int(float(response) == float(expected_answer))
    except Exception as e:
        print(f"Error in evaluate_response: {e}")
        return 0

# Função principal para rodar os experimentos
def run_gsm8(sample, api_key, model="meta-llama/llama-3.1-8b-instruct", type="gsm8-std"):
    results = []
    for idx, row in sample.iterrows():
        if type == "gsm8-std":
            question = row["original_question"]
            expected_answer = extract_answer_gsm_format(row["original_answer"])
        elif type == "gsm-symbolic":
            question = row["question"]
            expected_answer = extract_answer_gsm_format(row["answer"])
        
        print("Expected answer: ", expected_answer)
        
        #resultado base sem utilizar nenhuma técnica (Baseline)
        base_prompt = question
        base_full_response = query_model(api_key, base_prompt, model)
        if base_full_response is None:
            print("Timeout or error in base response")
            base_response = None
            base_score = 0
        else:
            base_response = extract_answer_gsm_format(base_full_response)
            base_score = evaluate_response(base_response, expected_answer)

        print(f"Base response: {base_response} - Score: {base_score}")

        # Resposta com CoT
        cot_prompt = generate_cot_prompt(question)
        cot_full_response = query_model(api_key, cot_prompt, model)
        if cot_full_response is None:
            print("Timeout or error in CoT response")
            cot_response = None
            cot_score = 0
        else:
            cot_response = extract_answer_gsm_format(cot_full_response)
            cot_score = evaluate_response(cot_response, expected_answer)

        print(f"COT response: {cot_response} - Score: {cot_score}")
        # Reflexão e correção (Self-Reflection)
        if cot_score == 1:
            reflection_response = cot_response
            reflection_score = cot_score
            reflection_full_response = ""
        else: #segundo o paper do Self-Reflection, a técnica somente é aplicada quando a resposta inicial não é correta
            initial_reflection_prompt = generate_initial_reflection_prompt(question, cot_response) #Conforme paper do Self-Reflection, a reflexão vem sobre a resposta em CoT.
            initial_reflection_full_response = query_model(api_key, initial_reflection_prompt, model)
            if initial_reflection_full_response is None:
                print("Timeout or error in reflection")
                reflection_full_response = None
                reflection_response = None
                reflection_score = 0
            else:
                reflection_prompt = generate_reanswer_prompt(question, base_response, reflection_full_response)
                reflection_full_response = query_model(api_key, reflection_prompt, model)
                if reflection_full_response is None:
                    print("Timeout or error in reflection")
                    reflection_response = None
                    reflection_score = 0
                else:
                    reflection_response = extract_answer_gsm_format(reflection_full_response)
                    reflection_score = evaluate_response(reflection_response, expected_answer)

            print(f"Reflection response: {reflection_response} - Score: {reflection_score}")

        results.append({
            "type": type,
            "model": model,
            "question": question,
            "expected_answer": expected_answer,
            "base_full_response": base_full_response,   
            "base_response": base_response,
            "base_score": base_score,
            "cot_full_response": cot_full_response,
            "cot_response": cot_response,
            "cot_score": cot_score,
            "reflection_full_response": reflection_full_response,
            "reflection_response": reflection_response,
            "reflection_score": reflection_score,
        })
    return pd.DataFrame(results)

# Função para salvar resultados
def save_results(results_df, filename="experiment_results.csv"):
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    results_df.to_csv(filename, index=False)
    print(f"Resultados salvos em {filename}")

# Pipeline principal
def main():
    # Configuração do API Key do OpenRouter
    API_KEY = os.getenv("OPENROUTER_API_KEY")

    # Load dataset
    sample_main, sample_p1, sample_p2 = prepare_dataset()

    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # User input for dataset type
    print("\nSelect dataset type:")
    print("1. Main dataset")
    print("2. P1 dataset")
    print("3. P2 dataset")
    while True:
        dataset_choice = input("Enter your choice (1-3): ").strip()
        if dataset_choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please select 1, 2, or 3.")

    # User input for GSM type
    print("\nSelect GSM type:")
    print("1. Regular GSM8")
    print("2. GSM Symbolic")
    while True:
        gsm_choice = input("Enter your choice (1-2): ").strip()
        if gsm_choice in ['1', '2']:
            break
        print("Invalid choice. Please select 1 or 2.")

    # User input for model
    print("\nSelect model:")
    print("1. meta-llama/llama-3.1-8b-instruct")
    print("2. meta-llama/llama-3.1-70b-instruct")
    print("3. Enter custom model")
    while True:
        model_choice = input("Enter your choice (1-3): ").strip()
        if model_choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please select 1, 2, or 3.")

    # Map choices to values
    dataset_map = {
        '1': ('main', sample_main),
        '2': ('p1', sample_p1),
        '3': ('p2', sample_p2)
    }
    gsm_map = {
        '1': 'gsm8-std',
        '2': 'gsm-symbolic'
    }
    model_map = {
        '1': 'meta-llama/llama-3.1-8b-instruct',
        '2': 'meta-llama/llama-3.1-70b-instruct'
    }

    # Get dataset name and sample
    dataset_name, sample = dataset_map[dataset_choice]
    gsm_type = gsm_map[gsm_choice]

    # Get model
    if model_choice == '3':
        model = input("\nEnter the model name: ").strip()
    else:
        model = model_map[model_choice]

    # Execute test with selected parameters
    print(f"\nExecuting test with:")
    print(f"Dataset: {dataset_name}")
    print(f"GSM type: {gsm_type}")
    print(f"Model: {model}")

    # Use a safe filename by replacing / with _
    safe_model_name = model.replace("/", "_")
    results_df = run_gsm8(sample, API_KEY, model, gsm_type)
    save_results(results_df, f"{results_dir}/results_dataset_{dataset_name}_{gsm_type}_{safe_model_name}.csv")


if __name__ == "__main__":
    main()
