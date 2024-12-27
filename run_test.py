# Importar bibliotecas necessárias
import pandas as pd
from datasets import load_dataset
import openai

# Função para carregar e preparar o dataset
def prepare_dataset():
    print("Carregando o dataset...")
    gsm_symbolic = load_dataset("apple/GSM-Symbolic", split="test")
    df = pd.DataFrame(gsm_symbolic)

    # Selecionar uma amostra aleatória por "original_id"
    sample = df.groupby("original_id").sample(n=1, random_state=42)
    print(f"Dataset carregado com {len(sample)} questões selecionadas.")
    return sample

# Função para gerar prompt usando Chain of Thought (CoT)
def generate_cot_prompt(question):
    return f"""
    You are solving a complex mathematical problem. Think step-by-step and provide intermediate steps for your reasoning.

    Question: {question}

    Provide your reasoning step-by-step:
    """

# Função para gerar prompt inicial para Self-Reflection
def generate_initial_reflection_prompt(question, answer):
    return f"""
    You previously answered this question incorrectly:
    Question: {question}
    Your initial answer was: {answer}

    Reflect on why your answer was incorrect and identify the type of error. Then, solve the problem again step-by-step with corrections.
    """

# Função para gerar prompt de re-resposta baseado na reflexão
def generate_reanswer_prompt(reflection):
    return f"""
    Based on the following reflection, solve the problem correctly:
    Reflection: {reflection}

    Provide your corrected reasoning and answer:
    """

# Função para interagir com os modelos usando OpenRouter API
def query_model(api_key, prompt, model="gpt-4"):
    try:
        openai.api_key = api_key
        response = openai.Completion.create(
            engine=model, prompt=prompt, max_tokens=256, temperature=0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Erro ao consultar o modelo: {e}")
        return None

# Função para avaliar respostas
def evaluate_response(response, expected_answer):
    return int(response.strip() == expected_answer.strip())

# Função principal para rodar os experimentos
def run_experiments(sample, api_key):
    results = []
    for idx, row in sample.iterrows():
        question = row["original_question"]
        expected_answer = row["original_answer"]

        # Resposta inicial (Baseline)
        initial_prompt = generate_cot_prompt(question)
        initial_response = query_model(api_key, initial_prompt)
        initial_score = evaluate_response(initial_response, expected_answer)

        # Reflexão e correção (Self-Reflection)
        reflection_prompt = generate_initial_reflection_prompt(question, initial_response)
        reflection = query_model(api_key, reflection_prompt)
        reanswer_prompt = generate_reanswer_prompt(reflection)
        final_response = query_model(api_key, reanswer_prompt)
        final_score = evaluate_response(final_response, expected_answer)

        # Armazenar resultados
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "initial_response": initial_response,
            "initial_score": initial_score,
            "reflection": reflection,
            "final_response": final_response,
            "final_score": final_score
        })
    return pd.DataFrame(results)

# Função para salvar resultados
def save_results(results_df, filename="experiment_results.csv"):
    results_df.to_csv(filename, index=False)
    print(f"Resultados salvos em {filename}")

# Função de análise dos resultados
def analyze_results(results_df):
    print("Análise de resultados:")
    accuracy = results_df[["initial_score", "final_score"]].mean()
    print("Acurácia média:")
    print(accuracy)

# Pipeline principal
def main():
    # Configuração do API Key do OpenRouter
    API_KEY = "sua_api_key_aqui"

    # Carregar dataset
    sample = prepare_dataset()

    # Rodar experimentos
    results_df = run_experiments(sample, API_KEY)

    # Salvar resultados
    save_results(results_df)

    # Analisar resultados
    analyze_results(results_df)

if __name__ == "__main__":
    main()
