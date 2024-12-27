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
    #TODO Ajustar conforme estudo CoT
    return f"""
    You are solving a complex mathematical problem. Think step-by-step and provide intermediate steps for your reasoning.

    Question: {question}

    Provide your reasoning step-by-step:
    """

# Função para gerar prompt inicial para Self-Reflection
def generate_initial_reflection_prompt(question, answer):
    #TODO Ajustar conforme estudo Self-Reflection e técnicas de Self-Reflection
    return f"""
    You previously answered this question incorrectly:
    Question: {question}
    Your initial answer was: {answer}

    Reflect on why your answer was incorrect and identify the type of error. Then, solve the problem again step-by-step with corrections.
    """

# Função para gerar prompt de re-resposta baseado na reflexão
def generate_reanswer_prompt(reflection):
    #TODO Ajustar conforme estudo Self-Reflection e técnicas de Self-Reflection
    return f"""
    Based on the following reflection, solve the problem correctly:
    Reflection: {reflection}

    Provide your corrected reasoning and answer:
    """

def extract_answer_gsm_format(response):
    #TODO: Implementar a extração da resposta no formato GSM
    pass

# Função para interagir com os modelos usando OpenRouter API
def query_model(api_key, prompt, model="gpt-4"):
    #TODO: ajustar para chamar com OpenRouter
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
def run_experiments(sample, api_key, model="gpt-4"):
    results = []
    for idx, row in sample.iterrows():
        question = row["original_question"]
        expected_answer = row["original_answer"]

        #resultado base sem utilizar nenhuma técnica
        base_prompt = question
        base_response = extract_answer_gsm_format(query_model(api_key, base_prompt, model))
        base_score = evaluate_response(base_response, expected_answer)

        # Resposta inicial (Baseline)
        cot_prompt = generate_cot_prompt(question)
        cot_response = extract_answer_gsm_format(query_model(api_key, cot_prompt, model))
        cot_score = evaluate_response(cot_response, expected_answer)

        # Reflexão e correção (Self-Reflection)
        #TODO Ajustar conforme estudo Self-Reflection e técnicas de Self-Reflection
        reflection_prompt = generate_initial_reflection_prompt(question, cot_response)
        reflection = query_model(api_key, reflection_prompt, model)
        reanswer_prompt = generate_reanswer_prompt(reflection)
        reflection_response = extract_answer_gsm_format(query_model(api_key, reanswer_prompt, model))
        reflection_score = evaluate_response(reflection_response, expected_answer)

        # Armazenar resultados
        #TODO Melhorar registro de resultados com somas, etc... (desse jeito ele está criando um registro por questão)
        results.append({
            "model": model,
            "question": question,
            "expected_answer": expected_answer,
            "base_response": base_response,
            "base_score": base_score,
            "cot_response": cot_response,
            "cot_score": cot_score,
            "reflection_response": reflection_response,
            "reflection_score": reflection_score,
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

    #TODO: ajustar para chamar os modelos no projeto de pesquisa, conforme nomeados no OpenRouter
    model_test_list = ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13"]
    # Carregar dataset
    sample = prepare_dataset()

    # Rodar experimentos
    for model in model_test_list:
        results_df = run_experiments(sample, API_KEY, model)

        # Salvar resultados
        save_results(results_df, f"results_{model}.csv")

        # Analisar resultados
        analyze_results(results_df)

if __name__ == "__main__":
    main()
