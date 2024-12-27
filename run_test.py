# Importar bibliotecas necessárias
import pandas as pd
from datasets import load_dataset
import openai

#TODO adicionar um 8-shot conforme o GSM8 original
EIGHT_SHOT_EXAMPLES = """
See the examples bellow to guide you on how your answer format should be:

"""

# Função para carregar e preparar o dataset
def prepare_dataset():
    print("Carregando o dataset...")
    # Carregar as variantes do GSM-Symbolic
    ds_main = load_dataset("apple/GSM-Symbolic", name="main", split="test")
    ds_p1 = load_dataset("apple/GSM-Symbolic", name="p1", split="test")
    ds_p2 = load_dataset("apple/GSM-Symbolic", name="p2", split="test")

    df_main = pd.DataFrame(ds_main) #dataset base do GSM Symbolic, com trocas de elementos conforme template
    df_p1 = pd.DataFrame(ds_p1) # dataset p1 do GSM Symbolic, que inclui 1 cláusula adicional à questão, aumentando o nível de dificuldade.
    df_p2 = pd.DataFrame(ds_p2) # dataset p1 do GSM Symbolic, que inclui 2 cláusulas adicionais à questão, aumentando o nível de dificuldade.

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

# Função para gerar prompt inicial para Self-Reflection
def generate_initial_reflection_prompt(question, answer):
    #TODO Ajustar conforme estudo Self-Reflection e técnicas de Self-Reflection
    return f"""Question: {question}
Your initial answer was: {answer}
You previously answered this question incorrectly. Reflect on why your answer was incorrect and identify the type of error. Then, solve the problem again step-by-step with corrections.
"""

# Função para gerar prompt de re-resposta baseado na reflexão
def generate_reanswer_prompt(question, answer, reflection):
    #TODO Ajustar conforme estudo Self-Reflection e técnicas de Self-Reflection
    return f"""{EIGHT_SHOT_EXAMPLES}

Now, look at this question:
Question: {question}
Your initial answer was: {answer}
Based on the following reflection, solve the problem correctly:
Reflection: {reflection}

Provide your corrected reasoning and answer:
"""

# Conforme instrução dataset gsm-symbolic
def extract_answer_gsm_format(response):
    # Remove commas so for example 5,000 becomes 5000
    model_resp = model_resp.replace(",", "")
    # Find the last number
    extracted_num = re.findall(r"-?\d+\.?\d*", model_resp)[-1]
    # Use float to ensure 3.0 and 3 are the same.
    return float(extracted_num)

# Função para interagir com os modelos usando OpenRouter API
def query_model(api_key, prompt, model="meta-llama/llama-3.1-8b-instruct"):
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
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
        #print(chat_completion.choices[0].message.content.strip())
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erro ao consultar o modelo {model}: {e}")
        return ""

# Função para avaliar respostas
def evaluate_response(response, expected_answer):
    return int(response.strip() == expected_answer.strip())

# Função principal para rodar os experimentos
def run_gsm8(sample, api_key, model="gpt-4", type="gsm8-std"):
    results = []
    for idx, row in sample.iterrows():
        if type == "gsm8-std":
            question = row["original_question"]
            #TODO VERIFICAR ISSO, PORQUE PODE SER QUE, NO DATASET, ESSE CAMPO SEJA DA RESPOSTA OBTIDA NO TESTE E NAO A CORRETA
            #TODO https://huggingface.co/datasets/apple/GSM-Symbolic (ver no cart)
            expected_answer = extract_answer_gsm_format(row["original_answer"])
        elif type == "gsm-symbolic":
            question = row["question"]
            #TODO VERIFICAR ISSO, PORQUE PODE SER QUE, NO DATASET, ESSE CAMPO SEJA DA RESPOSTA OBTIDA NO TESTE E NAO A CORRETA
            #TODO https://huggingface.co/datasets/apple/GSM-Symbolic (ver no cart)
            expected_answer = extract_answer_gsm_format(row["answer"])
        
        #resultado base sem utilizar nenhuma técnica (Baseline)
        base_prompt = question
        base_response = extract_answer_gsm_format(query_model(api_key, base_prompt, model))
        base_score = evaluate_response(base_response, expected_answer)

        # Resposta com CoT
        cot_prompt = generate_cot_prompt(question)
        cot_response = extract_answer_gsm_format(query_model(api_key, cot_prompt, model))
        cot_score = evaluate_response(cot_response, expected_answer)

        # Reflexão e correção (Self-Reflection)
        if cot_score == 1:
            reflection_response = cot_response
            reflection_score = cot_score
        else: #segundo o paper do Self-Reflection, a técnica somente é aplicada quando a resposta inicial não é correta
            #TODO Ajustar conforme estudo Self-Reflection e técnicas de Self-Reflection
            reflection_prompt = generate_initial_reflection_prompt(question, cot_response) #Conforme paper do Self-Reflection, a reflexão vem sobre a resposta em CoT.
            reflection = query_model(api_key, reflection_prompt, model)
            reanswer_prompt = generate_reanswer_prompt(question, base_response, reflection)
            reflection_response = extract_answer_gsm_format(query_model(api_key, reanswer_prompt, model))
            reflection_score = evaluate_response(reflection_response, expected_answer)

        # Armazenar resultados
        #TODO Melhorar registro de resultados com somas, etc... (desse jeito ele está criando um registro por questão)
        results.append({
            "type": type,
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

# Pipeline principal
def main():
    # Configuração do API Key do OpenRouter
    API_KEY = "sua_api_key_aqui"

    #TODO: ajustar para chamar os modelos no projeto de pesquisa, conforme nomeados no OpenRouter
    model_test_list = ["meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.1-70b-instruct"]
    gsm_types = ["gsm8-std", "gsm-symbolic"]
    # Carregar dataset
    sample_main, sample_p1, sample_p2 = prepare_dataset()

    print("Executando testes no dataset main...")
    for gsm_type in gsm_types:
        for model in model_test_list:
            results_df = run_gsm8(sample_main, API_KEY, model, gsm_type)
    
            # Salvar resultados
            save_results(results_df, f"results_dataset_main_{gsm_type}_{model}.csv")

    print("Executando testes no dataset p1...")
    for gsm_type in gsm_types:
        for model in model_test_list:
            results_df = run_gsm8(sample_p1, API_KEY, model, gsm_type)
    
            # Salvar resultados
            save_results(results_df, f"results_dataset_p1_{gsm_type}_{model}.csv")

    print("Executando testes no dataset p2...")
    for gsm_type in gsm_types:
        for model in model_test_list:
            results_df = run_gsm8(sample_p2, API_KEY, model, gsm_type)
    
            # Salvar resultados
            save_results(results_df, f"results_dataset_p2_{gsm_type}_{model}.csv")


if __name__ == "__main__":
    main()
