import openai
from dotenv import load_dotenv
import time

load_dotenv(override=True)

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