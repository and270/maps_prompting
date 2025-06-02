import openai
from dotenv import load_dotenv
import time
import os

load_dotenv(override=True)

def query_model(api_key, prompt, model, supports_sampling_params=True, api_provider="openrouter", max_retries=3, thinking_effort_support=False, reasoning_effort="medium"):
    for attempt in range(max_retries):
        try:
            if api_provider == "openrouter":
                #OPENAI compatible API. Response being called bellow
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    timeout=180.0
                )
            elif api_provider == "openai":
                #OPENAI compatible API. Response being called bellow
                client = openai.OpenAI(
                    api_key=api_key,
                )
            elif api_provider == "deepseek":
                #OPENAI compatible API. Response being called bellow
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com/v1",
                )
            elif api_provider == "google":
                # Not OpenAI compatible. Response being called here
                try:
                    from google import genai
                except ImportError:
                    raise ImportError("Google genai library not installed. Please install with: pip install google-genai")
                
                client = genai.Client(api_key=api_key)
                
                # For Google models, use the generate_content method
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                return response.text if response.text else None
                
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
            
            # Handle OpenAI reasoning models (o3, o4-mini, etc.)
            if api_provider == "openai" and thinking_effort_support:
                # Use the Responses API for reasoning models
                response = client.responses.create(
                    model=model,
                    reasoning={"effort": reasoning_effort},
                    input=[
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ]
                )
                return response.output_text if response.output_text else None
            else:
                # Standard chat completions API for non-reasoning models
                api_params = {
                    "model": model,
                    "messages": [
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