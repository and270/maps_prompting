# chatbot.py

import streamlit as st
import os
import json
from dotenv import load_dotenv

from llms_api import query_model
from methods import (
    generate_cot_prompt,
    generate_auto_reflection_auto_adapt_prompt,
    generate_reanswer_prompt,
    extract_answer_gsm_format,
    extract_answer_math,
    extract_answer_aime
)

# --- Configuration & Initialization ---

load_dotenv(override=True)

def load_config():
    """Loads config.json."""
    with open('config.json', 'r') as f:
        return json.load(f)

def initialize_session_state():
    """Initializes session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "reflection_layer" not in st.session_state:
        st.session_state.reflection_layer = 0
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "model_info" not in st.session_state:
        st.session_state.model_info = None
    if "benchmark_name" not in st.session_state:
        st.session_state.benchmark_name = "General"
    if "extract_numerical_answer" not in st.session_state:
        st.session_state.extract_numerical_answer = True
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {}

def get_api_key_from_state(api_key_env_var):
    """
    Gets API key, prioritizing user input in session state over environment variables.
    """
    if not api_key_env_var:
        return None
    # Priority 1: User-provided key in the UI
    user_key = st.session_state.api_keys.get(api_key_env_var)
    if user_key:
        return user_key
    # Priority 2: Key from .env file
    return os.getenv(api_key_env_var)

def extract_answer_by_benchmark(response, benchmark_name):
    """Extracts answer from response based on the selected benchmark."""
    if benchmark_name == "MATH":
        return extract_answer_math(response)
    elif benchmark_name == "AIME":
        return extract_answer_aime(response)
    # Default to GSM/General extraction logic
    elif benchmark_name in ["gsm-symbolic", "General"]:
         if st.session_state.get("extract_numerical_answer", True):
            return extract_answer_gsm_format(response)
         else:
            return response # Return the full response if not extracting a number
    else:
        # Fallback for other gsm_types or unforeseen cases
        return extract_answer_gsm_format(response)

# --- UI Components ---

def setup_sidebar(config):
    """Sets up the Streamlit sidebar for user inputs."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        models = config.get("models", {})
        if not models:
            st.error("No models found in `config.json`.")
            st.stop()
        
        # --- Model Selection ---
        selected_model_key = st.selectbox(
            "Select Model",
            options=list(models.keys()),
            format_func=lambda key: f"{key} ({models[key].get('provider', 'N/A')})",
            key="selected_model_key"
        )
        st.session_state.model_info = models[selected_model_key]

        # --- Benchmark Selection ---
        st.session_state.benchmark_name = st.selectbox(
            "Select Benchmark",
            options=["General", "gsm-symbolic", "MATH", "AIME"],
            index=0, # Default to "General"
            help="Select the format for prompts and answer extraction. 'General' is for open-ended questions."
        )

        if st.session_state.benchmark_name == "General":
            st.session_state.extract_numerical_answer = st.checkbox(
                "Extract Final Number",
                value=True,
                help="For 'General' mode, try to extract a final numerical answer. If unchecked, the whole response is the answer."
            )

        # --- API Key Management ---
        st.header("🔑 API Keys")
        st.markdown("Keys from `.env` are pre-filled. You can override them here for the session.")
        
        # Dynamically find all required API key env vars from config
        required_api_envs = {
            model_data.get("api_key_env")
            for model_data in models.values()
            if model_data.get("api_key_env")
        }
        # Also add the auto_prompt_model's key if it's different
        auto_prompt_model_key = config.get("auto_prompt_model")
        if auto_prompt_model_key and auto_prompt_model_key != "same" and auto_prompt_model_key in models:
            required_api_envs.add(models[auto_prompt_model_key].get("api_key_env"))
            
        for env_var in sorted(list(required_api_envs)):
            key_from_env = os.getenv(env_var)
            user_input = st.text_input(
                f"`{env_var}`",
                type="password",
                value=key_from_env if key_from_env else "",
                help=f"Enter key for models using `{env_var}`"
            )
            st.session_state.api_keys[env_var] = user_input

def display_chat_and_controls():
    """Displays the chat history and action buttons."""
    if st.session_state.current_question:
        st.info(f"**Current Question:** {st.session_state.current_question}")
        if st.button("Ask a New Question"):
            reset_chat()
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(f"**{msg['type']}**")
            st.markdown(msg["content"])
            if msg.get("extracted_answer") is not None:
                st.code(f"Extracted Answer: {msg['extracted_answer']}", language=None)

    # Show reflection button if the last message was an answer
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        if st.button(f"🧠 Run Reflection Layer {st.session_state.reflection_layer + 1}", use_container_width=True):
            handle_reflection()

# --- Core Logic ---

def reset_chat():
    """Resets the chat state for a new question."""
    st.session_state.messages = []
    st.session_state.reflection_layer = 0
    st.session_state.current_question = None

def handle_initial_question(config):
    """Handles the user's initial question input and gets the CoT answer."""
    if question := st.chat_input("Enter your math problem or question:"):
        st.session_state.current_question = question
        
        main_model_info = st.session_state.model_info
        api_key_env = main_model_info.get("api_key_env")
        api_key = get_api_key_from_state(api_key_env)

        if not api_key:
            st.error(f"⚠️ API key for `{api_key_env}` is missing. Please set it in the sidebar or your `.env` file.")
            return

        with st.spinner("Thinking... (Generating initial answer)"):
            cot_prompt = generate_cot_prompt(question, st.session_state.benchmark_name)
            
            full_response = query_model(
                api_key=api_key,
                prompt=cot_prompt,
                model=main_model_info["name"],
                api_provider=main_model_info["provider"],
                supports_sampling_params=main_model_info.get("supports_sampling_params", True),
                temperature=config.get("answer_temperature", 0.7),
                top_p=config.get("answer_top_p", 1.0)
            )

        if full_response:
            extracted_answer = extract_answer_by_benchmark(full_response, st.session_state.benchmark_name)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "type": "Initial Answer (CoT)",
                "extracted_answer": extracted_answer
            })
            st.rerun()
        else:
            st.error("Failed to get a response from the model.")

def handle_reflection():
    """Handles the multi-layered self-reflection process."""
    config = load_config()
    main_model_info = st.session_state.model_info
    
    # --- 1. Determine Meta-Prompting Model ---
    auto_prompt_config_key = config.get("auto_prompt_model", "same")
    if auto_prompt_config_key == "same" or auto_prompt_config_key not in config["models"]:
        auto_prompt_model_info = main_model_info
    else:
        auto_prompt_model_info = config["models"][auto_prompt_config_key]

    # --- 2. Get API Keys ---
    main_api_key = get_api_key_from_state(main_model_info.get("api_key_env"))
    auto_prompt_api_key = get_api_key_from_state(auto_prompt_model_info.get("api_key_env"))

    if not main_api_key or not auto_prompt_api_key:
        st.error("⚠️ API key for the main model or meta-model is missing.")
        return

    # --- 3. Generate Adaptive Reflection Instructions ---
    with st.spinner(f"Generating reflection instructions with `{auto_prompt_model_info['name']}`..."):
        previous_incorrect_answers = [
            msg["extracted_answer"] for msg in st.session_state.messages if msg["role"] == "assistant"
        ]
        
        reflection_instructions_prompt = generate_auto_reflection_auto_adapt_prompt(
            question=st.session_state.current_question,
            previous_incorrect_answers=previous_incorrect_answers,
            auto_prompt_model_name=auto_prompt_model_info["name"],
            auto_prompt_api_key=auto_prompt_api_key,
            auto_prompt_api_provider=auto_prompt_model_info["provider"],
            auto_prompt_supports_sampling_params=auto_prompt_model_info.get("supports_sampling_params", True),
            auto_prompt_thinking_effort_support=auto_prompt_model_info.get("thinking_effort_support", False),
            auto_prompt_reasoning_effort=auto_prompt_model_info.get("reasoning_effort", "medium"),
            auto_prompt_temperature=config.get("meta_prompt_generation_temperature", 1.0),
            auto_prompt_top_p=config.get("meta_prompt_generation_top_p", 1.0)
        )

    if not reflection_instructions_prompt:
        st.error("Failed to generate reflection instructions.")
        return

    # --- 4. Generate the Reflection Text using Main Model ---
    with st.spinner(f"Generating reflection text with `{main_model_info['name']}`..."):
        reflection_actual_text = query_model(
            api_key=main_api_key,
            prompt=reflection_instructions_prompt,
            model=main_model_info["name"],
            api_provider=main_model_info["provider"],
            supports_sampling_params=main_model_info.get("supports_sampling_params", True),
            temperature=config.get("answer_temperature", 0.7),
            top_p=config.get("answer_top_p", 1.0)
        )

    if not reflection_actual_text:
        st.error("Main model failed to generate reflection text.")
        return

    # --- 5. Generate the Re-Answer ---
    with st.spinner("Generating refined answer..."):
        last_answer = st.session_state.messages[-1]["extracted_answer"]
        reanswer_prompt = generate_reanswer_prompt(st.session_state.current_question, last_answer, reflection_actual_text)
        
        final_response = query_model(
            api_key=main_api_key,
            prompt=reanswer_prompt,
            model=main_model_info["name"],
            api_provider=main_model_info["provider"],
            supports_sampling_params=main_model_info.get("supports_sampling_params", True),
            temperature=config.get("answer_temperature", 0.7),
            top_p=config.get("answer_top_p", 1.0)
        )

    if final_response:
        st.session_state.reflection_layer += 1
        new_extracted_answer = extract_answer_by_benchmark(final_response, st.session_state.benchmark_name)
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_response,
            "type": f"Refined Answer (Layer {st.session_state.reflection_layer})",
            "extracted_answer": new_extracted_answer
        })
        st.rerun()
    else:
        st.error("Failed to get a refined answer from the model.")

# --- Main App ---

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("🤖 MAPS: Math Solver with Multi-Layered Self-Reflection")
    
    initialize_session_state()
    config = load_config()
    
    setup_sidebar(config)
    
    display_chat_and_controls()
    
    if not st.session_state.current_question:
        handle_initial_question(config)

if __name__ == "__main__":
    main()