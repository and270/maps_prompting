import streamlit as st
from run_test import (
    generate_reanswer_prompt,
    prepare_dataset,
    generate_cot_prompt,
    query_model,
    generate_auto_reflection_auto_adapt_prompt,
    extract_answer_gsm_format,
    evaluate_response
)
import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def load_config():
    with open('config.json', 'r') as config_file:
        return json.load(config_file)

def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "reflection_layer" not in st.session_state:
        st.session_state.reflection_layer = 0
    if "previous_answers" not in st.session_state:
        st.session_state.previous_answers = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "model_info" not in st.session_state:
        st.session_state.model_info = None
    if "is_gsm8k_format" not in st.session_state:
        st.session_state.is_gsm8k_format = True

def get_api_key(provider):
    return {
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY")
    }.get(provider, "")

def main():
    st.title("Math Problem Solver with Self-Reflection")
    initialize_session()
    
    config = load_config()
    models = config["models"]
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model",
        options=list(models.keys()),
        format_func=lambda x: f"{x} ({models[x]['provider']})"
    )
    
    # Answer format selection
    st.session_state.is_gsm8k_format = st.checkbox(
        "GSM8K Format (numerical answer)",
        value=st.session_state.is_gsm8k_format,
        help="If checked, expects a numerical final answer. If unchecked, treats the entire response as the answer."
    )
    
    # Store model info in session state
    if selected_model != st.session_state.get("selected_model"):
        st.session_state.selected_model = selected_model
        st.session_state.model_info = models[selected_model]
        st.session_state.reflection_layer = 0

    # Chat input
    if question := st.chat_input("Enter your question:"):
        st.session_state.current_question = question
        st.session_state.reflection_layer = 0
        st.session_state.previous_answers = []
        
        # Initial CoT response
        cot_prompt = generate_cot_prompt(question)
        api_key = get_api_key(st.session_state.model_info["provider"])
        
        response = query_model(
            api_key,
            cot_prompt,
            st.session_state.model_info["name"],
            api_provider=st.session_state.model_info["provider"]
        )
        
        if st.session_state.is_gsm8k_format:
            extracted_answer = extract_answer_gsm_format(response)
        else:
            extracted_answer = response
            
        st.session_state.previous_answers.append(response)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "type": "CoT Answer",
            "answer": extracted_answer
        })

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(f"**{msg['type']}**")
            st.write(msg["content"])
            if st.session_state.is_gsm8k_format and msg.get("answer"):
                st.write(f"Extracted answer: {msg['answer']}")

    # Reflection button logic
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_msg = st.session_state.messages[-1]
        if last_msg["type"] != "Reflection":
            if st.button(f"Run Reflection Layer {st.session_state.reflection_layer + 1}"):
                api_key = get_api_key(st.session_state.model_info["provider"])
                
                # Get previous answers based on format selection
                if st.session_state.is_gsm8k_format:
                    previous_answers = [msg["answer"] for msg in st.session_state.messages 
                                     if msg["role"] == "assistant" and "answer" in msg]
                else:
                    previous_answers = [msg["content"] for msg in st.session_state.messages 
                                     if msg["role"] == "assistant"]
                
                reflection_prompt = generate_auto_reflection_auto_adapt_prompt(
                    st.session_state.current_question,
                    previous_answers,
                    st.session_state.model_info["name"],
                    api_key,
                    st.session_state.model_info["provider"]
                )
                
                reflection_response = query_model(
                    api_key,
                    reflection_prompt,
                    st.session_state.model_info["name"],
                    api_provider=st.session_state.model_info["provider"]
                )
                
                # Generate re-answer with appropriate previous answer format
                reanswer_prompt = generate_reanswer_prompt(
                    st.session_state.current_question,
                    previous_answers[-1],  # Last answer (either extracted or full)
                    reflection_response
                )
                
                final_response = query_model(
                    api_key,
                    reanswer_prompt,
                    st.session_state.model_info["name"],
                    api_provider=st.session_state.model_info["provider"]
                )
                
                if st.session_state.is_gsm8k_format:
                    extracted_answer = extract_answer_gsm_format(final_response)
                else:
                    extracted_answer = final_response
                    
                st.session_state.previous_answers.append(final_response)
                st.session_state.reflection_layer += 1
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "type": f"Reflection Layer {st.session_state.reflection_layer}",
                    "answer": extracted_answer
                })
                
                st.rerun()

if __name__ == "__main__":
    main() 