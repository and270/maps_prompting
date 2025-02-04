import streamlit as st
from llms_api import query_model
from methods import extract_answer_gsm_format, generate_auto_reflection_auto_adapt_prompt, generate_cot_prompt, generate_reanswer_prompt
import json
import os
from dotenv import load_dotenv
import asyncio

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
    api_keys = {
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY")
    }
    key = api_keys.get(provider)
    if not key:
        raise ValueError(f"Missing API key for provider: {provider}")
    return key

def main():
    st.title("Math Problem Solver with Self-Reflection")
    initialize_session()
    
    config = load_config()
    models = config["models"]
    
    selected_model = st.selectbox(
        "Select Model",
        options=list(models.keys()),
        format_func=lambda x: f"{x} ({models[x]['provider']})"
    )
    
    try:
        provider = models[selected_model]["provider"]
        get_api_key(provider)
    except ValueError as e:
        st.error(f"⚠️ {str(e)}")
        st.stop()
    
    st.session_state.is_gsm8k_format = st.checkbox(
        "GSM8K Format (numerical answer)",
        value=st.session_state.is_gsm8k_format,
        help="If checked, expects a numerical final answer. If unchecked, treats the entire response as the answer."
    )
    
    if selected_model != st.session_state.get("selected_model"):
        st.session_state.selected_model = selected_model
        st.session_state.model_info = models[selected_model]
        st.session_state.reflection_layer = 0

    # Show either chat input or new question button
    if not st.session_state.messages:
        if question := st.chat_input("Enter your question:"):
            st.session_state.current_question = question
            st.session_state.reflection_layer = 0
            st.session_state.previous_answers = []
            
            api_key = get_api_key(st.session_state.model_info["provider"])
            
            # Create placeholder for response
            response_placeholder = st.empty()
            
            # Show loading message
            with response_placeholder:
                with st.spinner('Thinking... This might take a few seconds.'):
                    if st.session_state.is_gsm8k_format:
                        prompt = generate_cot_prompt(question)
                    else:
                        prompt = question
                    
                    full_response = query_model(
                        api_key,
                        prompt,
                        st.session_state.model_info["name"],
                        api_provider=st.session_state.model_info["provider"]
                    )
            
            if full_response:
                if st.session_state.is_gsm8k_format:
                    extracted_answer = extract_answer_gsm_format(full_response)
                else:
                    extracted_answer = full_response
                
                st.session_state.previous_answers.append(full_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "type": "CoT Answer",
                    "answer": extracted_answer
                })
                st.rerun()
    else:
        if st.button("New Question"):
            st.session_state.messages = []
            st.session_state.reflection_layer = 0
            st.session_state.previous_answers = []
            st.session_state.current_question = None
            st.rerun()

    # Display current question if it exists
    if st.session_state.current_question:
        st.write("**Current Question:**")
        st.info(st.session_state.current_question)

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
            col1, col2 = st.columns([1, 4])
            with col1:
                reflection_clicked = st.button(f"Run Reflection Layer {st.session_state.reflection_layer + 1}")
            
            if reflection_clicked:
                # Create placeholder for reflection process
                reflection_placeholder = st.empty()
                
                with reflection_placeholder:
                    with st.spinner('Starting reflection process...'):
                        api_key = get_api_key(st.session_state.model_info["provider"])
                        
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
                        
                        # Update loading message for reflection
                        reflection_placeholder.empty()
                        with reflection_placeholder:
                            with st.spinner('Generating reflection...'):
                                reflection_response = query_model(
                                    api_key,
                                    reflection_prompt,
                                    st.session_state.model_info["name"],
                                    api_provider=st.session_state.model_info["provider"]
                                )
                
                if reflection_response:
                    if st.session_state.is_gsm8k_format:
                        # Update loading message for reanswer
                        reflection_placeholder.empty()
                        with reflection_placeholder:
                            with st.spinner('Generating final answer...'):
                                reanswer_prompt = generate_reanswer_prompt(
                                    st.session_state.current_question,
                                    previous_answers[-1],
                                    reflection_response
                                )
                                
                                final_response = query_model(
                                    api_key,
                                    reanswer_prompt,
                                    st.session_state.model_info["name"],
                                    api_provider=st.session_state.model_info["provider"]
                                )
                            
                        extracted_answer = extract_answer_gsm_format(final_response)
                    else:
                        final_response = reflection_response
                        extracted_answer = reflection_response
                    
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