import streamlit as st
from openai import OpenAI
import os

def run():
    """Lab 3 – Chatbot with OpenAI"""
    # Page setup
    st.set_page_config(page_title="Lab 3 – Chatbot", page_icon="🤖", layout="wide")
    st.title("Lab 3 – OpenAI Chatbot 🤖")
    st.write("Chat with an AI assistant. The conversation is remembered in your session.")

    
    try:
        API = st.secrets["OPENAPI_KEY"]  
    except Exception:
        API = os.getenv("OPENAPI_KEY")

    if not API:
        st.error(" No API key found. Please set OPENAPI_KEY in `.streamlit/secrets.toml` or environment.")
        return

    client = OpenAI(api_key=API)

    # Sidebar
    st.sidebar.header("Options")
    use_advanced = st.sidebar.checkbox("Use Advanced Model (GPT-4o)")
    model_name = "gpt-4o" if use_advanced else "gpt-4o-mini"
    st.sidebar.write(f"**Current model:** {model_name}")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=st.session_state["messages"],
                    stream=True,
                )
                response = st.write_stream(stream)

        st.session_state["messages"].append({"role": "assistant", "content": response})