import streamlit as st
from openai import OpenAI
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
dotenv.load_dotenv()

# Function to initialize the OpenAI or DeepSeek client
def get_client(api_key, provider):
    if provider == "deepseek":
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return OpenAI(api_key=api_key)

# Function to query and stream the response from the LLM
def stream_llm_response(client, model_params):
    response_message = ""
    
    for chunk in client.chat.completions.create(
        model=model_params["model"],
        messages=st.session_state.messages,
        temperature=model_params.get("temperature", 0.3),
        max_tokens=4096,
        stream=True,
    ):
        response_message += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        yield chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""

    st.session_state.messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response_message}]
    })

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    return text


def main():
    st.set_page_config(page_title="Willieam Assistant", page_icon="😎", layout="centered")
    
    st.html("""<h1 style="text-align: center; color: #6ca395;"><i>Willieam Assistant</i></h1>""")
    
    with st.sidebar:
        st.image("itb_black.png", width=200)
        
        api_key = os.getenv("OPENAI_API_KEY")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        
        provider = st.radio("Select AI Provider:", ["OpenAI", "DeepSeek"], index=0).lower()
        
        if provider == "deepseek":
            openai_api_key = deepseek_api_key
        else:
            openai_api_key = api_key
        
        if not openai_api_key:
            st.warning("⬅️ Please enter your API key to continue...")
            return
        
        client = get_client(openai_api_key, provider)
        
        model_list = {
            "openai": ["gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18", "gpt-4-turbo", "gpt-3.5-turbo-16k"],
            "deepseek": ["deepseek-chat", "deepseek-reasoner"]
        }
        
        model = st.selectbox("Select a model:", model_list[provider], index=0)
        model_temp = st.slider("Temperature", 0.0, 2.0, 0.3, 0.1)
        model_params = {"model": model, "temperature": model_temp}
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"][0]["text"])
    
    if prompt := st.chat_input("Hi Boss, need help?"):
        st.session_state.messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            st.write_stream(stream_llm_response(client, model_params))

if __name__ == "__main__":
    main()
