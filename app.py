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

# Retrieve API keys from Streamlit secrets
api_keys = {
    "openai": st.secrets["OPENAI_API_KEY"],
    "deepseek": st.secrets["Deepseek_API_KEY"]
}

correct_password = st.secrets["CORRECT_PASSWORD"]

# Function to query and stream the response from the LLM
def stream_llm_response(client, model_params):
    response_message = ""

    for chunk in client.chat.completions.create(
        model=model_params["model"],
        messages=st.session_state.messages,
        temperature=model_params["temperature"],
        max_tokens=4096,
        stream=True,
    ):
        response_message += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        yield chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""

    st.session_state.messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response_message}]
    })

# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    return text

def main():
    st.set_page_config(
        page_title="Willieam Assistant",
        page_icon="😎",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.html("""<h1 style="text-align: center; color: #6ca395;"><i>Willieam Assistant</i></h1>""")

    with st.sidebar:
        st.image("itb_black.png", width=200)

        password_input = st.text_input("Enter Password to Access AI Settings", type="password")

        if password_input == correct_password:
            with st.expander("🔐 AI Settings"):
                with st.container():
                    model_provider = st.selectbox("Select Model Provider", ["OpenAI", "DeepSeek"])
                    api_key = api_keys["openai"] if model_provider == "OpenAI" else api_keys["deepseek"]
                    st.text_input("API Key", value=api_key, type="password")
        else:
            st.warning("Incorrect password. Please try again.")

        model = st.selectbox("Select a model:", [
            "gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18", "gpt-4-turbo", "gpt-3.5-turbo-16k",
            "gpt-4", "gpt-4-32k", "deepseek-chat", "deepseek-reasoner"
        ], index=1)

        model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

        model_params = {"model": model, "temperature": model_temp}

    if not api_key or "sk-" not in api_key:
        st.warning("⬅️ Please provide a valid API Key to continue...")
    else:
        if "deepseek" in model:
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")  # Correct API base URL
else:
    client = OpenAI(api_key=api_key)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":
                        st.image(content["image_url"]["url"])

        if prompt := st.chat_input("Hi Boss need help?..."):
            st.session_state.messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                st.write_stream(stream_llm_response(client, model_params))

if __name__ == "__main__":
    main()
