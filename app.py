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
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback

# Load environment variables from .env file
load_dotenv()

# Function to query and stream the response from the LLM
def stream_llm_response(client, model_params):
    response_message = ""

    for chunk in client.chat.completions.create(
        model=model_params.get("model", "gpt-4o-2024-05-13"),
        messages=st.session_state.messages,
        temperature=model_params.get("temperature", 0.3),
        max_tokens=4096,
        stream=True,
    ):
        content = chunk.choices[0].delta.content
        if content:
            response_message += content
            yield content

    # Append the assistant's response to the session state messages
    st.session_state.messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response_message}],
    })

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    return text

def main():
    # --- Page Config ---
    st.set_page_config(
        page_title="Willieam Assistant",
        page_icon="😎",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.html("""<h1 style="text-align: center; color: #6ca395;"><i>Willieam Assistant</i></h1>""")

    # --- Side Bar ---
    with st.sidebar:
        st.image("itb_black.png", width=200)

        # Load API keys and password
        api_key = os.getenv("OPENAI_API_KEY")
        correct_password = os.getenv("CORRECT_PASSWORD")
        password_input = st.text_input("Enter Password to Access AI Settings", type="password")

        # Validate password
        if password_input == correct_password:
            with st.expander("🔐 AI Settings"):
                openai_api_key = st.text_input("Input API Key (https://platform.openai.com/)", value=api_key, type="password")
        else:
            st.warning("Incorrect password. Please try again.")

        # Model selection
        model = st.selectbox("Select a model:", [
            "gpt-4o-2024-05-13",
            "gpt-4o-mini-2024-07-18",
            "gpt-4-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-32k",
        ], index=1)

        # Model parameters
        model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
        model_params = {"model": model, "temperature": model_temp}

    # Check if API key is provided
    if not openai_api_key or "sk-" not in openai_api_key:
        st.warning("⬅️ Please introduce your OpenAI API Key (make sure to have funds) to continue...")
        return

    client = OpenAI(api_key=openai_api_key)

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content["text"])
                elif content["type"] == "image_url":
                    st.image(content["image_url"]["url"])

    # --- PDF Upload Section ---
    st.write("### **📄 Upload a PDF file:**")
    uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_pdf is not None:
        st.write("PDF successfully uploaded!")

        # Extract text from PDF
        pdf_text = extract_text_from_pdf(uploaded_pdf)

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(pdf_text)

        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # User question input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)

    # Chat input
    prompt = st.chat_input("Hi Boss, need help?...") 
    if prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        })

        # Displaying the new messages
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.write_stream(stream_llm_response(client, model_params))

if __name__ == "__main__":
    main()