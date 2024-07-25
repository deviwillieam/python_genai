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
dotenv.load_dotenv()

# Function to query and stream the response from the LLM
def stream_llm_response(client, model_params):
    response_message = ""

    for chunk in client.chat.completions.create(
        model=model_params["model"] if "model" in model_params else "gpt-4o-2024-05-13",
        messages=st.session_state.messages,
        temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
        max_tokens=4096,
        stream=True,
    ):
        response_message += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        yield chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""

    st.session_state.messages.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})


# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()

    return base64.b64encode(img_byte).decode('utf-8')

# Function to convert file to base64
def get_pdf_base64(pdf_file):
    pdf_content = pdf_file.read()
    return base64.b64encode(pdf_content).decode('utf-8')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
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
        # default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
        # Define the correct password
        api_key = os.getenv("OPENAI_API_KEY")
        correct_password = os.getenv("CORRECT_PASSWORD")

        # Prompt the user for the password
        password_input = st.text_input("Enter Password to Access AI Settings", type="password")

        # Check if the entered password matches the correct password
        if password_input == correct_password:
            with st.expander("🔐 AI Settings"):
                with st.container():
                    openai_api_key = st.text_input("Input API (https://platform.openai.com/)", value=api_key,
                                                   type="password")
        else:
            st.warning("Incorrect password. Please try again.")
        with st.popover("✨ Model"):
            model = st.selectbox("Select a model:", [
                "gpt-4o-2024-05-13",
		"gpt-4o-mini-2024-07-18",
		"gpt-4-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-32k",
            ], index=1)

        with st.popover("⚙️ Model parameters"):
            model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

        model_params = {
            "model": model,
            "temperature": model_temp,
        }

    if openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key:
        st.write("#")
        st.warning("⬅️ Please introduce your OpenAI API Key (make sure to have funds) to continue...")


    else:
        client = OpenAI(api_key=openai_api_key)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Displaying the previous messages if there are any
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":
                        st.image(content["image_url"]["url"])

        # Side bar model options and inputs
        with st.sidebar:
            # st.divider()
            # Image Upload
            if model in ["gpt-4o-2024-05-13", "gpt-4-turbo"]:

                audio_response = st.toggle("Audio response", value=False)
                if audio_response:
                    cols = st.columns(2)
                    with cols[0]:
                        tts_voice = st.selectbox("Select a voice:",
                                                 ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
                    with cols[1]:
                        tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=1)
                st.write("### **🖼️ Add an image:**")
                def add_image_to_messages():
                    if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                        img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                        raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                        img = get_image_base64(raw_img)
                        st.session_state.messages.append(
                            {
                                "role": "user",
                                "content": [{
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{img_type};base64,{img}"}
                                }]
                            }
                        )
                cols_img = st.columns(2)

                with cols_img[0]:
                    with st.popover("📁 Upload"):
                        st.file_uploader(
                            "Upload an image",
                            type=["png", "jpg", "jpeg"],
                            accept_multiple_files=False,
                            key="uploaded_img",
                            on_change=add_image_to_messages,
                        )

                with cols_img[1]:
                    with st.popover("📸 Camera"):
                        activate_camera = st.checkbox("Activate camera")
                        if activate_camera:
                            st.camera_input(
                                "Take a picture",
                                key="camera_img",
                                on_change=add_image_to_messages,
                            )
            st.divider()
            # Audio Upload
            st.write("### **🎤 Voice Input:**")

            audio_prompt = None
            if "prev_speech_hash" not in st.session_state:
                st.session_state.prev_speech_hash = None

            speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395", )
            if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
                st.session_state.prev_speech_hash = hash(speech_input)
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.wav", speech_input),
                )

                audio_prompt = transcript.text


            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(
                "🗑️ Reset conversation",
                on_click=reset_conversation,
            )

            st.divider()
            # st.video("https://www.youtube.com/")
            # st.write("📋[Arme Studios](https://armestudios.co.id)")
             st.header("Ask your PDF 😎")

    st.write("### **📄 Upload a PDF file:**")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_pdf is not None:
        st.write("PDF successfully uploaded!")

        # Display uploaded PDF
        pdf_bytes = BytesIO(uploaded_pdf.read())
        st.write("### **PDF Preview:**")
        st.write(pdf_bytes)

        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_bytes)

        # Add PDF text to messages
        st.session_state.messages.append(
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": pdf_text,
                }]
            }
        )

        # Display extracted text
        st.write("### **Extracted Text from PDF:**")
        st.write(pdf_text)

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
        if prompt := st.chat_input("Hi Boss need helps?...") or audio_prompt:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt or audio_prompt,
                    }]
                }
            )

            # Displaying the new messages
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                st.write_stream(
                    stream_llm_response(client, model_params)
                )

            # --- Added Audio Response (optional) ---
            if audio_response:
                response =  client.audio.speech.create(
                    model=tts_model,
                    voice=tts_voice,
                    input=st.session_state.messages[-1]["content"][0]["text"],
                )
                audio_base64 = base64.b64encode(response.content).decode('utf-8')
                audio_html = f"""
                <audio controls autoplay>
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/mp3">
                </audio>
                """
                st.html(audio_html)


if __name__ == "__main__":
    main()

