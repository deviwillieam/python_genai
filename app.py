import streamlit as st
from openai import OpenAI, error as openai_error
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
from PyPDF2 import PdfReader
from dotenv import load_dotenv

audio_response = False
tts_model = None
tts_voice = None

# Load environment variables from .env file
load_dotenv()
dotenv.load_dotenv()


def stream_llm_response(client, model_params):
    response_message = ""

    try:
        for chunk in client.chat.completions.create(
            model=model_params.get("model", "gpt-4o-2024-05-13"),
            messages=st.session_state.messages,
            temperature=model_params.get("temperature", 0.3),
            max_tokens=1000,  # safer token limit, adjust as needed
            stream=True,
        ):
            delta = chunk.choices[0].delta
            content = delta.get("content", "") if isinstance(delta, dict) else ""
            response_message += content
            yield content

        # Append the full assistant message after streaming finishes
        st.session_state.messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response_message,
                }
            ]
        })

    except openai_error.RateLimitError:
        yield "\n\n**Rate limit exceeded. Please wait before retrying.**\n\n"
    except Exception as e:
        yield f"\n\n**Error: {str(e)}**\n\n"


def get_image_base64(image_raw):
    buffered = BytesIO()
    # Save image as PNG always to avoid format issues
    image_raw.save(buffered, format="PNG")
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')


def get_pdf_base64(pdf_file):
    pdf_content = pdf_file.read()
    return base64.b64encode(pdf_content).decode('utf-8')


def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def main():
    global audio_response, tts_model, tts_voice  # if you want to reuse globals
    audio_response = False
    tts_model = None
    tts_voice = None

    st.set_page_config(
        page_title="Willieam Assistant",
        page_icon="😎",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.markdown("""<h1 style="text-align: center; color: #6ca395;"><i>Willieam Assistant</i></h1>""", unsafe_allow_html=True)

    # Load API key and password from environment at the start of main()
    api_key_env = os.getenv("OPENAI_API_KEY")
    correct_password = os.getenv("CORRECT_PASSWORD")

    openai_api_key = None  # to be set after password input

    with st.sidebar:
        st.image("itb_black.png", width=200)

        password_input = st.text_input("Enter Password to Access AI Settings", type="password")

        if password_input == correct_password:
            with st.expander("🔐 AI Settings", expanded=True):
                openai_api_key = st.text_input(
                    "Input OpenAI API Key (https://platform.openai.com/)",
                    value=api_key_env if api_key_env else "",
                    type="password"
                )
        elif password_input != "":
            st.warning("Incorrect password. Please try again.")

        with st.expander("✨ Model Selection"):
            model = st.selectbox("Select a model:", [
                "gpt-4.5-preview-2025-02-27",
                "gpt-5.1",
                "gpt-4o-2024-08-06",
                "gpt-4o-2024-05-13",
                "gpt-4o-mini-2024-07-18",
                "gpt-4-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-32k",
            ], index=1)

        with st.expander("⚙️ Model Parameters"):
            model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

        model_params = {
            "model": model,
            "temperature": model_temp,
        }

    if not openai_api_key or not openai_api_key.startswith("sk-"):
        st.warning("⬅️ Please introduce your OpenAI API Key (make sure to have funds) to continue...")
        return

    client = OpenAI(api_key=openai_api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content["text"])
                elif content["type"] == "image_url":
                    st.image(content["image_url"]["url"])

    # Sidebar additional features: Image upload, audio input, PDF upload
    with st.sidebar:
        # Image upload for specific models
        if model in ["gpt-4o-2024-05-13", "gpt-4-turbo"]:
            audio_response = st.checkbox("Audio response", value=False)

            if audio_response:
                cols = st.columns(2)
                with cols[0]:
                    tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
                with cols[1]:
                    tts_model = st.selectbox("Select a TTS model:", ["tts-1", "tts-1-hd"], index=1)

            st.markdown("### **🖼️ Add an image:**")

            def add_image_to_messages():
                uploaded_img = st.session_state.get("uploaded_img")
                camera_img = st.session_state.get("camera_img")
                if uploaded_img or camera_img:
                    img_file = uploaded_img or camera_img
                    img_type = img_file.type if uploaded_img else "image/jpeg"
                    raw_img = Image.open(img_file)
                    img_b64 = get_image_base64(raw_img)
                    st.session_state.messages.append({
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{img_b64}"}
                        }]
                    })

            cols_img = st.columns(2)
            with cols_img[0]:
                uploaded_img = st.file_uploader(
                    "Upload an image",
                    type=["png", "jpg", "jpeg"],
                    key="uploaded_img",
                    on_change=add_image_to_messages,
                    help="Upload an image file"
                )
            with cols_img[1]:
                activate_camera = st.checkbox("Activate camera")
                if activate_camera:
                    camera_img = st.camera_input(
                        "Take a picture",
                        key="camera_img",
                        on_change=add_image_to_messages,
                    )

        st.markdown("---")

        # Audio input via microphone recording
        st.markdown("### **🎤 Voice Input:**")

        audio_prompt = None
        if "prev_speech_hash" not in st.session_state:
            st.session_state.prev_speech_hash = None

        speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395")
        if speech_input and hash(speech_input) != st.session_state.prev_speech_hash:
            st.session_state.prev_speech_hash = hash(speech_input)
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", speech_input),
            )
            audio_prompt = transcript.text

        def reset_conversation():
            if "messages" in st.session_state:
                st.session_state.pop("messages", None)

        st.button(
            "🗑️ Reset conversation",
            on_click=reset_conversation,
        )

        st.markdown("---")

        # PDF upload and text extraction
        st.markdown("### **📄 Upload a PDF file:**")
        uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

        if uploaded_pdf is not None:
            # Need to reset buffer position after reading for PyPDF2
            uploaded_pdf.seek(0)
            pdf_text = extract_text_from_pdf(uploaded_pdf)

            if pdf_text.strip() == "":
                st.warning("No extractable text found in the PDF.")
            else:
                # Add PDF text as user message once
                if len(st.session_state.messages) == 0 or \
                   (st.session_state.messages[-1]["content"][0].get("text") != pdf_text):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": pdf_text,
                        }]
                    })

                st.markdown("### **Extracted Text from PDF:**")
                st.write(pdf_text)

    # Chat input and send message to the model
    user_prompt = st.chat_input("Hi Boss need helps?...")
    
    if user_prompt or audio_prompt:
        prompt_text = user_prompt if user_prompt else audio_prompt

        # Append user message to session state messages
        st.session_state.messages.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": prompt_text,
            }]
        })

        # Show user message in chat UI
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # Placeholder for assistant response stream
        assistant_message_placeholder = st.empty()

        assistant_full_response = ""

        with st.chat_message("assistant"):
            for chunk in stream_llm_response(client, model_params):
                assistant_full_response += chunk
                # Update the message incrementally as it streams
                assistant_message_placeholder.markdown(assistant_full_response + "▌")

            # Final update without cursor symbol after streaming ends
            assistant_message_placeholder.markdown(assistant_full_response)

        # Optional: Convert assistant response text to speech and play audio
        if audio_response and assistant_full_response.strip() != "":
            try:
                response_audio = client.audio.speech.create(
                    model=tts_model,
                    voice=tts_voice,
                    input=assistant_full_response,
                )
                audio_base64 = base64.b64encode(response_audio.content).decode('utf-8')
                audio_html = f"""
                    <audio controls autoplay>
                        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    """
                st.markdown(audio_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Audio generation error: {e}")


if __name__ == "__main__":
    main()
