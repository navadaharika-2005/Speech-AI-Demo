import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Speech AI Demo", page_icon="🎙️")

st.title("🎙️ Speech AI Demo")
st.write("Convert spoken English audio into text using AI.")

audio = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "m4a", "aac"]
)

@st.cache_resource
def load_model():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny"
    )

if audio is not None:
    st.audio(audio)

    if st.button("Transcribe Audio"):
        try:
            with st.spinner("AI is listening..."):
                model = load_model()
                result = model(audio.getvalue())
                st.subheader("Recognized Text")
                st.write(result["text"])
                st.success("Transcription completed!")
        except Exception as e:
            st.error(f"Transcription failed: {e}")