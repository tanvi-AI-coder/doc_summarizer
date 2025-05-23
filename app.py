import asyncio
import sys

# Windows fix for asyncio + torch + streamlit
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
import os
from transformers import pipeline
import streamlit as st
import time

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")


# Load the summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_file(file):
    text = file.read().decode('utf-8')
    max_chunk_size = 1024
    text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

    summarized_text = ""
    for chunk in text_chunks:
        summary = summarizer(chunk)
        summarized_text += summary[0]['summary_text'] + "\n"

    return summarized_text


st.title("Text Summarizer")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    file_content = uploaded_file.read().decode("utf-8")
    st.subheader("File Content:")
    st.write(file_content)

    if st.button("Summarize"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Generating summary... Please wait.")

        for i in range(1, 101):
            time.sleep(0.05)
            progress_bar.progress(i)

       
        from io import BytesIO
        summary = summarize_file(BytesIO(file_content.encode("utf-8")))

        status_text.text("Summary generated!")
        st.subheader("Summary:")
        st.write(summary)
