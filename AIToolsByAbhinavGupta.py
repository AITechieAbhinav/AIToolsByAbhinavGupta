import streamlit as st
from transformers import pipeline

# Use a pipeline as a high-level helper
from transformers import pipeline

st.set_page_config(page_title="AI Tools by Abhinav Gupta", layout="centered")

st.title("📝 AI Tools by Abhinav Gupta")

tab_titles = ["Text Summarizer","Document Summarizer"]

tab1, tab2 = st.tabs(tab_titles)

with tab1:

	text = st.text_input("Paste text below to Summarize")

	if st.button("Submit") :

		summarizer = pipeline("summarization", model="pszemraj/long-t5-tglobal-base-16384-book-summary")
		sm_txt = summarizer(text)
		st.markdown(sm_txt)
