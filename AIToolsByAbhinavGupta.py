import streamlit as st
from transformers import pipeline

from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_community.callbacks import get_openai_callback
from PyPDF2 import PdfReader

from gtts import gTTS
from io import BytesIO

# Use a pipeline as a high-level helper
from transformers import pipeline

st.set_page_config(page_title="AI Tools by Abhinav Gupta", layout="centered")

st.title("📝 AI Tools by Abhinav Gupta")

tab_titles = ["Text Summarizer","PDF QnA", "Text to Speech"]

tab1, tab2, tab3 = st.tabs(tab_titles)

with tab1:

	text = st.text_input("Paste text below to Summarize")

	if st.button("Submit") :

		summarizer = pipeline("summarization", model="pszemraj/long-t5-tglobal-base-16384-book-summary")
		sm_txt = summarizer(text)
		st.markdown(sm_txt)

with tab2:

    api_key = st.secrets['api_key']
    
    pdf_file = st.file_uploader("Upload your file", type ="pdf")
    
    #extract text
    
    text=""
    
    if pdf_file is not None:
        reader = PdfReader(pdf_file)
        
        for page_text in reader.pages:
            if page_text:
                text+=page_text.extract_text()
    
    #split text into chunks
    
    text_spliter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function= len
    )
    
    #Save chunks
    data = text_spliter.split_text(text)
    
    if len(data) == 0:
        #st.stop()
		st.warning("Please upload a PDF file to continue.")

    
    #Create embeedings
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    database = FAISS.from_texts(data,embeddings)
    
    #User input
    
    u_input = st.text_input("Please ask questions about PDF file")
    
    llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", huggingfacehub_api_token=api_key)
    
    chat_model = ChatHuggingFace(llm=llm)
    
    if u_input :
        search_result = database.similarity_search(u_input)
        chain = load_qa_chain(chat_model,chain_type="stuff",verbose=True)
    
        with get_openai_callback() as cb:
            response = chain.run(input_documents= search_result, question=u_input)
            print(cb)
    
        st.write(response)

with tab3 :

  st.title("Simple Text to Speech Converter")

  text_input = st.text_area("Enter text to convert to speech", height=150)

  st.sidebar.title("Upload your file")
  uploaded_file = st.sidebar.file_uploader("Choose a .txt file", type="txt")

  if uploaded_file is not None:
     file_text = uploaded_file.read().decode("utf-8")
     st.subheader("Text from Uploaded file")
     st.text(file_text)
     text_input += "\n\n" + file_text

  language = st.selectbox("Select language", ["en", "fr", "ru", "hi", "es"])

  if st.button("Generate my speech"):
      if text_input:
          tts = gTTS(text_input, lang=language)
          audio_stream = BytesIO()
          st.success("Speech is generated successfully!")
          tts.write_to_fp(audio_stream)
          st.audio(audio_stream)
      else:
          st.warning("Please enter some text or upload from device.")
