from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
import tqdm


import streamlit as st
from transformers import pipeline

# Use a pipeline as a high-level helper
from transformers import pipeline

st.set_page_config(page_title="AI Tools by Abhinav Gupta", layout="centered")

st.title("📝 AI Tools by Abhinav Gupta")

tab_titles = ["Text Summarizer","Semantic Search"]

tab1, tab2 = st.tabs(tab_titles)

with tab1:

	text = st.text_input("Paste text below to Summarize")

	if st.button("Submit") :

		summarizer = pipeline("summarization", model="pszemraj/long-t5-tglobal-base-16384-book-summary")
		sm_txt = summarizer(text)
		st.markdown(sm_txt)

########################################################

with tab2:

  uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

  if uploaded_file is not None:
    # Step 2: Save to a temporary file to provide a path for the loader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Step 3: Load and split the document
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=500, #Number of tokens to keep in a chunk
  chunk_overlap=16 #Number of overlap tokens
  )
  
  data_chunks = uploaded_file.load_and_split(text_splitter)

  embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

  chromadb_client = chromadb.PersistentClient(path='./chroma_db')

 # Instantiate a Chroma vector store to store and retrieve document embeddings
  vector_store = Chroma(
    embedding_function=embedding_model,
    client=chromadb_client,
    collection_name='maxbupa_collection',
  )


  for i in tqdm.tqdm(range(len(data_chunks))):
      vector_store.add_documents(documents=[data_chunks[i]])

  vector_store._collection.count()
  user_query = "PREFACE"

#To retrieve relevant chunks for the user_query from the vectore store , we need to first create retriever interface to vectore store
  retriever = vector_store.as_retriever(
  search_type='similarity', #Metric to use to find relevant documents
  search_kwargs={'k': 5} #Number of chunks to return
  )

  relevant_chunks = retriever.invoke(user_query)
  relevant_chunks[1].page_content