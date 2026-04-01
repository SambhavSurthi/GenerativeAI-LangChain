from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import time

load_dotenv()
loader=TextLoader('genai.txt',encoding='utf-8')
embeddings_model=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
splitter=RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=50
)

docs=loader.load()
docs=splitter.split_documents(
    documents=docs,
)

vector_database=Chroma(
    collection_name='test_collection',
    embedding_function=embeddings_model,
    persist_directory='vector_db'
)

# res=vector_database.add_documents(documents=docs)

response=vector_database.similarity_search(query='what is generative AI')

for r in response:
    print(r.page_content)