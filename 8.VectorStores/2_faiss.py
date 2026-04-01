from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
loader=WikipediaLoader('Generative AI')
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)
embeddings_model=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
res=loader.load()
docs=splitter.split_documents(res)

vector_store=FAISS.from_documents(
    documents=docs,
    embedding=embeddings_model
)
response=vector_store.similarity_search('what is genai')
for r in response:
    print(r.page_content)