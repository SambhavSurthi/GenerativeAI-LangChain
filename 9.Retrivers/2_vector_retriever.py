from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#loading the data

loader=TextLoader(file_path='genai.txt',encoding='utf-8')
text_data=loader.load()

# splitting the data into document

splitter=RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80
)

docs=splitter.split_documents(documents=text_data)


#define embeddings model

embed_model=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

# adding documents into FAISS database

vector_store=FAISS.from_documents(
    documents=docs,
    embedding=embed_model
)

# defining retriever

retriever=vector_store.as_retriever(
    kwargs={
        "k":2,
        'search_type':'similarity'
    }
)
response=retriever.invoke(input='what is genai')

print(response)

for r in response:
    print('-----------result-------------------')
    print(r.page_content)