from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

documents=[
    'hi i am sambhav surthi',
    'anil kumar',
    'navyatha',
    'vaibhav'
]

embeddings=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
result=embeddings.embed_documents(documents)
print(result)