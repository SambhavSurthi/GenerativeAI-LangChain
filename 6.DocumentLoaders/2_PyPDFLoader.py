from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

loader=PyPDFLoader(file_path='resume.pdf')
docs=loader.load()
print(docs)
print()
print()
print()
print(docs[0])
