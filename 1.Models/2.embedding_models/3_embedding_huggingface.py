from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

text='hi i am sambhav surthi'

result=embeddings.embed_query(text=text)
print(result)