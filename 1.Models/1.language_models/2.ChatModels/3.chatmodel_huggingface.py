from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv(dotenv_path='D:/GenAI/1.LangChain/.env')

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-32B:groq', 
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)
result = model.invoke('who is pm of india?')
print(result.content)