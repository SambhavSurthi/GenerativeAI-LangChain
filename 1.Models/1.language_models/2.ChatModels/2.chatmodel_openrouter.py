from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv(dotenv_path='D:/GenAI/1.LangChain/.env')

model=ChatOpenRouter(model='openrouter/free',temperature=0.9)

result=model.invoke('tell me About india')
print(result)