from langchain.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


chat_history=[
    SystemMessage(content='You are a helpful AI assistant')
]

model=ChatGoogleGenerativeAI(model='gemini-3-flash-preview')


while True:
    prompt = input("Say something: ")
    if prompt=='exit':
        break
    chat_history.append(HumanMessage(content=prompt))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content[0]['text']))
    print(response.content[0]['text'])
print(chat_history)
