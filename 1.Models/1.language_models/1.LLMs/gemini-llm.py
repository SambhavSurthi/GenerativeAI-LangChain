from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='D:/GenAI/1.LangChain/.env')

# print(os.getenv("GOOGLE_API_KEY"))

llm=GoogleGenerativeAI(model='gemini-3-flash-preview')

# result=llm.invoke("What is Captial City of India?")
# print(result)

# Creating a Simple chatmodel

while True:
    inp=input('Enter Your Prompt(quit to exit): ')
    if inp=='quit':
        break
    result=llm.invoke(inp)
    print(f'Gemini: {result}\n\n')