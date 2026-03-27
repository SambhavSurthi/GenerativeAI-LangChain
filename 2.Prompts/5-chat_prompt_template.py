from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

chat_template=ChatPromptTemplate(
    [('system','You are a {domain} expert'),
    ('human','tell me about {something}')]
)

prompt=chat_template.invoke({'domain':'designing','something':'ux designing in 10 words'})

print(prompt)