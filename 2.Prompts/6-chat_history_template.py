from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')


prompt_template=ChatPromptTemplate([
    ('system','Your are an helpful assistant who solves user queries.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','solve this query {query}')
])
chat_history=[]
with open('chat_history.txt') as file:
    chat_history.extend(file.readlines())
    
# print(f'Chat history: {chat_history}')

# prompt=prompt_template.invoke({'chat_history':chat_history,'query':'what is status of my order'})

# print(f'Final prompt: \n {prompt}')

# print()

# print()

while True:
    query=input('Enter Your query: ')
    if query=='exit':
        break
    prompt=prompt_template.invoke({'chat_history':chat_history,'query':query})
    response=model.invoke(prompt)
    print('response: \n\n')
    print(response.content)
    

