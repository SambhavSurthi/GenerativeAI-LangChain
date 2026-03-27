from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# prompt=PromptTemplate.from_template(
#     template='Greet this person {name} in 5 languages',
# )
# prompt=prompt.format(name='sambhav')
# print(prompt)

prompt=PromptTemplate(
    template='Greet this person {name} in 5 languages',
    input_variables=['name']
)

prompt=prompt.invoke({'name':'Sambhav'})

model=ChatGoogleGenerativeAI(model='gemini-3-flash-preview')
response=model.invoke(prompt)
print(response.content[0]['text'])