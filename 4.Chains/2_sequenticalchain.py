from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()
parser=StrOutputParser()
model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

prompt1=PromptTemplate(
    template='give me indepth report about the following topic -> {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate 5 lines summary for the following decription -> {description}',
    input_variables=['description']
)

chain= prompt1 | model | parser | prompt2 | model | parser

print(chain.invoke({"topic":"Langchain"}))

chain.get_graph().print_ascii()