from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
parser=StrOutputParser()

prompt=PromptTemplate(
    template='Give me brief description about the following topic {topic}.',
    input_variables=['topic']
)

chain= prompt | model | parser

response=chain.invoke({'topic':'AI/ML'})

print(response)
print()
chain.get_graph().print_ascii()
