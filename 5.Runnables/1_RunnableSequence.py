from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt=PromptTemplate(
    template='tell me about {topic} in 5 lines',
    input_variables=['topic']
)

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

parser=StrOutputParser()

# chain=prompt|model|parser
chain=RunnableSequence(prompt,model,parser)
response=chain.invoke({'topic':'GenAI'})
print(response)