from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableSequence

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

prompt_tweet=PromptTemplate(
    template='generate a simple tweet for the following topic -> {topic}',
    input_variables=['topic']
)

prompt_linkedin=PromptTemplate(
    template='generate a simple linkedin post for the following topic -> {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

chain=RunnableParallel(
    {
        "tweet": prompt_tweet | model | parser,
        "linkedin": prompt_linkedin | model | parser
    }
)

response=chain.invoke({'topic':"GenAI"})
print(response)
print()
print()
print(response['tweet'])
print()
print()
print(response['linkedin'])