from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda,RunnableBranch
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenRouter(model='openrouter/free')
parser=StrOutputParser()

prompt_1=PromptTemplate(
    template='tell me about {topic}',
    input_variables=['topic']
)

prompt_2=PromptTemplate(
    template='reduce the size and make a small summary for the following description -> {description}',
    input_variables=['description']
)

topic_model=prompt_1 | model | parser

summarizer=RunnableBranch(
    (lambda x:len(x.split())<50, RunnablePassthrough()),
    (lambda x:len(x.split())>50, prompt_2 | model | parser),
    RunnableLambda(lambda x:x)
)

chain=topic_model | summarizer
print(response)

# response=chain.invoke({'topic':'genai'})

# response= chain.batch([
#     {'topic':'AI'},
#     {'topic':'ML'}
# ])
# print(response)
# print()
# print()
# for chunk in chain.stream({'topic':'GenAI'}):
#     print(chunk)

# response=chain.map().invoke([{"topic": "AI"}, {"topic": "ML"}])
# print(response)