from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
parser=StrOutputParser()

prompt_1=PromptTemplate(
    template='tell me a joke on {topic}',
    input_variables=['topic']
)

prompt_2=PromptTemplate(
    template='explain me the following joke -> {joke}',
    input_variables=['joke']
)

joke_chain=prompt_1 | model | parser

parallel_chain=RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'explaination': prompt_2 | model | parser,
        'size':RunnableLambda(lambda x:len(x.split()))
    }
)

chain=joke_chain | parallel_chain

response=chain.invoke({'topic':'GenAI'})

print(response)