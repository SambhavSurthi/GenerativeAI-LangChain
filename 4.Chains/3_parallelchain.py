from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
parser=StrOutputParser()

gemini_model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct',
    task='text-generation'
)

huggingface_model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='give me a brief description about the following topic -> {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='generate 5 question and answer about the following topic -> {topic}',
    input_variables=['topic']
)

prompt3=PromptTemplate(
    template='combine topic -> {topic} and description -> {description} into a single document.',
    input_variables=['topic','description']
)

parallelChain=RunnableParallel(
    {
        'topic': prompt1 | gemini_model | parser,
        'description': prompt2 | huggingface_model | parser
    }
)

mergeChain= prompt3 | gemini_model | parser

chain= parallelChain | mergeChain

response=chain.invoke({"topic":"LangChain"})
print(response)


chain.get_graph().print_ascii()