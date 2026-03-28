from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

'''
1) get details of one topic from LLM
2) the give that topic to the LLM to generate 5 points summary

'''

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct',
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)
parser=StrOutputParser()

template1=PromptTemplate(
    template="Tell me Indetail about the {topic}.",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="give me a 5 point summary of the following description. {description}",
    input_variables=['description']
)
# prompt1=template1.invoke()

# response=parser.parse(model.invoke(prompt1))

# prompt2=template2.invoke({'description':response})
# response=parser.parse(model.invoke(prompt2))
# print(response)

chain=template1 | model | parser | template2 | model | parser
response=chain.invoke({'topic':'blackhole'})
print(response)
