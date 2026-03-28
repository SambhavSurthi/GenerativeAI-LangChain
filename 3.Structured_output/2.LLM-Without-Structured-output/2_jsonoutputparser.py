from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct',
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

prompt=PromptTemplate(
    template='Give me Name,Age,Salary of some fictional Charector who is {person} in the format of \n {parser}',
    input_variables=['person'],
    partial_variables={'parser':parser.get_format_instructions()}
)

# prompt=prompt.invoke({'person':'Indian'})

# response=model.invoke(prompt)
# print(parser.parse(response.content))

chain=prompt | model | parser
print(chain.invoke({'person':'Indian'}))