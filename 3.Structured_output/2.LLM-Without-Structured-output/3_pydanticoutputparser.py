from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct',
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)

class output_schema(BaseModel):
    name:str=Field(description='Name of the person')
    age:int=Field(ge=18,description='age of the person')
    salary:float=Field(description='Expected Salary of the person')

parser=PydanticOutputParser(pydantic_object=output_schema)

template=PromptTemplate(
    template='Give me Name,Age,Salary of some fictional Charector who is {person} in the format of \n {parser}',
    input_variables=['person'],
    partial_variables={'parser':parser.get_format_instructions()}
)
print(template.invoke({'person':'Indian'}))
print()
chain=template | model | parser
print(chain.invoke({'person':'Indian'}))