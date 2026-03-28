from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableBranch
from typing import Literal

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

class output_structure(BaseModel):
    sentiment: Literal['Positive','Negative'] = Field(description='Find the sentiment of the review')

parser=PydanticOutputParser(pydantic_object=output_structure)
str_parser=StrOutputParser()

prompt=PromptTemplate(
    template='Attached the Review please find the sentiment of the review. \n review -> {review} \n output_format -> {output_format}',
    input_variables=['review'],
    partial_variables={'output_format':parser.get_format_instructions()}
)

positive_prompt=PromptTemplate(
    template='for the attached sentiment give a formal tome reply for the user give in the way like you are replying for the user for his feedback in a message para  \n {sentiment}',
    input_variables=['sentiment']    
)

negative_prompt=PromptTemplate(
    template='for the attached sentiment give a formal tome reply for the user give in the way like you are replying for the user for his feedback in a message para \n {sentiment}',
    input_variables=['sentiment']
)


review='''
The phone was amazing
'''

sentiment_analyzer=prompt | model | parser

branch_chain=RunnableBranch(
    # (condition,chain)
    (lambda x:x.sentiment =='Positive', positive_prompt|model|str_parser),
    (lambda x:x.sentiment=='Negative', negative_prompt|model|str_parser),
    RunnableLambda(lambda x:'Could Not Find Sentiment')
)

chain=sentiment_analyzer | branch_chain
response=chain.invoke({'review':review})

print(response)