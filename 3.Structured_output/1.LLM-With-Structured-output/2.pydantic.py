from pydantic import BaseModel, Field
from typing import Literal,Annotated,Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

class output_structure(BaseModel):
    summary:str=Field(description='generate Summary for the user query')
    sentiment:Literal['Positive','Negative']=Field(description='provide sentiment for the user query')
    user_name:Optional[str]=Field(description='provide name of the user')
    product_name:str=Field(description='Provide name of the product',default='NA')
    
model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
structed_model=model.with_structured_output(output_structure)
prompt="I had the most incredible dining experience at this restaurant! The food was absolutely delicious - every dish was perfectly prepared and beautifully presented. Our server was attentive, knowledgeable, and made excellent recommendations. The atmosphere was elegant yet comfortable. I can't wait to come back and try more items from their menu. Definitely a new favorite spot! Review by SambhavSurthi"

response=structed_model.invoke(prompt)

print(response)