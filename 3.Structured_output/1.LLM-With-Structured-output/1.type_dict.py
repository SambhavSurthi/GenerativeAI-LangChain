from typing import TypedDict,Optional,Literal,Annotated
from langchain_google_genai import ChatGoogleGenerativeAI


from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

class output_structure(TypedDict):
    summary: Annotated[str,'Provide a Detailed Summary About The user query']
    sentiment:Optional[Annotated[Literal['positive','negative'], 'provide the sentiment of the query postive or negative']]
    
structure_model=model.with_structured_output(output_structure)
prompt="I had the most incredible dining experience at this restaurant! The food was absolutely delicious - every dish was perfectly prepared and beautifully presented. Our server was attentive, knowledgeable, and made excellent recommendations. The atmosphere was elegant yet comfortable. I can't wait to come back and try more items from their menu. Definitely a new favorite spot!"
response=structure_model.invoke(prompt)


print(response)
    