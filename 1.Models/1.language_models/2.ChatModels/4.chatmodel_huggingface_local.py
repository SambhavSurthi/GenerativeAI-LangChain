from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm=HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(temperature=0.5)
)

model=ChatHuggingFace(llm=llm)
response=model.invoke('who is pm of india?')
print(response)
print()
print(response.content)