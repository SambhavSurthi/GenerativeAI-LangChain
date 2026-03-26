from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv(dotenv_path='D:/GenAI/1.LangChain/.env')

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview',temperature=1.5,max_output_tokens=20)
result=model.invoke('write 5 line poem on genai')
print(result)
print(f'Result: {result.content[0]['text']}')
'''
output:

content=[{'type': 'text', 'text': 'The capital of India is **New Delhi**.', 'extras': {'signature': 'EjQKMgG+Pvb7aq+Z0FxqWCcz7pIsyxjEfk0/8DX22RI6u6T+0xvrkbAvz9zDCX1P1P89Gq/F'}}] additional_kwargs={} response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-3.1-flash-lite-preview', 'safety_ratings': [], 'model_provider': 'google_genai'} id='lc_run--019d25cb-83f4-71b2-9c93-0398bf9a4d46-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 9, 'output_tokens': 9, 'total_tokens': 18, 'input_token_details': {'cache_read': 0}}

'''
