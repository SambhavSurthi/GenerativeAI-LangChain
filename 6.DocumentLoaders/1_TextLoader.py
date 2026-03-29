from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
parser=StrOutputParser()
text_loader=TextLoader(file_path='poem.txt',encoding='utf-8')

docs=text_loader.load()

prompt=PromptTemplate(
    template='generate summary of the following poem. \n {poem}',
    input_variables=['poem']
)
poem=docs[0].page_content


chain=prompt | model | parser



response=chain.invoke({'poem':poem})
print(response)