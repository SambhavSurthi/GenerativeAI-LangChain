from langchain_community.document_loaders import WebBaseLoader

import os
os.environ["USER_AGENT"] = "Mozilla/5.0"

url='https://en.wikipedia.org/wiki/Attention_Is_All_You_Need'

loader=WebBaseLoader(web_path=url)
docs=loader.load()
print(docs)
print()
print()
print()
print(docs[0].page_content)