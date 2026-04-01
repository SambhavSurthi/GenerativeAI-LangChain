from langchain_community.retrievers import WikipediaRetriever

retriever=WikipediaRetriever(top_k_results=5)

response=retriever.invoke(input='what is genai')

print(response)