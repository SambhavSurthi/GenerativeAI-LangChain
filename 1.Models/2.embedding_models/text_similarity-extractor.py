from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

model=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

input_str='tell me about god of cricket'

embedded_document=model.embed_documents(documents)
embedded_text=model.embed_query(input_str)

similarity=cosine_similarity([embedded_text],embedded_document)

index=similarity.argmax()

print(input_str)
print(documents[index])
print("similarity score is:", similarity[0][index])
print('All Sentence Similarit: ',similarity[0])
