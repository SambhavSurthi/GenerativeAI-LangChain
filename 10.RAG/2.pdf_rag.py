from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="""
You are an assistant for question-answering tasks.
Use the context to answer the question.
If you don't know, say you don't know.
Max 3 sentences.

Question: {question}
Context: {context}
Answer:
""",
    input_variables=["question", "context"]
)

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

parser = StrOutputParser()


def main():
    pdf_path = input("Enter Path of Your PDF: ")

    if not pdf_path:
        print("PDF not available")
        return

    try:
        loader = PyMuPDFLoader(file_path=pdf_path)
        docs = loader.load()

        split_docs = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(
            documents=split_docs,
            embedding=embedding_model
        )

        print("✅ PDF loaded successfully")

        while True:
            user_input = input(">> ")

            if user_input.lower() == "exit":
                print("Exiting Chatbot")
                break

            results = vector_store.similarity_search(user_input, k=3)

            context = "\n\n".join([doc.page_content for doc in results])

            chain = prompt | chat_model | parser

            response = chain.invoke({
                "question": user_input,
                "context": context
            })

            print("\n💬", response, "\n")

    except Exception as e:
        print("❌ Error:", str(e))


if __name__ == "__main__":
    main()