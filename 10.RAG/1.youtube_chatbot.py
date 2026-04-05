import streamlit as st
import re
import requests
import time
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# -------------------------------
# SESSION STATE INIT
# -------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "last_video_id" not in st.session_state:
    st.session_state.last_video_id = None


# -------------------------------
# MODELS
# -------------------------------
chat_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------------
# 1. Extract Video ID
# -------------------------------
def extract_video_id(url):
    pattern = r"(?:v=|youtu\.be/)([^&?/]+)"
    match = re.search(pattern, url)
    return match.group(1) if match else None


# -------------------------------
# 2. Fetch Transcript
# -------------------------------
def fetch_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        try:
            transcript = transcript_list.find_transcript(['en'])
            return transcript.fetch(), 'en'
        except:
            pass

        for t in transcript_list:
            try:
                return t.fetch(), t.language_code
            except:
                continue

        return None, None

    except Exception as e:
        print("Error:", e)
        return None, None


# -------------------------------
# 3. Convert to text
# -------------------------------
def transcript_to_text(snippets):
    return " ".join([s.text for s in snippets])


# -------------------------------
# 4. Split text
# -------------------------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)


# -------------------------------
# 5. Translate if needed
# -------------------------------
def translate_chunk(text, source_lang):
    url = "https://api.mymemory.translated.net/get"

    params = {
        "q": text,
        "langpair": f"{source_lang}|en"
    }

    try:
        res = requests.get(url, params=params).json()
        time.sleep(0.3)
        return res["responseData"]["translatedText"]
    except:
        return text


def process_transcript(snippets, source_lang):
    full_text = transcript_to_text(snippets)

    if source_lang and source_lang.startswith("en"):
        return full_text

    chunks = split_text(full_text)

    translated_chunks = []
    for chunk in chunks:
        translated_chunks.append(translate_chunk(chunk, source_lang))

    return " ".join(translated_chunks)


# -------------------------------
# 6. Chunk for RAG
# -------------------------------
def chunk_data(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.create_documents([text])


# -------------------------------
# 7. Vector DB
# -------------------------------
def vector_database(chunks):
    return FAISS.from_documents(chunks, embedding_model)


# -------------------------------
# 8. QA (RAG)
# -------------------------------
def ask_question(vector_store, question):
    docs = vector_store.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = chat_model.invoke(prompt)
    return response.content


# -------------------------------
# MAIN APP
# -------------------------------
def main():
    st.set_page_config(page_title="YouTube ChatBot", layout="wide")

    st.title("🎥 Chat With Your Video")

    video_url = st.text_input("Enter YouTube Video URL")

    video_id = extract_video_id(video_url) if video_url else None

    # 🔥 AUTO RESET when URL changes or removed
    if video_id != st.session_state.last_video_id:
        st.session_state.vector_store = None
        st.session_state.processed = False

    # -------------------------------
    # PROCESS BUTTON
    # -------------------------------
    if st.button("Process"):

        if not video_id:
            st.warning("⚠️ Please enter a valid URL")
            return

        st.success(f"✅ Video ID: {video_id}")

        # Fetch transcript
        with st.spinner("📥 Fetching transcript..."):
            snippets, source_lang = fetch_transcript(video_id)

        if not snippets:
            st.error("❌ No transcript available")
            return

        # Language info
        if source_lang.startswith("en"):
            st.success("✅ English transcript found")
        else:
            st.info(f"🌍 Translating from {source_lang}")

        # Process transcript
        with st.spinner("🔄 Processing..."):
            final_text = process_transcript(snippets, source_lang)

        # Chunk + Vector DB
        chunks = chunk_data(final_text)

        st.session_state.vector_store = vector_database(chunks)
        st.session_state.processed = True
        st.session_state.last_video_id = video_id  # ✅ store current video

        st.success("✅ Ready! Ask questions below 👇")

    # -------------------------------
    # RESET BUTTON
    # -------------------------------
    if st.button("🔄 Reset"):
        st.session_state.vector_store = None
        st.session_state.processed = False
        st.session_state.last_video_id = None
        st.rerun()

    # -------------------------------
    # QUESTION SECTION
    # -------------------------------
    if st.session_state.processed and st.session_state.vector_store:

        st.divider()

        question = st.text_input("Ask a question about the video")

        if question:
            with st.spinner("🤖 Thinking..."):
                answer = ask_question(st.session_state.vector_store, question)

            st.subheader("Answer:")
            st.write(answer)


# -------------------------------
if __name__ == "__main__":
    main()