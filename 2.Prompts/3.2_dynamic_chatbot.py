from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('prompt.json')

model=ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

# if st.button('Generate'):
#     prompt=template.invoke({
#         'paper_input':paper_input,
#         'style_input':style_input,
#         'length_input':length_input
#     })
#     response=model.invoke(prompt)
#     st.write(response.content[0]['text'])

if st.button('Genrate'):
    chain=template | model
    response=chain.invoke({
        'paper_input':paper_input,
         'style_input':style_input,
         'length_input':length_input
    })
    st.write(response.content[0]['text'])