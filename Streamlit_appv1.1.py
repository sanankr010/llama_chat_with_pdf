import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings #, SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from Class import webuiLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from PIL import Image

import pandas as pd
import streamlit as st
import speech_recognition as sr

# from gtts import gTTS
import pygame
from io import BytesIO
import pyttsx3



def recognize_speech():
    r = sr.Recognizer()
    mic = sr.Microphone()
    lang_code = "en-US" #input("Enter language code (e.g. en-US, hi-IN): ")
    with mic as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source,phrase_time_limit=18) #will only listen and strore in audio variable
        try:
            text = r.recognize_google(audio, language=lang_code)
            
            st.text("You Said : " +text)
            return text 
#             speak_text(text)
        except sr.UnknownValueError:
            st.text("Could not understand audio, Please try speaking again")  
            return "Could not understand audio, Please try speaking again"


custom_css = """
    <style>
    
    
    
    #custom-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }

    .golden-text {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        color: transparent;
    }

    .green-text {
        color: #00FF00;
    }

    .gray-text {
        color: #808080;
    }
    </style>
    """

# Function to read PDF content
def read_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Mapping of PDFs
pdf_mapping = {
    'HealthInsurance Benefits': 'TaxBenefits_HealthInsurance.pdf',
    'Tax Regime': 'New-vs-Old-Tax.pdf',
    'Reinforcement Learning': 'SuttonBartoIPRLBook2ndEd.pdf',
    'GPT4 All Training': '2023_GPT4All_Technical_Report.pdf',
    # Add more mappings as needed
}

st.markdown(custom_css, unsafe_allow_html=True)
# Load environment variables
load_dotenv()
folder_path=r"C:\Users\siddharth.jha\Generative_AI\demo_turfview_chatbot\pdfs"
nihilent_logo_img = Image.open(r"C:\Users\siddharth.jha\Generative_AI\demo_turfview_chatbot\nihilent_logo.jpg")
castrol_logo_img = Image.open(r"C:\Users\siddharth.jha\Generative_AI\demo_turfview_chatbot\castrol_logo.png")
Turfviw_BI_logo = Image.open(r"C:\Users\siddharth.jha\Generative_AI\demo_turfview_chatbot\BP_logo.png")
# Main Streamlit app
def main():
    st.markdown(custom_css, unsafe_allow_html=True)
    
    st.image(Turfviw_BI_logo, width=20)
    st.title("📝 Ask TurfView BI")
    

    
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            # st.write("Nihilent")
            st.image(nihilent_logo_img)

        with col2:
            # st.write("Castrol")
            st.image(castrol_logo_img,)
           
           
                
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<div id="custom-title"><span class="golden-text">NIHILENT</span>  <span class="green-text">CASTROL</span>  <span class="gray-text">TURFVIEW BI</span></div>', unsafe_allow_html=True)    
        st.markdown('''
            <style>
            custom-css
            {
                 text-align: center;
                 color: #3366FF;
                 font-size: 24px;
             }
             </style>

            <div id="custom-css">
                <h3>Turfview Master Guide Information</h3>
            </div>

            Welcome to our automated system dedicated to providing you with comprehensive insights into the Turfview Master Guide.

            Please don't hesitate to reach out and seek answers to any questions you may have regarding the Turfview Master Guide.
            ''', unsafe_allow_html=True)



        # load Vector from vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    load_vectorstore = FAISS.load_local(r"C:\Users\siddharth.jha\Generative_AI\demo_turfview_chatbot\vector_store\TVBI - TurfView Master Guide V7_en.pdf", embeddings)
    

    st.session_state.processed_data = {
        "vectorstore": load_vectorstore,
    }

    # Load the Langchain chatbot
    llm = webuiLLM()
    qa = ConversationalRetrievalChain.from_llm(llm, load_vectorstore.as_retriever(search_kwargs={"k": 3}), return_source_documents=True)

    # Initialize Streamlit chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if  st.button("▶️ Audio Input", key="play_pause"):
        st.write('Start talking...')
        transcript = recognize_speech()
        st.write('Time over, thankyou...')
        
            # If the transcript is empty, tell the user to try speaking again
        if transcript == "Could not understand audio, Please try speaking again":
            Answer = "Please try speaking again"
            Answer = "Please try speaking again"
            print(Answer)

        # Otherwise, answer the question
        else:
            prompt = transcript
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            print("chat = ",[(message["role"], message["content"]) for message in st.session_state.messages])

            result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
            print("prompt=",prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = result["answer"]
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
            print(result)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        
    
    elif prompt := st.chat_input("Ask your queries about Turfvie Master Guide ?") :
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        print("chat = ",[(message["role"], message["content"]) for message in st.session_state.messages])

        result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
        print("prompt=",prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = result["answer"]
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)
        print(result)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    
    # elif prompt := st.audio(recognize_speech()) :
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.markdown(prompt)
    #     print("chat = ",[(message["role"], message["content"]) for message in st.session_state.messages])

    #     result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
    #     print("prompt=",prompt)

    #     with st.chat_message("assistant"):
    #         message_placeholder = st.empty()
    #         full_response = result["answer"]
    #         message_placeholder.markdown(full_response + "|")
    #     message_placeholder.markdown(full_response)
    #     print(result)
    #     st.session_state.messages.append({"role": "assistant", "content": full_response})    
    
        

if __name__ == "__main__":
    main()