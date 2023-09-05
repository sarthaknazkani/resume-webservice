from flask import Flask, request, jsonify
import base64
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import PyPDF2
import io
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

app = Flask(__name__)

def get_pdf_text(pdf_docs):
    
    # for pdf in pdf_docs:
    #     pdf_reader = PdfReader(pdf)
    #     for page in pdf_reader.pages:
    #         text += page.extract_text()

    try:
        num_pages = len(pdf_docs.pages)
        # print('line 299____>',num_pages)
        # pdf_reader = PdfReader(pdf_docs)
        # print('pdf reader-->',pdf_docs)
        text = ""
        for page in pdf_docs.pages:
            # print('line 34')
            text += page.extract_text()
            # print('text--->',text)

        return text
    
    except Exception as e:
        return jsonify({'error': str(e)})
    

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # print('line 55')
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    # st.write('embeddings',embeddings)
    # st.write('client-->',embeddings.client[0].max_seq_length)
    # embeddings.client[0].max_seq_length = 5000
    # print('line 61')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # print('line 63-->', vectorstore)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":1024})
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl")
    # st.write('llm-->',llm)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    # print('line 78-->', conversation_chain)
    return conversation_chain





    
def base64_to_pdf(base64_data):
    try:
        # Decode the base64-encoded PDF data
        pdf_bytes = base64.b64decode(base64_data)
        # print('length--->',len(pdf_bytes))

        # Create a PDF file-like object
        pdf_file = io.BytesIO(pdf_bytes)
        # print('pdfFile-->',pdf_file)
        # pdf_file.write(pdf_bytes)
        # print('line 68')
        # Create a PyPDF2 PdfFileReader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        # print('line 70')
        # print('odf_reader-->',pdf_reader)

        # You can now work with the PDF, for example:
        # pdf_text = ""
        # for page_num in range(pdf_reader.getNumPages()):
        #     pdf_text += pdf_reader.getPage(page_num).extractText()
        
        # Return the PyPDF2 PdfFileReader object (or any other data as needed)
        return pdf_reader

    except Exception as e:
        print(f'Error: {e}')
        return None

@app.route('/pdf-to-base', methods=['POST'])
def pdf_to_base64():
    try:
        # Get the PDF data from the request's data attribute
        pdf_file = request.data

        if not pdf_file:
            return jsonify({'error': 'No PDF data provided'})

        # Convert base64 data to a PDF
        pdf_reader = base64_to_pdf(pdf_file)

        if pdf_reader:
            # Pass the PDF reader object to another function or perform additional processing here
            # For demonstration purposes, we're just returning the number of pages
            # print('Im in')
            num_pages = len(pdf_reader.pages)
            # print("num_pages-->",num_pages)

            # get pdf text
            
            raw_text = get_pdf_text(pdf_reader)
            # print('raw_text--?',raw_text)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            # print('text_chunks-->',text_chunks)

            # create vector store
            vectorstore = get_vectorstore(text_chunks)

            # create conversation chain
            conversation = get_conversation_chain(
                vectorstore)
            # print('st-->',conversation)
            response = conversation({'question': 'Tell me the list of skillset'})
            # print('response-->',response)
            chat_history = response['chat_history']
        
            # response = st.session_state.conversation({'question': 'Tell me the list of skillset'})
            # st.session_state.chat_history = response['chat_history']

            # print('st-->',st)

            if(chat_history!=None):
                print('line 160-->')
                for i, message in enumerate(chat_history):
                    # print('hello')
                    # st.write('i-->',i)
                    # st.write('message-->',message)
                    if i % 2 == 0:
                        # st.write('message.content-->',message.content)
                        print('test')
                        # st.write(user_template.replace(
                        #     "{{MSG}}", message.content), unsafe_allow_html=True)
                    
                    else:
                        # print(message.content)
                        return jsonify({'skillset': message.content})
                        # return response(status=400, response =message.content)
                        

            
            # return jsonify({'num_pages': num_pages})
        else:
            return jsonify({'error': 'Failed to convert base64 data to PDF'})

        # Decode the base64-encoded PDF data
        # pdf_bytes = base64.b64decode(pdf_file)

        # print('pdfbyters-->',pdf_bytes)

        # file_result = open('sample_decoded.pdf', 'wb') 
        # file_result.write(pdf_bytes)
        
        # Call the pdf reader function 
        # get_pdf_text(file_result)
        # print('fileResult-->',file_result)


        # Convert the PDF bytes to a string (you can use other libraries for more advanced PDF processing)
        # pdf_string = pdf_bytes.decode('utf-8')

        # Return the PDF content as a string in the response
        # return jsonify({'pdf_content': pdf_string})

    except Exception as e:
        return jsonify({'error': str(e)})
    
def main():
    # os.environ["OPENAI_API_KEY"] = "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"
    # load_dotenv()
    os.environ.get("OPENAI_API_KEY")
    # if "conversation" not in st.session_state:
    #     st.session_state.conversation = None
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = None
    

if __name__ == '__main__':
    
    main()
    app.run(debug=True)
