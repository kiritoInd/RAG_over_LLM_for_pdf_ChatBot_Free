from flask import Flask, request, jsonify, render_template

from werkzeug.utils import secure_filename

import os
import openai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI


app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'Upload/'

chat_history = []

# Load environment variables
load_dotenv()

# OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
llm_name = "gpt-3.5-turbo"

def load_db(file, chain_type, k):

    # Load documents

    loader = PyPDFLoader(file)
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # Define embedding
    embeddings = OpenAIEmbeddings()

    # Create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    # Define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Define LLM

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )


    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm= llm,
        chain_type=chain_type,
        memory = memory,
        retriever=retriever,
        return_source_documents=False,
        return_generated_question=False,
    )

    return qa

# Initialize your conversational chain
loaded_file = "corpus.pdf"

qa = load_db(loaded_file, "stuff", 4)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])

def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        global loaded_file
        loaded_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        global qa
        qa = load_db(loaded_file, "stuff", 4)
        return 'File uploaded successfully'

@app.route('/query', methods=['POST'])

def query():
    query = request.json["query"]
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    helpful_answer = result['answer'].split('Helpful Answer: ')[-1]
    question = result['question']
    response = {
        'helpful_answer': helpful_answer,
        'question' : question,
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
