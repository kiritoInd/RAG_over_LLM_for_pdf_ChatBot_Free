from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'Upload/'

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings

import os

from dotenv import load_dotenv

chat_history = []   

load_dotenv()

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1" #llm id

# llm to use Note - You Have to take the persmition from hugging face key
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.8, "top_k": 50},
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
)

def load_db(file, chain_type, k):
    # Load documents

    loader = PyPDFLoader(file)
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # Define embedding
    embeddings = HuggingFaceEmbeddings()

    # Create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    # Define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Create a conversational chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

# Initialize your conversational chain
loaded_file = "MachineLearning-Lecture01.pdf"
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
