import os  # Operating system interfaces
import shutil  # High-level file operations
import uuid

from typing import List  # Type hinting for lists
from fastapi import FastAPI, UploadFile, File, HTTPException  # FastAPI framework and HTTP handling
from fastapi.responses import JSONResponse  # FastAPI response handling
from pydantic import BaseModel  # Data validation and settings management using Python type annotations

from langchain_community.chat_models import ChatOllama  # Chat model from Langchain Community
from langchain_community.document_loaders import PyPDFLoader  # Reading in PDFs
from langchain_community.embeddings.ollama import OllamaEmbeddings  # Ollama embeddings for text representation
from langchain_community.vectorstores import Chroma  # Chroma vector store for efficient retrieval
from langchain.prompts import ChatPromptTemplate  # Class for prompts
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Tool to split text recursively by characters
from langchain_core.runnables import RunnablePassthrough  # Pass-through runnable for data processing
from langchain.prompts import ChatPromptTemplate # class for promts
from langchain_core.output_parsers import StrOutputParser  # Parses output into strings

# Define a Pydantic model for the request body
class QuestionRequest(BaseModel):
    question: str
    
# Initialize FastAPI app
app = FastAPI()

# Assuming we have the Langserve app compiled as `workflow`
# Recreate the workflow setup here
OLLAMA_LARGE_LANGUAGE_MODEL='wizardlm2:7b'
OLLAMA_SERVER='http://ollama:11434'
OLLAMA_EMBEDDING_MODEL='mxbai-embed-large'

llm = ChatOllama(model=OLLAMA_LARGE_LANGUAGE_MODEL, base_url=OLLAMA_SERVER, format="json")

# Define global variables
vectorstore = None
retriever = None


@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore, retriever

    # Read the uploaded file
    content = await file.read()

    # Remove all folders in the ./www directory
    www_path = "./www"
    if os.path.exists(www_path):
        shutil.rmtree(www_path)
    os.makedirs(www_path, exist_ok=True)

    # Save the uploaded file
    save_path = f"./www/%s" % str(uuid.uuid4())
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(content)


    # Process the document using PyPDFLoader
    loader = PyPDFLoader(save_path)
    docs = loader.load()

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, 
        chunk_overlap=20
    )

    # Make splits
    splits = text_splitter.split_documents(docs)

    # Add to vectorDB
    try:
        Chroma.delete_collection(vectorstore)
        vectorstore = Chroma.from_documents(
            collection_name=str(uuid.uuid4()),
            documents=splits,
            embedding=OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_SERVER)
        )
    except:
        vectorstore = Chroma.from_documents(
            collection_name=str(uuid.uuid4()),
            documents=splits,
            embedding=OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_SERVER)
        )
    retriever = vectorstore.as_retriever()

    return
    # return JSONResponse(content={"filename": file.filename})



@app.post("/query/")
async def query_langserve(question_request: QuestionRequest):
    
    global retriever

    template = "You are an assistant specialized in question-answering tasks. Use the given context to answer the question concisely. If the answer is not present in the context, clearly state that you don't know the answer and do not provide any further answer.\nQuestion: {question} \nContext: {context} \nAnswer:"
    prompt = ChatPromptTemplate.from_template(template)

    if retriever is None:
        raise HTTPException(status_code=400, detail="No documents have been uploaded yet.")

    # This is handled by RunnablePassthrough() function which can add a 
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question_request.question)
    
    return JSONResponse(content={"Answer": answer})


@app.post("/query_directly/")
async def query_directly(question_request: QuestionRequest):
    answer = llm.invoke(question_request.question)
    return JSONResponse(content={"Answer": answer.content})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)