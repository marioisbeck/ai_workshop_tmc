from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import os

from langchain_core.documents import Document
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

# Initialize FastAPI app
app = FastAPI()

# Assuming we have the Langserve app compiled as `workflow`
# Recreate the workflow setup here
# local_llm = "tinyllama:chat"
local_llm = "wizardlm2:7b"
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Define global variables
vectorstore = None
retriever = None


@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore, retriever

    # Read the uploaded file
    content = await file.read()

    # Save the uploaded file
    with open(file.filename, "wb") as f:
        f.write(content)

    # Process the document using UnstructuredFileLoader
    loader = UnstructuredFileLoader(file.filename)
    docs_list = loader.load()

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=20
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(local_llm, base_url="http://ollama:11434"),
    )
    retriever = vectorstore.as_retriever()

    return JSONResponse(content={"filename": file.filename})


@app.post("/query/")
async def query_langserve(question: dict):
    global retriever

    inputs = {"question": question["question"]}
    results = []
    docs = retriever.invoke(inputs["question"])

    if not docs:
        generation = "I don't know."
    else:
        doc_txt = docs[1].page_content

        # Define the LLM and prompts for different tasks
        retrieval_grader = (
            PromptTemplate(
                template="""system You are a grader assessing relevance 
            of a retrieved document to a user question. If the document contains keywords related to the user question, 
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
             user
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n assistant""",
                input_variables=["question", "document"],
            )
            | llm
            | JsonOutputParser()
        )

        rag_chain = (
            PromptTemplate(
                template="""system You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise user
            Question: {question} 
            Context: {context} 
            Answer: assistant""",
                input_variables=["question", "context"],
            )
            | llm
            | StrOutputParser()
        )

        generation = rag_chain.invoke(
            {"context": doc_txt, "question": inputs["question"]}
        )

    results.append({"generation": generation})

    return JSONResponse(content={"results": results})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
