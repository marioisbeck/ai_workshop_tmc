#!/usr/bin/env python
# coding: utf-8

# # Part II</br>Building the AI Assistant

# ## Introduction

# * [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) Vaswani et al. (2017, Google Brain/ Research)
# * 5 days to 1 million users (OpenAI)
# * 1.8 billion monthly visits in March 2023 (OpenAI)

# - **Agricultural Revolution**: Around 10,000 BCE, shift to settled farming.
# - **Industrial Revolution**: Late 18th century, rise of industrialization.
# - **Digital (Computer) Revolution**: Mid-20th century, advent of computers.
# - **AI Revolution**: Early 21st century, integration of artificial intelligence.

# ### Agenda

# * Introduction to LLMs
# * Starting Docker Containers
# * Build AI Assistant (Walkthrough & HandsOn)
#     * Ingestion (Load, Split, Embed, Store)
#     * Similarity Search
#     * Combine Context
#     * Response Generation
# * Langserve and Streamlit App

# * We will build a simple retrieval augmented generation (RAG) pipeline and complete HandsOn tasks.
# * The notebook is based on [langchains rag intro](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_1_to_4.ipynb).
# * We build towards a broader understanding of the RAG langscape langchain's [rag from scratch](https://github.com/langchain-ai/rag-from-scratch/tree/main).

# ### Learning Objectives

# 1. Complete Installation and Understand Functioning of Essential Tools
# 2. **Understand the Basics of Large Language Models (LLMs)**
# 3. **Understand on a Programmatic Level how AI Assistants are Built**

# ## Starting Docker Containers

# ```bash
# docker-compose -up -d --build
# ```

# ## Introduction to LLMs

# Slides are shamelessly taken from 3Blue1Brown's [But what is a GPT? Visual intro to transformers | Chapter 5, Deep Learning](https://www.youtube.com/watch?v=wjZofJX0v4M&ab_channel=3Blue1Brown)

# ![/overview.jpeg](assets/imgs/overview.jpeg)

# ![/tokens.jpeg](assets/imgs/tokens.jpeg)

# ![/giving_meaning.jpeg](assets/imgs/giving_meaning.jpeg)

# ## Build AI Assistant

# ![assets/imgs/simple_rag.png](assets/imgs/simple_rag.png)

# - **Ingestion**: Load and preprocess documents for further processing.
#     - **Load**: Upload documents to the backend.
#     - **Split**: Split documents into manageable chunks using characters, sections, semantic meaning, and delimiters.
#     - **Embed**: Convert document chunks (and query) into vector embeddings for representation.
#     - **Store**: Store the embeddings in a vector database (Vectorstore) for efficient retrieval.
# - **Similarity Search**: Use the query embedding to search and retrieve the most relevant document chunks from the Vectorstore.
# - **Combine Context**: Combine retrieved document chunks with the query to provide context for the generation model.
# - **Response Generation**: Use a language model to generate a response based on the query and retrieved context.

# ### Setup

# #### Keep it Clean

# The following is only to suppress output which we do not care about in this workshop.

# In[9]:


import warnings
import logging
import os
import numpy as np

# Setting USER_AGENT variable for jupyter notebook
os.environ["USER_AGENT"] = "jovyan"

# Disable warnings
warnings.filterwarnings("ignore")

# Disable info messages
logging.getLogger().setLevel(logging.WARNING)


# #### #REMOVE Package Installation

# In[258]:


# ! pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain pypdf


# ### Load Libraries

# In[348]:


import bs4  # Library for web scraping and parsing HTML/XML
from langchain import hub  # Access langchain hub for pre-built tools and models
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # Tool to split text recursively by characters
from langchain_core.output_parsers import StrOutputParser  # Parses output into strings
from langchain_core.runnables import (
    RunnablePassthrough,
)  # Pass-through runnable for data processing
from langchain_community.chat_models import (
    ChatOllama,
)  # Chat model from Langchain Community
from langchain_community.embeddings.ollama import (
    OllamaEmbeddings,
)  # Ollama embeddings for text representation
from langchain_community.document_loaders import (
    WebBaseLoader,
)  # Load documents from the web
from langchain_community.vectorstores import (
    Chroma,
)  # Chroma vector store for efficient retrieval
from langchain_community.document_loaders import PyPDFLoader  # reading in pdfs
from langchain.prompts import ChatPromptTemplate  # class for promts


# ### Settings

# In[12]:


OLLAMA_LARGE_LANGUAGE_MODEL = (
    "wizardlm2:7b"  # Specifies the large language model version
)
OLLAMA_SERVER = "http://ollama:11434"  # URL for the Ollama server


# In[282]:

question = "What is the TMC Entrepreneurial Lab?"


# ## Ingestion

# ### Load

# [Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)

# #### PDF Loader

# In[335]:


# pdf document loader
loader = PyPDFLoader("./backend/tmc_tel_lab.pdf")
docs = loader.load()


# In[336]:


len(docs)


# In[337]:


docs[0].page_content[:300]


# #### HandsOn --- Web Loader

# > How can you use the WebBaseLoader to load the contents of the following website: "https://www.themembercompany.com/nl/employeneurship"?
#
# > How long is the page_content of the resulting document?

# In[318]:


# HandsOn: - Web Loader
loader = WebBaseLoader(
    web_paths=("https://www.themembercompany.com/nl/employeneurship",)
)
docs = loader.load()


# In[292]:


# show character length page content
len(docs[0].page_content)


# ### Split

# [Splitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/)

# ![assets/imgs/splitting_documents.png](assets/imgs/splitting_documents.png)

# In[383]:


# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=20
)

# Make splits
splits = text_splitter.split_documents(docs)


# This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

# In[384]:


len(splits)


# In[386]:


splits[0].page_content[:500]


# In[387]:


docs[0].page_content


# In[388]:


splits[0].page_content


# In[389]:


splits[1].page_content


# ### Embed

# [Text embedding models](https://python.langchain.com/docs/integrations/text_embedding/openai)

# #### Text Embedding

# In[306]:


OLLAMA_EMBEDDING_MODEL = "all-minilm"


# In[307]:


embedding = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_SERVER)
query_result = embedding.embed_query(question)
split_result = embedding.embed_query(splits[0].page_content)


# In[308]:


len(query_result)


# In[309]:


len(split_result)


# In[310]:


split_result[:4]


# #### Cosine Similarity

# [Cosine similarity](https://platform.openai.com/docs/guides/embeddings/frequently-asked-questions) is reccomended (1 indicates identical).

# In[312]:


def cosine_similarity(vec1, vec2, print_output=False):

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    similarity = dot_product / (norm_vec1 * norm_vec2)

    if print_output:
        print("Cosine Similarity:", similarity)

    return similarity


# In[313]:


similarity = cosine_similarity(query_result, split_result, True)


# #### HandsOn --- Better Embedding

# > Write code to use the more sophisticated `mxbai-embed-large` instead of the `all-miniml` embedding model with the local Ollama instance. This enables better performance and more accurate results.

# In[314]:


OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large"


# In[315]:


embedding = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_SERVER)
query_result = embedding.embed_query(question)
split_result = embedding.embed_query(splits[0].page_content)


# In[317]:


similarity = cosine_similarity(query_result, split_result, True)


# #### HandsOn --- Stroopwafel

# > Similar to the `Japan - Germany` example from the `Introduction to LLMs` we will now calculate the distance between Netherlands and Germany in the vector space. This we can then use to understand what item in Germany corresponds to what the stroopwafel is in the Netherlands.

# In[338]:


OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large"
embedding = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_SERVER)


# In[339]:


# Assuming embedding is some pre-trained embedding model with an embed_query method
words = ["Bratwurst", "Mercedes", "Schwarzwälder Kirschtorte", "Berliner", "Lebkuchen"]


# In[340]:


# getting vectors of tokens/ words
stroopwafel_embedding = np.array(embedding.embed_query("Stroopwafel"))
netherlands_embedding = np.array(embedding.embed_query("Netherlands"))
germany_embedding = np.array(embedding.embed_query("Germany"))


# In[341]:


# calculating the comparison vector
comparison_embedding = stroopwafel_embedding - (
    netherlands_embedding - germany_embedding
)


# In[342]:


# initiating variables
highest_similarity = -1
closest_word = None


# In[343]:


# running the loop
for word in words:

    # embedding the query word
    word_embedding = np.array(embedding.embed_query(word))

    # generating output
    print(word)
    similarity = cosine_similarity(adjusted_stroopwafel_embedding, word_embedding, True)
    print("")

    # capturing highest similarity
    if similarity > highest_similarity:
        highest_similarity = similarity
        closest_word = word


# In[344]:


# final evaluation
print(
    f"The word closest to 'stroopwafel' is '{closest_word}' with a cosine similarity of {highest_similarity}."
)


# ### Store

# [Vectorstores](https://python.langchain.com/docs/integrations/vectorstores/)

# ![assets/imgs/langchain_vectorstores_rag.png](assets/imgs/langchain_vectorstores_rag.png)

# In[345]:


vectorstore = Chroma.from_documents(
    collection_name=OLLAMA_EMBEDDING_MODEL,
    documents=splits,
    embedding=OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_SERVER),
)


# In[347]:


# missing write code to look at what is in the chroma db


# ### Similarity Search

# In[206]:


result[0].page_content


# In[207]:


result[0].metadata


# In[208]:


for split in splits:
    print(split.page_content)
    print("")


# ### Combine Context

# In[351]:


# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
prompt


# In[215]:


# LLM
llm = ChatOllama(model=OLLAMA_LARGE_LANGUAGE_MODEL, base_url=OLLAMA_SERVER)


# In[216]:


# Chain
chain = prompt | llm


# ### Response Generation

# In[352]:


# Invoking the RAG chain
response = chain.invoke({"context": result, "question": question})
response.content


# In the `chain.invoke()` example above we used directly the result output of a similarity search of the vector database. Langchain has a better approach for this via retrievers.

# #### Retriever

# In[353]:


# here we create a retriever from the vectorstore which can perform similarity search and returns one document
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})


# In[ ]:


vectorstore.get()


# In[395]:


response.dict()


# In[363]:


# with this retreiver the context (relevant split) is directry passed to the question addressing the LLM.
response = chain.invoke({"context": retriever, "question": question})
response.content


# In[364]:


response


# #### HandsOn --- Answer not in Splits

# > What happens if the answer is not in the splits of any retreived document?

# In[ ]:


response = chain.invoke(
    {"context": retriever, "question": "What is a large language model?"}
)
response.content


# #### Better Prompts

# In[221]:


from langchain import hub

prompt_hub_rag = hub.pull("rlm/rag-prompt")


# In[222]:


prompt_hub_rag


# In[ ]:


# In[ ]:


# In[ ]:


# In[362]:


answer


# In[393]:


splits[0].metadata["source"]


# In[235]:


# In the final output we might want to know in which document we can find the information of the similarity search.
# This is handled by RunnablePassthrough() function which can add a
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is a large language model?")


# In[230]:


template = "You are an assistant specialized in question-answering tasks. Use the given context to answer the question concisely. If the answer is not present in the context, clearly state that you don't know the answer and do not provide any further answer.\nQuestion: {question} \nContext: {context} \nAnswer:"
prompt = ChatPromptTemplate.from_template(template)
prompt


# In[231]:


rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is a large language model?")


# [RAG chains](https://python.langchain.com/docs/expression_language/get_started#rag-search-example)

# ## HandsOn

# 1. Rerun the overall app using a question which relates to the [website](https://www.themembercompany.com/nl/employeneurship). You first will have to load the website via WebBaseLoader.
# 2. ...

# ## Summary

# Implementation of Simple Retrieval-Augmented Generation (RAG) from Scratch
# - **Ingestion Phase**:
#     - **Load**: Loading documents into the system.
#     - **Split**: Splitting documents into manageable chunks.
#     - **Embed**: Embedding document chunks into vector representations.
#     - **Store**: Storing the embedded documents in a vector store.
# - **Similarity Search**: Searching for relevant documents using embedded query.
# - **Combine Context**: Combining the retrieved document context with the query.
# - **Response Generation**: Generating the final response using a Large Language Model (LLM).

# ## Next Steps

# There is plenty more to discover at [langchain's](https://github.com/langchain-ai) and many other websites! Especially check out: [YouTube](https://www.youtube.com/watch?v=sVcwVQRHIc8&ab_channel=freeCodeCamp.org) and [Github](https://github.com/langchain-ai/rag-from-scratch/tree/main). Here an overview:

# ![assets/imgs/langchain_rag_overview.png](assets/imgs/langchain_rag_overview.png)

# ## TMChampionship

# * TEL organises/-ed a project journey towards a shark tank like investor pitch in November
# * Milan and I - started TMChampionship project Prometheon.ai to build a sustainable manufacturing knowledge expert
# * TEL has many wonderful opportunities for you to try new things, learn, connect and especially grow

# ## Thank You

# - **Technical Support**: Milan and Raul
# - **Organisational Support**: Marlies, Wendy, and Varsha
# - **Motivational Support**: TMChampionship/ TEL/ Pepijn

# ## Feedback

# ![assets/imgs/ai_workshop_tmc__feedback.png](assets/imgs/ai_workshop_tmc__feedback.png)
#
# https://forms.office.com/e/CwRvint3LY?origin=lprLink

# ## Appendix

# - **credits**: this notebook heavily borrows from langchain's [rag_from_scratch_1_to_4.ipynb]("https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_1_to_4.ipynb")

# * **GPT**: Generative Pre-trained Transformer, a type of language model developed by OpenAI that generates human-like text using transformer architecture.
# * **LLM**: Large Language Model, a machine learning model trained on vast amounts of text data to understand and generate human language.
# * **Transformer**: Deep learning model using attention mechanism for context understanding and parallel processing, introduced in the "Attention is All You Need" paper.
# * **Embedding Models**: Convert text to vector representations (e.g., BERT).
# * **Generation Models**: Generate text from prompts (e.g., GPT-3).
# * **Softmax Function**: Converts values to probabilities, used in classification models.
# * **Fine-Tune vs. Retrieval "Augmented Generation**
#     * **Fine-Tuning an LLM**: Adapts model to specific tasks using labeled data.
#     * **RAG (Retrieval-Augmented Generation)**: Combines retrieval with generation for context-specific responses.
# * **Micro Timeline**
#     * **2017**: "Attention is All You Need" paper.
#     * **2018**: BERT, GPT-2
#     * **2020**: GPT-3.
# * **Quantization**: Reduces precision of model parameters.
#     * **Benefits**: Smaller size, faster inference, lower power consumption.
#     * **Types**: Static, Dynamic, Quantization-Aware Training.
#     * **Challenges**: Accuracy loss, hardware support needed.
# * **Not all LLMs are GPTs**: Other models include BERT, T5, XLNet, RoBERTa.
# * **Not all LLMs use transformers**: Other architectures include RNNs, CNNs, MoE, Memory-Augmented Networks.
