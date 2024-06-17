# Use the latest version of the Jupyter base notebook image
FROM quay.io/jupyter/base-notebook:ubuntu-22.04

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/requirements.txt

# Install necessary Python packages using pip
# - langchain: A library for building language models
# - streamlit: An app framework for Machine Learning and Data Science
# - streamlit_chat: A Streamlit component for chat applications
# - chromadb: A database for managing chromatic data
# - pypdf: A library for working with PDF files
# - fastembed: A library for fast embedding of data
# RUN pip install langchain streamlit streamlit_chat chromadb pypdf fastembed langchain_community alembic==1.7.7

# Install the Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt
