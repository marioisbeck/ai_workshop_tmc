import streamlit as st
import requests
from requests.exceptions import ConnectionError, Timeout
import time

# FastAPI server URL
FASTAPI_URL = "http://backend:8001"


def upload_document(file):
    try:
        response = requests.post(f"{FASTAPI_URL}/upload/", files={"file": file})
        response.raise_for_status()
        return response.json()
    except ConnectionError:
        st.error("Failed to connect to the backend service. Please try again later.")
        return None
    except Timeout:
        st.error("The request timed out. Please try again later.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


def query_langserve(question):
    try:
        response = requests.post(f"{FASTAPI_URL}/query/", json={"question": question})
        response.raise_for_status()
        return response.json()
    except ConnectionError:
        st.error("Failed to connect to the backend service. Please try again later.")
        return None
    except Timeout:
        st.error("The request timed out. Please try again later.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


st.title("AI Workshop TMC")

# Upload Document Section
st.header("Upload Document")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    result = upload_document(uploaded_file)
    if result:
        st.write(result)

# Query Section
st.header("Query Document")
question = st.text_input("Enter your question")

if st.button("Submit"):
    if question:
        response = query_langserve(question)
        if response:
            st.write(response)
    else:
        st.error("Please enter a question.")
