import streamlit as st
import requests
from requests.exceptions import ConnectionError, Timeout
import json

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


def query_directly(question):
    try:
        response = requests.post(
            f"{FASTAPI_URL}/query_directly/", json={"question": question}
        )
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


st.sidebar.title("AI Workshop TMC")

# Upload Document Section
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if uploaded_file.size > 5 * 1024 * 1024:
        st.sidebar.error("File size should be less than 5MB.")
        uploaded_file = None

if uploaded_file is not None:
    result = upload_document(uploaded_file)
    if result:
        st.sidebar.write(result)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the backend service
    if uploaded_file is not None:
        response = query_langserve(prompt)
    else:
        response = query_directly(prompt)

    # Display assistant response in chat message container
    if response:
        if "error" in response:
            assistant_message = response["error"]
        else:
            try:
                # Try to extract and parse the "Answer" field
                answer = response.get("Answer", "")
                parsed_answer = (
                    json.loads(answer) if isinstance(answer, str) else answer
                )
                assistant_message = json.dumps(parsed_answer, indent=4)
            except (ValueError, TypeError, json.JSONDecodeError):
                # If parsing fails, just convert the response to a string
                assistant_message = str(response)

        with st.chat_message("assistant"):
            st.markdown(
                f"<pre style='background-color: #f0f0f0; padding: 10px; border-radius: 10px;'>{assistant_message}</pre>",
                unsafe_allow_html=True,
            )

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_message}
        )
