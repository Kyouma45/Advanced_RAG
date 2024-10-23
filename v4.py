import streamlit as st
import os
import json
import shutil
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Initialize session state
if 'projects' not in st.session_state:
    st.session_state.projects = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_project' not in st.session_state:
    st.session_state.current_project = None


# Helper functions
def load_projects():
    try:
        with open('./data.json', 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_projects():
    with open('./data.json', 'w') as file:
        json.dump(st.session_state.projects, file)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Text cleaning (same as in the original script)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\\x', '', text)
    text = re.sub(r'\(\d+\)', '', text)
    text = re.sub(r'\b\d+(\.\d+)?k\b', '', text)
    text = re.sub(r'\b\d+(\.\d+)?(/\d+(\.\d+)?)?\b', '', text)
    text = re.sub(r'\[SEP\\]|\[CLS\\]|\[SEP]|\[CLS]', '', text)
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks, project_name):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(f"./projects/{project_name}_faiss")


# Streamlit app
def main():
    st.title("PDF Question Answering System")

    # Load projects
    st.session_state.projects = load_projects()

    # Sidebar for project management
    st.sidebar.title("Project Management")
    project_action = st.sidebar.radio("Choose an action:",
                                      ["Create Project", "Load Project", "Delete Project", "List Projects"])

    if project_action == "Create Project":
        create_project()
    elif project_action == "Load Project":
        load_project()
    elif project_action == "Delete Project":
        delete_project()
    elif project_action == "List Projects":
        list_projects()

    # Main area for chat interface
    if st.session_state.current_project:
        st.write(f"Current Project: {st.session_state.current_project}")
        chat_interface()


def create_project():
    st.sidebar.subheader("Create a New Project")
    project_name = st.sidebar.text_input("Enter project name")
    pdf_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if st.sidebar.button("Create Project"):
        if project_name and pdf_files:
            if project_name not in st.session_state.projects:
                st.session_state.projects.append(project_name)
                os.makedirs(f"./projects/{project_name}", exist_ok=True)

                with st.spinner("Processing PDFs..."):
                    text = get_pdf_text(pdf_files)
                    text_chunks = get_text_chunks(text)
                    get_vector_store(text_chunks, project_name)

                save_projects()
                st.sidebar.success(f"Project '{project_name}' created successfully")
            else:
                st.sidebar.error(f"Project '{project_name}' already exists")
        else:
            st.sidebar.error("Please enter a project name and upload PDF files")


def load_project():
    st.sidebar.subheader("Load a Project")
    project_name = st.sidebar.selectbox("Select a project", st.session_state.projects)

    if st.sidebar.button("Load Project"):
        if project_name:
            st.session_state.current_project = project_name
            st.session_state.chat_history = []
            st.sidebar.success(f"Project '{project_name}' loaded successfully")
        else:
            st.sidebar.error("Please select a project")


def delete_project():
    st.sidebar.subheader("Delete a Project")
    project_name = st.sidebar.selectbox("Select a project to delete", st.session_state.projects)

    if st.sidebar.button("Delete Project"):
        if project_name:
            try:
                shutil.rmtree(f'./projects/{project_name}_faiss')
                st.session_state.projects.remove(project_name)
                save_projects()
                st.sidebar.success(f"Project '{project_name}' has been removed")
                if st.session_state.current_project == project_name:
                    st.session_state.current_project = None
                    st.session_state.chat_history = []
            except Exception as e:
                st.sidebar.error(f"Error deleting project: {e}")
        else:
            st.sidebar.error("Please select a project to delete")


def list_projects():
    st.sidebar.subheader("Available Projects")
    for project in st.session_state.projects:
        st.sidebar.write(f"- {project}")


def chat_interface():
    st.subheader("Chat Interface")

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.write(f"You: {message.content}")
        else:
            st.write(f"Assistant: {message}")

    # Chat input
    user_input = st.text_input("Ask a question:")
    if st.button("Send"):
        if user_input:
            process_user_input(user_input)


def process_user_input(user_input):
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4")

    # Load the vector store
    vector_store = FAISS.load_local(f"./projects/{st.session_state.current_project}_faiss", embeddings,
                                    allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()

    # Set up the retrieval chain
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question, formulate a standalone question which can be understood "
         "without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it "
         "as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer "
         "the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and "
         "keep the answer concise.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt, output_parser=StrOutputParser())
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Process the user input
    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})

    # Update chat history
    st.session_state.chat_history.extend([HumanMessage(content=user_input), response["answer"]])

    # Display the latest response
    st.write(f"Assistant: {response['answer']}")


if __name__ == "__main__":
    main()
