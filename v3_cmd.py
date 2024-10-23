import typer
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import shutil
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tracers.log_stream import LogEntry, LogStreamCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain import hub
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress

load_dotenv()

console = Console()
app = typer.Typer()


def load_projects():
    try:
        with open('./data.json', 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def get_pdf_text(pdf_docs, progress, task):
    text = ""
    for i, pdf in enumerate(pdf_docs):
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        progress.update(task, advance=100 * (i + 1) / len(pdf_docs))
    # Remove special characters
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\\x', '', text)
    # Remove numbers in parentheses
    text = re.sub(r'\(\d+\)', '', text)
    # Remove numerical values with "k"
    text = re.sub(r'\b\d+(\.\d+)?k\b', '', text)
    # Remove other numerical sequences (including those with slashes)
    text = re.sub(r'\b\d+(\.\d+)?(/\d+(\.\d+)?)?\b', '', text)
    # Remove token-like patterns and placeholders
    text = re.sub(r'\[SEP\\]|\[CLS\\]|\[SEP]|\[CLS]', '', text)
    # Remove words that contain both alphabets and numbers
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_text_chunks(text, progress, task):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    progress.update(task, advance=100)
    return chunks


class ProjectManager:
    def __init__(self):
        self.projects = load_projects()
        # self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o")
        self.summary = ""
        prompt_template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the \
        question. If you don't know the answer, just say that you don't know. Use ten sentences maximum and keep the \
        answer concise.
        Question: {question}
        Context: {context}
        Answer:"""
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        # self.prompt = hub.pull("rlm/rag-prompt")
        # console.print(f"[bold red]{self.prompt}[/bold red]")

    def save_projects(self):
        with open('./data.json', 'w') as file:
            json.dump(self.projects, file)

    def create_project(self):
        project_name = Prompt.ask("Enter the project name")
        if project_name in self.projects:
            console.print(f"[bold red]Error: Project '{project_name}' already exists.[/bold red]")
            return

        self.projects.append(project_name)
        os.makedirs(f"./projects/{project_name}", exist_ok=True)
        console.print(f"[bold green]Project '{project_name}' created successfully[/bold green]")

        path = Prompt.ask("Please enter path to the PDF files you want to use for the project")
        pdf_docs = [os.path.join(path, file) for file in os.listdir(path) if file.lower().endswith('.pdf')]

        if not pdf_docs:
            console.print("[bold red]No PDF files found in the specified directory.[/bold red]")
            return

        with Progress() as progress:
            task1 = progress.add_task("[green]Processing PDFs...", total=100)
            text = get_pdf_text(pdf_docs, progress, task1)

            task2 = progress.add_task("[blue]Chunking text...", total=100)
            text_chunks = get_text_chunks(text, progress, task2)

            task3 = progress.add_task("[yellow]Creating vector store...", total=100)
            self.get_vector_store(text_chunks, project_name, progress, task3)

        console.print("[bold green]Text Processing Complete[/bold green]")
        self.save_projects()

    def get_vector_store(self, text_chunks, project_name, progress, task):
        vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
        vector_store.save_local(f"./projects/{project_name}_faiss")
        # keyword_retriever = BM25Retriever.from_texts(text_chunks)
        # keyword_retriever.persist(f"./projects/{project_name}_keyword")
        # keyword_retriever.save_local(f"./projects/{project_name}_keyword")
        progress.update(task, advance=100)

    def load_project(self, project_name):
        try:
            new_db = FAISS.load_local(f"./projects/{project_name}_faiss", self.embeddings,
                                      allow_dangerous_deserialization=True)
        except Exception as e:
            console.print(f"[bold red]Error loading project: {e}[/bold red]")
            return

        retriever = new_db.as_retriever()
        # retriever = EnsembleRetriever(retrievers=[retriever_vector,keyword_db],
        #                                weights=[0.5, 0.5])
        console.print(f"[bold green]Project '{project_name}' loaded successfully[/bold green]")

        chat_history = []

        while True:
            question = Prompt.ask("\nEnter your question", default="exit")
            if question.lower() == "exit":
                break

            # LangChain Expression Language (LCEL)
            # rag_chain = (
            #     {"context": retriever, "question": RunnablePassthrough()}
            #     | self.prompt
            #     | self.llm
            #     | StrOutputParser()
            # )

            # with console.status("[bold green]Thinking..."):
            #     response = rag_chain.invoke(question)

            # Run Parallel, also returns list of documents retrieved (LCEL)
            # rag_chain_from_docs = (
            #     RunnablePassthrough()
            #     | self.prompt
            #     | self.llm
            #     | StrOutputParser()
            # )

            # rag_chain_with_source = RunnableParallel(
            #     {"context": retriever, "question": RunnablePassthrough()}
            # ).assign(answer=rag_chain_from_docs)

            # with console.status("[bold green]Thinking..."):
            #   response = rag_chain_with_source.invoke(question)

            # With chat history
            # from operator import itemgetter

            # contextualize_q_system_prompt = """Given a chat history and the latest user question \
            # which might reference context in the chat history, formulate a standalone question \
            # which can be understood without the chat history. Do NOT answer the question, \
            # just reformulate it if needed and otherwise return it as is."""
            # contextualize_q_prompt = ChatPromptTemplate.from_messages(
            #     [
            #         ("system", contextualize_q_system_prompt),
            #         MessagesPlaceholder(variable_name="chat_history"),
            #         ("human", "{question}"),
            #     ]
            # )
            # contextualize_q_chain = (contextualize_q_prompt | llm | StrOutputParser()).with_config(
            #     tags=["contextualize_q_chain"]
            # )

            # qa_system_prompt = """You are an assistant for question-answering tasks. \
            # Use the following pieces of retrieved context to answer the question. \
            # If you don't know the answer, just say that you don't know. \
            # Use three sentences maximum and keep the answer concise.\

            # {context}"""
            # qa_prompt = ChatPromptTemplate.from_messages(
            #     [
            #         ("system", qa_system_prompt),
            #         MessagesPlaceholder(variable_name="chat_history"),
            #         ("human", "{question}"),
            #     ]
            # )

            # def contextualized_question(input: dict):
            #     if input.get("chat_history"):
            #         return contextualize_q_chain
            #     else:
            #         return input["question"]

            # rag_chain = (
            #     RunnablePassthrough.assign(context=contextualize_q_chain | retriever | format_docs)
            #     | qa_prompt
            #     | llm
            # )

            # response = rag_chain.invoke({"chat_history": chat_history, "question": question})

            # Memory using chains
            contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            history_aware_retriever = create_history_aware_retriever(
                self.llm, retriever, contextualize_q_prompt
            )

            qa_system_prompt = """You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            Use three sentences maximum and keep the answer concise.\

            {context}"""
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            with console.status("[bold green]Thinking..."):
                question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt,
                                                                     output_parser=StrOutputParser())

                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                response = rag_chain.invoke({"input": question, "chat_history": chat_history})

                console.print(Panel(response['answer'], title="Answer", border_style="blue"))

            chat_history.extend([HumanMessage(content=question), response["answer"]])

    def delete_project(self, project_name):
        if project_name not in self.projects:
            console.print(f"[bold red]Error: Project '{project_name}' not found.[/bold red]")
            return

        try:
            shutil.rmtree(f'./projects/{project_name}_faiss')
            # shutil.rmtree(f'./projects/{project_name}_keyword')
            self.projects.remove(project_name)
            self.save_projects()
            console.print(f"[bold green]Project '{project_name}' has been removed.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error deleting project: {e}[/bold red]")

    def list_projects(self):
        if not self.projects:
            console.print("[yellow]No projects found.[/yellow]")
        else:
            console.print("[bold]Available projects:[/bold]")
            for i, project in enumerate(self.projects, 1):
                console.print(f"  {i}. {project}")


def main_menu(pm: ProjectManager):
    while True:
        console.rule("[bold blue]PDF Question Answering System[/bold blue]")
        console.print("\n[bold cyan]Main Menu:[/bold cyan]")
        console.print("1. Create a new project")
        console.print("2. Load an existing project")
        console.print("3. Delete a project")
        console.print("4. List all projects")
        console.print("5. Exit")

        choice = Prompt.ask("Enter your choice", choices=["1", "2", "3", "4", "5"])

        if choice == "1":
            pm.create_project()
        elif choice == "2":
            pm.list_projects()
            if pm.projects:
                project_name = Prompt.ask("Enter the project name to load")
                if project_name in pm.projects:
                    pm.load_project(project_name)
                else:
                    console.print("[bold red]Invalid project name.[/bold red]")
        elif choice == "3":
            pm.list_projects()
            if pm.projects:
                project_name = Prompt.ask("Enter the project name to delete")
                if project_name in pm.projects:
                    pm.delete_project(project_name)
                else:
                    console.print("[bold red]Invalid project name.[/bold red]")
        elif choice == "4":
            pm.list_projects()
        elif choice == "5":
            console.print("[bold green]Saving Complete[/bold green]")
            break

        console.print("\n")


@app.command()
def start():
    pm = ProjectManager()
    main_menu(pm)


if __name__ == "__main__":
    app()
