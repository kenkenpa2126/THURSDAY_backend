import os
from tempfile import TemporaryDirectory
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant

from langchain.agents import Tool, load_tools, initialize_agent, AgentType, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

from langchain.tools.render import format_tool_to_openai_function
from langchain.tools import DuckDuckGoSearchRun
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.document_loaders import TextLoader

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

import base64

from . import settings
from .models import request_models, response_models

app = FastAPI()

EMBEDDING_MODEL = OpenAIEmbeddings(
    openai_api_key=settings.OPENAI_API_KEY,
    model="text-embedding-ada-002"
)

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

QDRANT_PATH = "localhost"
COLLECTION_NAME = "retrieved_pdf"
client = QdrantClient(path=QDRANT_PATH, port=6333)
def qdrants_load(collection_name):
    collection_names = [collection.name for collection in client.get_collections().collections]
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    return Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=EMBEDDING_MODEL
    )

def read_files_in_pdf_dir():
    docs = []
    for filename in os.listdir("./pdf/"):
        filepath = os.path.join("./pdf/", filename)
        if filename.endswith('.txt'):
            loader = TextLoader(filepath)
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            pages = loader.load_and_split(text_splitter)
            docs.extend(pages)
        elif filename.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            pages = loader.load_and_split(text_splitter)
            docs.extend(pages)
    if len(docs) != 0:
        qdrant = qdrants_load(COLLECTION_NAME)
        qdrant.add_documents(docs)
        print("succeded to read files!")

read_files_in_pdf_dir()
llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=settings.OPENAI_API_KEY)

@app.put("/pdf")
def read_pdf(input_value: request_models.StorePDF):
    pdf_data = base64.b64decode(input_value.pdf)
    with TemporaryDirectory() as tempdir:
        file_name = f"{tempdir}/1.pdf"
        with open(file_name, "wb") as f:
            f.write(pdf_data)
        loader = PyPDFLoader(f"{tempdir}/1.pdf")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        pages = loader.load_and_split(text_splitter)
    qdrant = qdrants_load(COLLECTION_NAME)
    qdrant.add_documents(pages)

    return "OK"

@app.put("/text")
def read_pdf(input_value: request_models.StoreText):
    text_data = input_value.text
    with TemporaryDirectory() as tempdir:
        file_name = f"{tempdir}/1.text"
        with open(file_name, "wb") as f:
            f.write(text_data)
        loader = TextLoader(f"{tempdir}/1.txt")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        pages = loader.load_and_split(text_splitter)
    qdrant = qdrants_load(COLLECTION_NAME)
    qdrant.add_documents(pages)

    return "OK"

@app.put("/question")
def question(qa_params: request_models.AskAgent):
    qdrant = qdrants_load(COLLECTION_NAME)
    retriever = qdrant.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=OpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=settings.OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        verbose=False)

    # result = qa({"question": qa_params.question}, return_only_outputs=True)
    result = qa({"question": qa_params.message.replace("pdf", "") + "この質問の返答を日本語で答えてください。"}, return_only_outputs=True)
    return response_models.AskQuestionResponse(answer=result['answer'], status="ok")

def get_agent_executor():
    search = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="duckduckgo-search",
            func=search.run,
            description="ウェブで最新の情報を検索する必要がある場合に便利です。"
        ),
        # Tool(
        #     name=COLLECTION_NAME,
        #     func=qa.run,
        #     description="ユーザーが意図したテキストを検索する必要がある場合に便利です。"
        # )
        
    ]

    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True,
    # )

    MEMORY_KEY = "chat_history"
    prompt = ChatPromptTemplate.from_messages([
        ("system", "あなたは親切なアシスタントです。Webを検索して、ユーザーの質問に答えてください。"),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm_with_tools = llm.bind(
        functions=[format_tool_to_openai_function(t) for t in tools]
    )

    agent = {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps']),
        "chat_history": lambda x: x["chat_history"]
    } | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()
    # agent = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

chat_history = []

@app.put("/conversation")
def conversation(qa_params: request_models.AskAgent):
    message = qa_params.message
    agent_executor = get_agent_executor()
    result = agent_executor.invoke({"input": message, "chat_history": chat_history})
    chat_history.append(HumanMessage(content=message))
    chat_history.append(AIMessage(content=result['output']))
    return response_models.AgentResponse(answer=result['output'], status="ok")

