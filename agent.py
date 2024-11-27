import os
import streamlit as st
from dataclasses import dataclass
from typing import Annotated, Sequence, Optional

from langchain.callbacks.base import BaseCallbackHandler
# from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_community.llms import Replicate
from langchain.agents import Tool

from tools import retriever_tool, search
from PIL import Image
from io import BytesIO
from utils.snow_connect import SQLConnection

@dataclass
class MessagesState:
    messages: Annotated[Sequence[BaseMessage], add_messages]


memory = MemorySaver()


@dataclass
class ModelConfig:
    model_name: str
    api_key: str
    base_url: Optional[str] = None

model_configurations = {
    # "gpt-4o": ModelConfig(
    #     model_name="gpt-4o", 
    #     api_key=os.getenv("OPENAI_API_KEY")
    #     # OpenAI API key from .env is sufficient, no need for explicit base URL
    # )
    # "gpt-4o": ModelConfig(
    #     model_name="gemini-pro",
    #     api_key=st.secrets["GOOGLE_API_KEY"],
    #     base_url="https://generativelanguage.googleapis.com/v1",
    # ),
    # "Mistral 7B": ModelConfig(
    #     model_name="mistralai/mistral-7b-v0.1", api_key=st.secrets["REPLICATE_API_TOKEN"]
    # ),
    # "llama-3.2-3b": ModelConfig(
    #     model_name="accounts/fireworks/models/llama-v3p2-3b-instruct",
    #     api_key=st.secrets["FIREWORKS_API_KEY"],
    #     base_url="https://api.fireworks.ai/inference/v1",
    # ),
    # "llama-3.1-405b": ModelConfig(
    #     model_name="accounts/fireworks/models/llama-v3p1-405b-instruct",
    #     api_key=st.secrets["FIREWORKS_API_KEY"],
    #     base_url="https://api.fireworks.ai/inference/v1",
    # ),
    "Llama 3.1 70B": ModelConfig(
        model_name="nvidia/llama-3.1-nemotron-70b-instruct",
        api_key=st.secrets["NVIDIA_API_KEY"],
        base_url="https://integrate.api.nvidia.com/v1",
    )

    
}
sys_msg = SystemMessage(
    content="""You're an AI assistant specializing in data analysis with SQL server DB. When providing responses, strive to exhibit friendliness and adopt a conversational tone, similar to how a friend or tutor would communicate. Do not ask the user for schema or database details. You have access to the following tools:
    - Database_Schema: This tool allows you to search for database schema details when needed to generate the SQL code.
    - Internet_Search: This tool allows you to search the internet for SQL server DB related information when needed to generate the SQL code.
    - SQL_Executor: This tool allows you to execute SQL server DB queries when needed to generate the SQL code. You only have read access to the database, do not modify the database in any way.

    Make sure to always return both the SQL code and the result of the query
    """
)
tools = [
    Tool(
        name="sql_query",
        func=retriever_tool().run,
        description="Useful for when you need to execute SQL queries"
    ),
    Tool(
        name="search",
        func=search().run,
        description="Useful for when you need to search for information in the database"
    )
]

def create_agent(callback_handler: BaseCallbackHandler, model_name: str) -> StateGraph:
    config = model_configurations.get(model_name)
    if not config:
        raise ValueError(f"Unsupported model name: {model_name}")

    if not config.api_key:
        raise ValueError(f"API key for model '{model_name}' is not set. Please check your environment variables or secrets configuration.")

    llm = ChatOpenAI(
        model=config.model_name,
        api_key=config.api_key,
        callbacks=[callback_handler],
        streaming=True,
        base_url=config.base_url,
        temperature=0.01,
        default_headers={"HTTP-Referer": "https://sql-bot.streamlit.app/", "X-Title": "SQL Bot"},
    )

    llm_with_tools = llm.bind_tools(tools)

    def llm_agent(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state.messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("llm_agent", llm_agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "llm_agent")
    builder.add_conditional_edges("llm_agent", tools_condition)
    builder.add_edge("tools", "llm_agent")

    react_graph = builder.compile(checkpointer=memory)

    # png_data = react_graph.get_graph(xray=True).draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png_data)

    # image = Image.open(BytesIO(png_data))
    # st.image(image, caption="React Graph")

    return react_graph
