import os
import re
import warnings

import streamlit as st
from langchain_core.messages import HumanMessage

from agent import MessagesState, create_agent

from utils.snow_connect import SQLConnection
from utils.snowchat_ui import StreamlitUICallbackHandler, message_func
from utils.snowddl import Snowddl
import pandas as pd

warnings.filterwarnings("ignore")
chat_history = []
snow_ddl = Snowddl()

# Initialize database connection at startup
@st.cache_resource
def init_db():
    db = SQLConnection()
    return db

# Initialize the database connection
sql_conn = init_db()
if not sql_conn:
    st.stop()

# Try to get OpenAI API key from secrets or environment variable
from dotenv import load_dotenv

load_dotenv()

nvidia_api_key = "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"
# os.getenv('NVIDIA_API_KEY')
# if nvidia_api_key:
#     os.environ['NVIDIA_API_KEY'] = nvidia_api_key
# else:
#     # Provide a text input for the API key
#     api_key = st.text_input("Enter your NVIDIA API key:", type="password")
#     if api_key:
#         os.environ['NVIDIA_API_KEY'] = api_key
#     else:
#         st.error("Please provide an NVIDIA API key!")
#         st.stop()

gradient_text_html = """
<style>
.gradient-text {
    font-weight: bold;
    background: -webkit-linear-gradient(left, red, orange);
    background: linear-gradient(to right, red, orange);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 3em;
}
</style>
<div class="gradient-text">snowChat</div>
"""

st.markdown(gradient_text_html, unsafe_allow_html=True)

st.caption("Talk your way through data")

model_options = {
    # "gpt-4o": "GPT-4o",
    "Llama 3.1 70B": "Llama 3.1 70B",
#     "llama-3.1-405b": "Llama 3.1 405B",
#     "llama-3.2-3b": "Llama 3.2 3B",
#     "Gemini Pro 1.5": "Gemini Pro 1.5",
# 
}

model = st.radio(
    "Choose your AI Model:",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
    index=0,
    horizontal=True,
)
st.session_state["model"] = model

if "assistant_response_processed" not in st.session_state:
    st.session_state["assistant_response_processed"] = True  # Initialize to True

if "toast_shown" not in st.session_state:
    st.session_state["toast_shown"] = False

if "rate-limit" not in st.session_state:
    st.session_state["rate-limit"] = False

# # Show the toast only if it hasn't been shown before
# if not st.session_state["toast_shown"]:
#     st.toast("The snowflake data retrieval is disabled for now.", icon="👋")
#     st.session_state["toast_shown"] = True

# Show a warning if the model is rate-limited
if st.session_state["rate-limit"]:
    st.toast("Probably rate limited.. Go easy folks", icon="⚠️")
    st.session_state["rate-limit"] = False

if st.session_state["model"] == "Mixtral 8x7B":
    st.warning("This is highly rate-limited. Please use it sparingly", icon="⚠️")

INITIAL_MESSAGE = [
    {"role": "user", "content": "Hi!"},
    {
        "role": "assistant",
        "content": "Hey there! I'm your SQL assistant, ready to help you query the database and find the information you need! 📊",
    },
]
config = {"configurable": {"thread_id": "42"}}

with open("ui/sidebar.md", "r") as sidebar_file:
    sidebar_content = sidebar_file.read()

with open("ui/styles.md", "r") as styles_file:
    styles_content = styles_file.read()

st.sidebar.markdown(sidebar_content)

selected_table = st.sidebar.selectbox(
    "Select a table:", options=list(snow_ddl.ddl_dict.keys())
)
st.sidebar.markdown(f"### DDL for {selected_table} table")
st.sidebar.code(snow_ddl.ddl_dict[selected_table], language="sql")

# Add a reset button
if st.sidebar.button("Reset Chat"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state["messages"] = INITIAL_MESSAGE
    st.session_state["history"] = []

st.sidebar.markdown(
    "**Note:** Use natural language to ask questions about the data.",
    unsafe_allow_html=True,
)

st.write(styles_content, unsafe_allow_html=True)

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = INITIAL_MESSAGE

if "history" not in st.session_state:
    st.session_state["history"] = []

if "model" not in st.session_state:
    st.session_state["model"] = model

# Prompt for user input and save
if prompt := st.chat_input():
    if len(prompt) > 500:
        st.error("Input is too long! Please limit your message to 500 characters.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state["assistant_response_processed"] = False  # Assistant response not yet processed

messages_to_display = st.session_state.messages.copy()
# if not st.session_state["assistant_response_processed"]:
#     # Exclude the last assistant message if assistant response not yet processed
#     if messages_to_display and messages_to_display[-1]["role"] == "assistant":
#         print("\n\nthis is messages_to_display \n\n", messages_to_display)
#         messages_to_display = messages_to_display[:-1]

for message in messages_to_display:
    message_func(
        message["content"],
        is_user=(message["role"] == "user"),
        is_df=(message["role"] == "data"),
        model=model,
    )

callback_handler = StreamlitUICallbackHandler(model)

react_graph = create_agent(callback_handler, st.session_state["model"])


def append_chat_history(question, answer):
    st.session_state["history"].append((question, answer))


def get_sql(text):
    sql_match = re.search(r"```sql\n(.*)\n```", text, re.DOTALL)
    return sql_match.group(1) if sql_match else None


def append_message(content, role="assistant"):
    """Appends a message to the session state messages."""
    if content.strip():
        st.session_state.messages.append({"role": role, "content": content})


def handle_sql_exception(query, conn, e, retries=2):
    """Handle SQL execution errors"""
    error_message = f"Error executing query: {str(e)}"
    print(error_message)  # For debugging
    append_message(f"I encountered an error: {error_message}")
    return None


def execute_sql(query, conn, retries=2):
    """Execute SQL query with better error handling"""
    if not query:
        return None
        
    # Check for unsafe operations
    if re.match(r"^\s*(drop|alter|truncate|delete|insert|update)\s", query, re.I):
        st.error("Sorry, I can't execute queries that modify the database.")
        return None
        
    try:
        # Execute query using SQLConnection with caching
        result = conn.execute_query(query, use_cache=True)
        if result:
            df = pd.DataFrame(result)
            return df
        return None
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        if "timeout" in str(e).lower():
            st.warning("The query timed out. Please try a simpler query or contact support.")
        return None


if (
    "messages" in st.session_state
    and st.session_state["messages"][-1]["role"] == "user"
    and not st.session_state["assistant_response_processed"]
):
    user_input_content = st.session_state["messages"][-1]["content"]

    if isinstance(user_input_content, str):
        try:
            # Start loading animation
            with st.spinner("Thinking..."):
                callback_handler.start_loading_message()

                messages = [HumanMessage(content=user_input_content)]
                state = MessagesState(messages=messages)
                result = react_graph.invoke(state, config=config, debug=True)

                if result["messages"]:
                    assistant_message = callback_handler.final_message
                    append_message(assistant_message)
                    
                    # Check for SQL query in the response
                    sql_query = get_sql(assistant_message)
                    if sql_query:
                        with st.spinner("Executing query..."):
                            df = execute_sql(sql_query, sql_conn)
                            if df is not None and not df.empty:
                                # Display the DataFrame with pagination
                                st.write("Query Results:")
                                page_size = 10
                                total_rows = len(df)
                                num_pages = (total_rows + page_size - 1) // page_size
                                
                                if num_pages > 1:
                                    page = st.number_input(
                                        "Page", 
                                        min_value=1, 
                                        max_value=num_pages, 
                                        value=1
                                    )
                                    start_idx = (page - 1) * page_size
                                    end_idx = min(start_idx + page_size, total_rows)
                                    st.dataframe(df.iloc[start_idx:end_idx])
                                    st.caption(f"Showing {start_idx+1}-{end_idx} of {total_rows} rows")
                                else:
                                    st.dataframe(df)
                                
                                # Store a summary in chat history
                                summary = f"Query returned {total_rows} rows."
                                append_message(summary, "data")
                            else:
                                st.info("Query returned no results.")
                
                st.session_state["assistant_response_processed"] = True
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state["assistant_response_processed"] = True


if (
    st.session_state["model"] == "Mixtral 8x7B"
    and st.session_state["messages"][-1]["content"] == ""
):
    st.session_state["rate-limit"] = True
