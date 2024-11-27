from typing import Any, Dict, Optional, Type, ClassVar, List
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
import pandas as pd
from utils.snow_connect import SQLConnection
import re

class SQLInput(BaseModel):
    query: str = Field(..., description="The SQL query to execute")

class SearchInput(BaseModel):
    query: str = Field(..., description="The search query to find relevant information")

class SQLQueryTool(BaseTool):
    name: ClassVar[str] = "sql_query"
    description: ClassVar[str] = "Execute SQL queries on the database"
    args_schema: Type[BaseModel] = SQLInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Any:
        """Execute the SQL query."""
        # Check for unsafe queries
        if re.match(r"^\s*(drop|alter|truncate|delete|insert|update)\s", query, re.I):
            return "Sorry, I can't execute queries that modify the database."
        
        try:
            # Create SQL connection
            sql_conn = SQLConnection()
            
            # Execute query
            result = sql_conn.execute_query(query, use_cache=False)
            
            # Convert to DataFrame if results exist
            if result:
                df = pd.DataFrame(result)
                return df.to_string()
            return "No results found."
            
        except Exception as e:
            return f"Error executing query: {str(e)}"

class SearchTool(BaseTool):
    name: ClassVar[str] = "search"
    description: ClassVar[str] = "Search for information in the database"
    args_schema: Type[BaseModel] = SearchInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Any:
        """Search for information in the database."""
        try:
            # Create SQL connection
            sql_conn = SQLConnection()
            
            # Create a search query using SQL Server's full-text search capabilities
            search_query = f"""
            SELECT TOP 5 content, metadata
            FROM document_embeddings
            WHERE CONTAINS(content, '{query}')
            OR content LIKE '%{query}%'
            ORDER BY 
                CASE 
                    WHEN CONTAINS(content, '{query}') THEN 1
                    WHEN content LIKE '%{query}%' THEN 2
                    ELSE 3
                END
            """
            
            result = sql_conn.execute_query(search_query, use_cache=False)
            
            if result:
                # Format the results
                formatted_results = []
                for item in result:
                    content = item.get('content', '')
                    metadata = item.get('metadata', '{}')
                    formatted_results.append(f"Content: {content}\nMetadata: {metadata}\n---")
                return "\n".join(formatted_results)
            return "No relevant information found."
            
        except Exception as e:
            return f"Error searching: {str(e)}"

def retriever_tool():
    """Create and return the SQL query tool."""
    return SQLQueryTool()

def search() -> SearchTool:
    """Create and return the search tool."""
    return SearchTool()
