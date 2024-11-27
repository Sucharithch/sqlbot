from typing import Any, Dict
import json
import requests
import streamlit as st
from sqlalchemy import create_engine


class SQLConnection:
    """SQL Server connection class"""

    def __init__(self):
        self.connection_parameters = self._get_connection_parameters_from_secrets()
        self.engine = None
        

    @staticmethod
    def _get_connection_parameters_from_secrets() -> Dict[str, Any]:
        try:
            return {
                "server": st.secrets["sql_connection"]["server"],
                "database": st.secrets["sql_connection"]["database"],
                "username": st.secrets["sql_connection"]["username"],
                "password": st.secrets["sql_connection"]["password"],
                "driver": st.secrets["sql_connection"]["driver"]
            }
        except Exception as e:
            print(f"Error loading secrets: {str(e)}")
            # Fallback to hardcoded values if secrets are not available
            return {
                "server": "10.0.0.200",
                "database": "AdventureWorksDW2019",
                "username": "sa",
                "password": "btcde@123",
                "driver": "ODBC Driver 17 for SQL Server"
            }

    def get_engine(self):
        if self.engine is None:
            params = self.connection_parameters
            conn_str = f"mssql+pyodbc://{params['username']}:{params['password']}@{params['server']}/{params['database']}?driver={params['driver'].replace(' ', '+')}"
            self.engine = create_engine(conn_str)
        return self.engine

    def execute_query(self, query: str, use_cache: bool = True) -> list:
        engine = self.get_engine()
        with engine.connect() as connection:
            result = connection.execute(query)
            result_list = [dict(row) for row in result]
        return result_list
