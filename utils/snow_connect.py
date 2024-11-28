from typing import Any, Dict
import json
import requests
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from urllib.parse import quote_plus



class SQLConnection:
    """SQL Server connection class"""

    def __init__(self):
        self.connection_parameters = self._get_connection_parameters_from_secrets()
        self.server = self.connection_parameters["server"]
        self.database = self.connection_parameters["database"]
        self.username = self.connection_parameters["username"]
        self.password = self.connection_parameters["password"]
        self.driver = self.connection_parameters["driver"]
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
        if self.engine is not None:
            return self.engine

        # Create a more detailed connection string
        connection_string = (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            "TrustServerCertificate=yes;"
            "Encrypt=yes;"
            "Timeout=30;"
        )
        
        try:
            # Use the pyodbc-style connection string
            self.engine = create_engine(
                f"mssql+pyodbc:///?odbc_connect={quote_plus(connection_string)}",
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT @@VERSION"))
                version = result.scalar()
                print(f"Connected successfully to: {version}")
                
            return self.engine
            
        except Exception as e:
            print(f"Connection failed with error: {str(e)}")
            print("Connection details:")
            print(f"Server: {self.server}")
            print(f"Database: {self.database}")
            print(f"Driver: {self.driver}")
            # Test direct ODBC connection
            try:
                import pyodbc
                conn = pyodbc.connect(connection_string)
                print("Direct ODBC connection successful!")
                conn.close()
            except Exception as odbc_e:
                print(f"Direct ODBC connection failed: {str(odbc_e)}")
            raise

    def execute_query(self, query: str, use_cache: bool = True) -> list:
        """Execute SQL query with caching support"""
        @st.cache_data(ttl=3600) if use_cache else lambda f: f  # Cache for 1 hour if use_cache is True
        def _execute_query(q):
            try:
                with self.get_engine().connect() as connection:
                    result = connection.execute(text(q))
                    return [dict(row) for row in result]
            except Exception as e:
                st.error(f"Database error: {str(e)}")
                return []

        return _execute_query(query)
