from typing import Any, Dict
import os
import json
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from pydantic import BaseModel
from utils.snow_connect import SQLConnection
from sqlalchemy import text
import numpy as np
from dotenv import load_dotenv

load_dotenv() 

class Config(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 0
    docs_dir: str = "docs/"
    docs_glob: str = "**/*.md"
    embeddings_table: str = "document_embeddings"
    schema_name: str = "dbo"

class DocumentProcessor:
    def __init__(self, api_key: str, config: Config):
        try:
            self.sql_connection = SQLConnection()
            self.engine = self.sql_connection.get_engine()
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                print("Database connection successful!")
        except Exception as e:
            print(f"Database connection failed: {str(e)}")
            raise
        self.loader = DirectoryLoader(config.docs_dir, glob=config.docs_glob)
        self.text_splitter = CharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
        self.embeddings_table = f"{config.schema_name}.{config.embeddings_table}"

    def _create_embeddings_table(self):
        check_table_query = text("""
        SELECT COUNT(*) 
        FROM sys.tables t 
        JOIN sys.schemas s ON t.schema_id = s.schema_id 
        WHERE t.name = 'document_embeddings' 
        AND s.name = 'dbo'
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(check_table_query)
            table_exists = result.scalar()
            
            if not table_exists:
                create_table_query = text(f"""
                CREATE TABLE {self.embeddings_table} (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    content TEXT,
                    embedding VARBINARY(MAX),
                    metadata NVARCHAR(MAX)
                )
                """)
                conn.execute(create_table_query)
                conn.commit()

    def store_embeddings(self, documents, embeddings):
        """Store document embeddings in SQL Server"""
        with self.engine.connect() as conn:
            for doc, embedding_array in zip(documents, embeddings):
                # Convert numpy array to bytes directly
                embedding_bytes = np.array(embedding_array).tobytes()
                
                insert_query = text(f"""
                INSERT INTO {self.embeddings_table} (content, embedding, metadata)
                SELECT 
                    :content,
                    CAST(:embedding AS VARBINARY(MAX)),
                    :metadata
                """)
                
                try:
                    conn.execute(
                        insert_query,
                        {
                            "content": doc.page_content,
                            "embedding": embedding_bytes,
                            "metadata": json.dumps(doc.metadata)
                        }
                    )
                except Exception as e:
                    print(f"Error inserting document: {str(e)}")
                    print(f"Content length: {len(doc.page_content)}")
                    print(f"Embedding size: {len(embedding_bytes)} bytes")
                    print(f"Metadata: {doc.metadata}")
                    raise
                
            conn.commit()

    def process(self) -> Dict[str, Any]:
        """Process documents and store embeddings in SQL Server"""
        try:
            # Create table if it doesn't exist
            self._create_embeddings_table()
            
            # Load and split documents
            print("Loading documents...")
            documents = self.loader.load()
            print(f"Loaded {len(documents)} documents")
            
            print("Splitting documents...")
            split_docs = self.text_splitter.split_documents(documents)
            print(f"Split into {len(split_docs)} chunks")
            
            # Create embeddings
            print("Creating embeddings...")
            embeddings = []
            for doc in split_docs:
                embedding = self.embeddings.embed_documents([doc.page_content])[0]
                embeddings.append(embedding)
            print(f"Created {len(embeddings)} embeddings")
            
            # Store in SQL Server
            print("Storing embeddings in database...")
            self.store_embeddings(split_docs, embeddings)
            
            print(f"Successfully processed and stored {len(split_docs)} document chunks")
            return {
                "status": "success", 
                "document_count": len(documents),
                "chunk_count": len(split_docs)
            }
            
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            raise

def main():
    print("Starting document processing...")
    
    # Get OpenAI API key from environment variable
    api_key = os.getenv('NVIDIA_API_KEY')
    if not api_key:
        raise ValueError("Please set NVIDIA_API_KEY environment variable")
    
    config = Config()
    processor = DocumentProcessor(api_key=api_key, config=config)
    
    result = processor.process()
    print(f"Processing complete: {result}")

if __name__ == "__main__":
    main()
