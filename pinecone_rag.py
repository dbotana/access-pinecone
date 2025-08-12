import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PineconeRAG:
    def __init__(self, 
                 openai_api_key: str,
                 pinecone_api_key: str, 
                 pinecone_index_name: str,
                 embedding_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-4o-mini"):
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        
        logger.info(f"Initialized PineconeRAG with index: {pinecone_index_name}")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for query text"""
        text = text.replace("\n", " ")
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model,
            dimensions=1024  # Match your index dimensions
        )
        return response.data[0].embedding

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents in Pinecone"""
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            documents = []
            for match in results["matches"]:
                documents.append({
                    'content': match['metadata']['text'],
                    'score': match['score'],
                    'source': {
                        'file_name': match['metadata']['filename'],
                        'chunk': match['metadata']['chunk'],
                        'total_chunks': match['metadata'].get('total_chunks', 'Unknown'),
                        'source_type': 'PDF'
                    }
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def generate_response(self, query: str, documents: List[Dict], chat_history: List = None) -> Dict[str, Any]:
        """Generate response using retrieved documents"""
        try:
            # Prepare context from retrieved documents
            context = "\n\n".join([
                f"Document {i+1} (Score: {doc['score']:.3f}):\n{doc['content']}"
                for i, doc in enumerate(documents)
            ])
            
            # Prepare chat history
            messages = [
                {
                    "role": "system", 
                    "content": """You are a helpful assistant specializing in disability science research. 
                    Use the provided context to answer questions about disability data, research, employment, 
                    education, and related topics. If the information isn't in the context, say so clearly."""
                }
            ]
            
            # Add chat history if provided
            if chat_history:
                for msg in chat_history[-10:]:  # Last 10 messages for context
                    messages.append(msg)
            
            # Add current query with context
            messages.append({
                "role": "user",
                "content": f"""Context from disability science documents:
                
{context}

Question: {query}

Please answer based on the provided context."""
            })
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1500
            )
            
            return {
                'response': response.choices[0].message.content,
                'sources': [doc['source'] for doc in documents],
                'total_documents': len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': f"Error generating response: {str(e)}",
                'sources': [],
                'total_documents': 0
            }

    def chat(self, query: str, chat_history: List = None, top_k: int = 5) -> Dict[str, Any]:
        """Main chat function that combines search and generation"""
        # Search for relevant documents
        documents = self.search_documents(query, top_k)
        
        # Generate response
        result = self.generate_response(query, documents, chat_history)
        
        return result
