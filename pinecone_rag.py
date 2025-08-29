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
                 llm_model: str = "gpt-5-nano"):
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        try:
            self.index = self.pc.Index(pinecone_index_name)
            logger.info(f"Successfully connected to index: {pinecone_index_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index: {e}")
            raise        
        # Initialize Pinecone without environment
        self.pc = Pinecone(
            api_key=pinecone_api_key,
        )
        self.index = self.pc.Index(pinecone_index_name)
        
        logger.info(f"Initialized PineconeRAG with index: {pinecone_index_name}")

    def get_model_config(self, model: str) -> dict:
        """Get configuration details for each model - matching streamlit_app.py"""
        model_configs = {
            "gpt-5-nano": {
                "supports_temperature": True,
                "token_parameter": "max_completion_tokens",
                "endpoint": "chat/completions",
                "description": "✅ Full chat features"
            },
            "gpt-4o-mini-search-preview": {
                "supports_temperature": False,
                "token_parameter": "max_tokens",
                "endpoint": "chat/completions",
                "description": "🔍 Search the internet for additional sources"
            },
            "o4-mini": {
                "supports_temperature": True,
                "token_parameter": "max_completion_tokens",
                "endpoint": "chat/completions",
                "description": "⚡ Reasoning model for complex questions"
            },
            "o4-mini-deep-research": {
                "supports_temperature": True,
                "token_parameter": "max_tokens",
                "endpoint": "responses",
                "description": "🔬 Research model for generating long text outputs"
            }
        }
        
        return model_configs.get(model, {
            "supports_temperature": True,
            "token_parameter": "max_completion_tokens",
            "endpoint": "chat/completions",
            "description": "❓ Unknown model"
        })

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for query text"""
        text = text.replace("\n", " ")
        index_stats = self.index.describe_index_stats()
        index_dims = index_stats.dimension
        
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model,
            dimensions=index_dims
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
                include_metadata=True,
                namespace="__default__",
                filter={}
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
                        'source_type': 'PDF',
                        'score': match['score']  # Include score in source for easy access
                    }
                })
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def generate_response(self, query: str, documents: List[Dict], chat_history: List = None) -> Dict[str, Any]:
        """Generate response using retrieved documents with model-specific handling"""
        try:
            # Prepare context from retrieved documents
            context = "\n\n".join([
                f"Document {i+1} (Score: {doc['score']:.3f}):\n{doc['content']}"
                for i, doc in enumerate(documents)
            ])
            
            # Get model configuration
            config = self.get_model_config(self.llm_model)
            
            # Prepare system message
            system_content = """You are a helpful assistant specializing in disability science research.
Use the provided context to answer questions about disability data, research, employment,
education, and related topics. If the information isn't in the context, say so clearly."""
            
            # Prepare messages
            messages = [{"role": "system", "content": system_content}]
            
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
            
            # Handle different model types based on endpoint
            if config["endpoint"] == "responses":
                # For models that use the responses endpoint
                full_prompt = f"{system_content}\n\nContext: {context}\n\nQuestion: {query}"
                try:
                    request_params = {
                        "model": self.llm_model,
                        "prompt": full_prompt,
                        config["token_parameter"]: 1500
                    }
                    
                    if config["supports_temperature"]:
                        request_params["temperature"] = 0.1
                    
                    response = self.openai_client.responses.create(**request_params)
                    response_content = response.choices[0].text
                    
                except AttributeError:
                    # Fallback to completions if responses endpoint doesn't exist
                    request_params = {
                        "model": self.llm_model,
                        "prompt": full_prompt,
                        config["token_parameter"]: 1500
                    }
                    
                    if config["supports_temperature"]:
                        request_params["temperature"] = 0.1
                    
                    response = self.openai_client.completions.create(**request_params)
                    response_content = response.choices[0].text
                    
            else:
                # For chat completion models
                request_params = {
                    "model": self.llm_model,
                    "messages": messages,
                    config["token_parameter"]: 1500
                }
                
                if config["supports_temperature"]:
                    request_params["temperature"] = 0.1
                
                response = self.openai_client.chat.completions.create(**request_params)
                response_content = response.choices[0].message.content

            return {
                'response': response_content,
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
