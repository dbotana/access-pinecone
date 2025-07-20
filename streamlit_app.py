import streamlit as st
from openai import OpenAI
import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Disability Science Research Assistant",
    page_icon="♿",
    layout="wide"
)

# Default source configurations
DISABILITY_DATASETS = {
    "CDC Disability Datasets": "https://www.cdc.gov/dhds/datasets/index.html",
    "Bureau of Labor Statistics": "https://data.bls.gov/dataQuery/find?removeAll=1",
    "Census Disability Characteristics": "https://data.census.gov/table/ACSST1Y2022.S1810",
    "SSA Disability Data": "https://www.ssa.gov/disability/data/SSA-SA-MOWL.csv"
}

RESEARCH_SOURCES = {
    "PubMed": "https://pubmed.ncbi.nlm.nih.gov/",
    "arXiv": "https://arxiv.org/"
}

# Simplified dataset configuration for downloads
DISABILITY_DATASETS_EXCEL = {
    "SSA Disability Data": {
        "url": "https://www.ssa.gov/disability/data/SSA-SA-MOWL.csv",
        "file_type": "csv"
    }
}

def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        'chat_history': [],
        'datasets': {},
        'system_initialized': False,
        'rag_system': None,
        'openai_api_key': '',
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'embedding_model': 'text-embedding-3-small',
        'llm_model': 'gpt-4',
        'vector_store_type': 'faiss'
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def setup_rag_system():
    """Simple RAG system initialization."""
    try:
        if not st.session_state.get('openai_api_key'):
            st.error("Please enter your OpenAI API key first!")
            return False
        
        # Simple initialization - replace with your actual RAG system
        st.session_state.rag_system = True
        st.session_state.system_initialized = True
        st.success("✅ RAG system initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        logger.error(f"RAG initialization error: {str(e)}")
        return False

def display_chat_message(message: Dict, is_user: bool = True):
    """Display a chat message with proper styling."""
    role = "user" if is_user else "assistant"
    
    with st.chat_message(role):
        st.markdown(message.get('content', ''))
        
        # Display sources if available and it's an assistant message
        if not is_user and 'sources' in message and message['sources']:
            with st.expander(f"📚 Referenced Sources ({len(message['sources'])})", expanded=False):
                for source in message['sources']:
                    name = source.get('name', 'Unknown')
                    url = source.get('url', '#')
                    st.markdown(f"• **[{name}]({url})**")

@st.cache_data(ttl=3600)
def download_excel_file(url: str, filename: str) -> pd.DataFrame:
    """Download CSV file from URL and return as DataFrame."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Only handle CSV for simplicity
        df = pd.read_csv(BytesIO(response.content))
        
        # Save locally for backup
        os.makedirs("data", exist_ok=True)
        df.to_excel(f"data/{filename}", index=False)
        
        return df
        
    except Exception as e:
        st.error(f"Error downloading {filename}: {str(e)}")
        logger.error(f"Download error: {str(e)}")
        return pd.DataFrame()

def get_openai_api_key():
    """Get OpenAI API key from session state or environment."""
    return st.session_state.get('openai_api_key') or os.getenv("OPENAI_API_KEY")

def generate_enhanced_response(prompt: str) -> dict:
    """Generate AI response with context."""
    try:
        api_key = get_openai_api_key()
        if not api_key:
            return {
                "content": "Please provide an OpenAI API key to enable AI responses.",
                "sources": []
            }
        
        client = OpenAI(api_key=api_key)
        
        # Build context from available datasets
        context_parts = ["You are a disability science research assistant."]
        
        if st.session_state.datasets:
            context_parts.append("\nAvailable datasets:")
            for name, info in st.session_state.datasets.items():
                df = info['data']
                context_parts.append(f"- {name}: {len(df)} records")
        
        system_message = "\n".join(context_parts)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model=st.session_state.get('llm_model', 'gpt-4'),
            messages=messages,
            temperature=0.3,
            max_tokens=800
        )
        
        # Extract referenced sources
        sources = []
        response_text = response.choices[0].message.content.lower()
        
        for name, url in DISABILITY_DATASETS.items():
            if any(keyword in response_text for keyword in name.lower().split()):
                sources.append({"name": name, "url": url})
        
        return {
            "content": response.choices[0].message.content,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "content": f"I encountered an error: {str(e)}. Please check your API key and try again.",
            "sources": []
        }

def main():
    """Simplified main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("♿ Disability Science Research Assistant")
    st.markdown("**AI-powered chatbot for disability research and data analysis**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key
        api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key)
        if api_key:
            st.session_state.openai_api_key = api_key
        
        # Simple RAG settings
        st.subheader("Settings")
        llm_model = st.selectbox("LLM Model", ["gpt-4", "gpt-3.5-turbo"], 
                                index=0 if st.session_state.llm_model == "gpt-4" else 1)
        st.session_state.llm_model = llm_model
        
        # RAG System
        if st.button("🚀 Initialize System"):
            setup_rag_system()
        
        # Dataset Management
        st.subheader("📊 Data Management")
        
        if st.button("📥 Load Sample Data"):
            with st.spinner("Loading disability data..."):
                for name, config in DISABILITY_DATASETS_EXCEL.items():
                    filename = f"{name.replace(' ', '_').lower()}.xlsx"
                    df = download_excel_file(config["url"], filename)
                    if not df.empty:
                        st.session_state.datasets[name] = {
                            "data": df,
                            "last_updated": datetime.now()
                        }
                st.success("✅ Sample data loaded!")
                st.rerun()
        
        # Show dataset status
        if st.session_state.datasets:
            st.write("**Loaded Datasets:**")
            for name, info in st.session_state.datasets.items():
                df = info['data']
                st.write(f"• {name}: {len(df)} records")
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(message, message.get('role') == 'user')
        
        # Chat input
        if prompt := st.chat_input("Ask about disability research..."):
            # Add user message
            user_message = {"role": "user", "content": prompt}
            st.session_state.chat_history.append(user_message)
            display_chat_message(user_message, is_user=True)
            
            # Generate response
            with st.spinner("🤔 Thinking..."):
                response = generate_enhanced_response(prompt)
                
                assistant_message = {
                    "role": "assistant",
                    "content": response["content"],
                    "sources": response.get("sources", [])
                }
                st.session_state.chat_history.append(assistant_message)
                display_chat_message(assistant_message, is_user=False)
            
            st.rerun()
        
        # Quick buttons
        st.subheader("🎯 Quick Questions")
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("📊 Employment Stats"):
                prompt = "What are the latest disability employment statistics?"
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.rerun()
        
        with col_b:
            if st.button("📋 Data Sources"):
                prompt = "What disability datasets are available?"
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.rerun()
    
    with col2:
        st.header("📈 Data Overview")
        
        if st.session_state.datasets:
            for name, info in st.session_state.datasets.items():
                df = info['data']
                
                with st.expander(f"📋 {name}"):
                    st.write(f"**Records:** {len(df):,}")
                    st.write(f"**Columns:** {len(df.columns)}")
                    
                    if st.checkbox(f"Preview {name}", key=f"preview_{name}"):
                        st.dataframe(df.head())
                    
                    # Simple download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        f"📥 Download {name}",
                        csv,
                        file_name=f"{name.replace(' ', '_').lower()}.csv",
                        mime="text/csv",
                        key=f"download_{name}"
                    )
        else:
            st.info("Click 'Load Sample Data' to get started with disability datasets.")
            
            st.subheader("📚 Available Sources")
            for name, url in DISABILITY_DATASETS.items():
                st.markdown(f"• **[{name}]({url})**")
    
    # Footer
    st.markdown("---")
    
    # System status
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        if st.session_state.system_initialized:
            st.success("🟢 System Ready")
        else:
            st.warning("🟡 System Not Initialized")
    
    with status_col2:
        if st.session_state.datasets:
            st.success(f"🟢 {len(st.session_state.datasets)} Datasets Loaded")
        else:
            st.info("🔵 No Data Loaded")

if __name__ == "__main__":
    main()