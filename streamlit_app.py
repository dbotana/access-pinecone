import streamlit as st
from openai import OpenAI
import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict
import hmac

# Import authentication module
from auth import check_password, get_api_key

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Disability Science Research Assistant",
    page_icon="♿",
    layout="wide"
)

# Your existing configurations
DISABILITY_DATASETS = {
    "CDC Disability Datasets": "https://www.cdc.gov/dhds/datasets/index.html",
    "Bureau of Labor Statistics": "https://data.bls.gov/dataQuery/find?removeAll=1",
    "Census Disability Characteristics": "https://data.census.gov/table/ACSST1Y2022.S1810",
    "SSA Disability Data": "https://www.ssa.gov/disability/data/SSA-SA-MOWL.csv"
}

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
        'llm_model': 'gpt-4.1-nano',
        # Removed all authentication-related session state variables
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


import bcrypt
import streamlit as st

def check_password():
    """Returns True if the user entered the correct password."""
    
    def password_entered():
        """Check if entered password matches the bcrypt hash in secrets."""
        entered_password = st.session_state["password"]
        stored_hash = st.secrets["app_password"]
        
        # Verify bcrypt hash
        if bcrypt.checkpw(entered_password.encode('utf-8'), stored_hash.encode('utf-8')):
            st.session_state["password_correct"] = True
            # Clear password from session for security
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Return True if password already validated
    if st.session_state.get("password_correct", False):
        return True

    # Show password input
    st.text_input(
        "🔐 Enter Application Password", 
        type="password", 
        on_change=password_entered, 
        key="password",
        help="Enter the password to access the Disability Science Research Assistant"
    )
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("🚫 Incorrect password - please try again")
    
    return False


def get_api_key():
    """Get the OpenAI API key from secrets."""
    return st.secrets["openai_api_key"]

def display_chat_message(message: Dict, is_user: bool = True):
    """Display a chat message with proper styling."""
    role = "user" if is_user else "assistant"
    
    with st.chat_message(role):
        st.markdown(message.get('content', ''))
        
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
        
        df = pd.read_csv(BytesIO(response.content))
        
        os.makedirs("data", exist_ok=True)
        df.to_excel(f"data/{filename}", index=False)
        
        return df
        
    except Exception as e:
        st.error(f"Error downloading {filename}: {str(e)}")
        logger.error(f"Download error: {str(e)}")
        return pd.DataFrame()

def generate_enhanced_response(prompt: str, model: str, api_key: str) -> dict:
    """Generate AI response using the current API key."""
    try:
        if not api_key:
            return {
                "content": "Please provide an OpenAI API key to enable AI responses.",
                "sources": []
            }
        
        client = OpenAI(api_key=api_key)
        
        # Build context
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
            model=model,
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
            "content": f"Error generating response: {str(e)}. Please check your API key and model selection.",
            "sources": []
        }

def main():
    """Main application with simplified password-based authentication."""
    initialize_session_state()
    
    # Header
    st.title("♿ Disability Science Research Assistant")
    
    # Password protection - check first
    if not check_password():
        st.info("🔐 Please enter the password to access the application.")
        st.stop()
    
    # User is authenticated - show main application
    with st.sidebar:
        st.success("✅ Access Granted")
        
        # Logout button
        if st.button("🚪 Logout"):
            if "password_correct" in st.session_state:
                del st.session_state["password_correct"]
            st.rerun()
        
        st.markdown("---")
        
        # Model selection
        st.subheader("🤖 Model Settings")
        model_options = ["gpt-4.1-nano", "o4-mini", "o4-mini-deep-research", "gpt-4o-mini-search-preview"]
        current_model = st.session_state.llm_model
        default_index = model_options.index(current_model) if current_model in model_options else 0
        
        selected_model = st.selectbox("LLM Model", model_options, index=default_index)
        st.session_state.llm_model = selected_model
        
        # API Key status
        st.markdown("---")
        st.subheader("🔍 Status")
        st.success("🔑 API Key Active (from secrets)")
        
        # Dataset management
        st.markdown("---")
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

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(message, message.get('role') == 'user')
        
        # Chat input - API key automatically retrieved from secrets
        if prompt := st.chat_input("Ask about disability research..."):
            # Add user message
            user_message = {"role": "user", "content": prompt}
            st.session_state.chat_history.append(user_message)
            display_chat_message(user_message, is_user=True)
            
            # Generate response using API key from secrets
            with st.spinner("🤔 Thinking..."):
                api_key = get_api_key()  # Simple function to get API key from secrets
                response = generate_enhanced_response(
                    prompt, 
                    st.session_state.llm_model,
                    api_key
                )
                
                assistant_message = {
                    "role": "assistant",
                    "content": response["content"],
                    "sources": response.get("sources", [])
                }
                st.session_state.chat_history.append(assistant_message)
                display_chat_message(assistant_message, is_user=False)
            
            st.rerun()
            
        # Quick action buttons
        st.subheader("🎯 Quick Questions")
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("📊 Employment Stats"):
                prompt = "What are the latest disability employment statistics?"
                user_message = {"role": "user", "content": prompt}
                st.session_state.chat_history.append(user_message)
                st.rerun()
        
        with col_b:
            if st.button("📋 Data Sources"):
                prompt = "What disability datasets are available?"
                user_message = {"role": "user", "content": prompt}
                st.session_state.chat_history.append(user_message)
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
                    
                    # Download option
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

if __name__ == "__main__":
    main()