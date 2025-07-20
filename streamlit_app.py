import streamlit as st
from openai import OpenAI
import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict

# Import authentication module
from auth import initialize_authenticator, get_special_user_api_key

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
        'authenticated_user': False,
        'manual_api_key': '',
        'llm_model': 'gpt-4.1-nano',
        'user_mode': 'manual'  # 'manual' or 'authenticated'
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_current_api_key():
    """Get the appropriate API key based on user mode."""
    if st.session_state.user_mode == 'authenticated' and st.session_state.authenticated_user:
        return get_special_user_api_key()
    else:
        return st.session_state.get('manual_api_key', '')

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
    """Main application with dual access modes."""
    initialize_session_state()
    
    # Header
    st.title("♿ Disability Science Research Assistant")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Access Mode")
        
        # Mode selection
        access_mode = st.radio(
            "Choose access method:",
            ["Manual API Key", "Special User Login"],
            index=0 if st.session_state.user_mode == 'manual' else 1
        )
        
        # Update mode in session state
        if access_mode == "Manual API Key":
            st.session_state.user_mode = 'manual'
            st.session_state.authenticated_user = False
        else:
            st.session_state.user_mode = 'authenticated'
        
        st.markdown("---")
        
        # Handle different access modes
        if st.session_state.user_mode == 'manual':
            # Manual API Key Mode (Original Implementation)
            st.subheader("🔑 API Configuration")
            
            manual_api_key = st.text_input(
                "OpenAI API Key", 
                type="password", 
                value=st.session_state.manual_api_key,
                help="Enter your personal OpenAI API key"
            )
            
            if manual_api_key:
                st.session_state.manual_api_key = manual_api_key
                st.success("🟢 API Key Entered")
            else:
                st.warning("⚠️ API Key Required")
                
        else:
            # Special User Authentication Mode
            st.subheader("👤 User Login")
            
            # Initialize authenticator
            authenticator, special_username = initialize_authenticator()
            
            if authenticator:
                try:
                    if not st.session_state.authenticated_user:
                        # Show login form
                        name, authentication_status, username = authenticator.login('Login', 'sidebar')
                        
                        if authentication_status == True:
                            st.session_state.authenticated_user = True
                            st.session_state.special_user_name = name
                            st.session_state.special_username = username
                            st.success(f"✅ Welcome {name}!")
                            st.rerun()
                            
                        elif authentication_status == False:
                            st.error('❌ Incorrect credentials')
                            
                        elif authentication_status == None:
                            st.info('👆 Please login')
                    else:
                        # Show logged in user info
                        st.success(f"🟢 Logged in as: {st.session_state.special_user_name}")
                        
                        if st.button("🚪 Logout"):
                            authenticator.logout('Logout', 'sidebar')
                            st.session_state.authenticated_user = False
                            if 'special_user_name' in st.session_state:
                                del st.session_state['special_user_name']
                            if 'special_username' in st.session_state:
                                del st.session_state['special_username']
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"Authentication error: {e}")
            else:
                st.error("Authentication system unavailable")
        
        st.markdown("---")
        
        # Model selection (available for both modes)
        st.subheader("🤖 Model Settings")
        model_options = ["gpt-4.1-nano", "o4-mini", "o4-mini-deep-research", "gpt-4o-mini-search-preview"]
        current_model = st.session_state.llm_model
        default_index = model_options.index(current_model) if current_model in model_options else 0
        
        selected_model = st.selectbox("LLM Model", model_options, index=default_index)
        st.session_state.llm_model = selected_model
        
        # Show current API key status
        st.markdown("---")
        st.subheader("🔍 Status")
        
        current_api_key = get_current_api_key()
        if current_api_key:
            if st.session_state.user_mode == 'authenticated':
                st.success("🔑 Using Special User API Key")
            else:
                st.success("🔑 Manual API Key Active")
        else:
            st.error("🔑 No API Key Available")
        
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
        # Show current mode
        if st.session_state.user_mode == 'authenticated' and st.session_state.authenticated_user:
            st.markdown(f"**Mode:** Special User ({st.session_state.special_user_name})")
        else:
            st.markdown("**Mode:** Manual API Key")
        
        st.header("💬 Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(message, message.get('role') == 'user')
        
        # Chat input
        current_api_key = get_current_api_key()
        
        if current_api_key:
            if prompt := st.chat_input("Ask about disability research..."):
                # Add user message
                user_message = {"role": "user", "content": prompt}
                st.session_state.chat_history.append(user_message)
                display_chat_message(user_message, is_user=True)
                
                # Generate response
                with st.spinner("🤔 Thinking..."):
                    response = generate_enhanced_response(
                        prompt, 
                        st.session_state.llm_model,
                        current_api_key
                    )
                    
                    assistant_message = {
                        "role": "assistant",
                        "content": response["content"],
                        "sources": response.get("sources", [])
                    }
                    st.session_state.chat_history.append(assistant_message)
                    display_chat_message(assistant_message, is_user=False)
                
                st.rerun()
        else:
            st.info("💡 Please provide an API key or login as special user to start chatting.")
            
        # Quick buttons (only show if API key is available)
        if current_api_key:
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

if __name__ == "__main__":
    main()