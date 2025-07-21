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
    page_title="ACCESS: AI-Curated Comprehensive Evidence Search System",
    page_icon="♿",
    layout="wide"
)

# Your existing configurations
DISABILITY_DATASETS = {
    "CDC Disability Datasets": "https://www.cdc.gov/dhds/datasets/index.html",
    "Bureau of Labor Statistics": "https://data.bls.gov/dataQuery/find?removeAll=1",
    "Bureau of Labor Statistics 2025 Labor Force Characteristics Summary": "https://www.bls.gov/news.release/pdf/disabl.pdf",
    "National Center for College Students with Disabilities": "https://nccsd.ici.umn.edu/clearinghouse/audience-specific-resources/researchers--policy-makers/stats-college-stds-with-disabilities",
    "NIH Resources for Researchers with Disabilities": "https://grants.nih.gov/new-to-nih/information-for/researchers/researchers-with-disabilities",
    "UNH Center for Research on Disability ADSC Compendium": "https://universitysystemnh.sharepoint.com/:u:/r/teams/IODPublicLinks/Shared%20Documents/Research%20on%20Disability%20Website%20Public%20Links/Compendium/2025%20ADSC%20HTML%20Docs/HTML-Full-Compendium.html?csf=1&web=1&e=qBfYO9",
    "UNH Institute on Disability Annual Report on People with Disabilities in America: 2025": "https://universitysystemnh.sharepoint.com/:u:/r/teams/IODPublicLinks/Shared%20Documents/Research%20on%20Disability%20Website%20Public%20Links/Compendium/2025%20ADSC%20HTML%20Docs/HTML-Full-Compendium.html?csf=1&web=1&e=qBfYO9"
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

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_dataset_texts():
    """Fetch text content from all dataset URLs."""
    texts = {}
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for name, url in DISABILITY_DATASETS.items():
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            if url.endswith('.csv'):
                texts[name] = response.text[:5000]  # First 5k chars
            else:
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                text_content = ' '.join(p.get_text(strip=True) for p in paragraphs)
                texts[name] = text_content[:5000]  # First 5k chars
                
        except Exception as e:
            texts[name] = f"Error fetching content: {e}"
    
    return texts

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = sum(a*a for a in vec1) ** 0.5
    norm2 = sum(b*b for b in vec2) ** 0.5
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def get_relevant_sources(prompt: str, api_key: str, top_k: int = 2):
    """Get most relevant dataset sources for a given prompt."""
    texts = fetch_dataset_texts()
    client = OpenAI(api_key=api_key)
    
    # Get prompt embedding
    prompt_response = client.embeddings.create(
        model="text-embedding-3-small", 
        input=prompt
    )
    prompt_embedding = prompt_response.data[0].embedding
    
    # Calculate similarities
    similarities = []
    for name, text in texts.items():
        if not text.startswith("Error"):
            text_response = client.embeddings.create(
                model="text-embedding-3-small", 
                input=text
            )
            text_embedding = text_response.data[0].embedding
            similarity = cosine_similarity(prompt_embedding, text_embedding)
            similarities.append((name, similarity))
    
    # Return top k most similar
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

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
    """Display a chat message with enhanced source information."""
    role = "user" if is_user else "assistant"
    
    with st.chat_message(role):
        st.markdown(message.get('content', ''))
        
        if not is_user and 'sources' in message and message['sources']:
            with st.expander(f"📚 Smart Sources ({len(message['sources'])})", expanded=False):
                for source in message['sources']:
                    name = source.get('name', 'Unknown')
                    url = source.get('url', '#')
                    relevance = source.get('relevance', 'N/A')
                    st.markdown(f"• **[{name}]({url})** (relevance: {relevance})")
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
    """Generate AI response with model-specific parameter handling & endpoint selection."""
    try:
        if not api_key:
            return {
                "content": "Please provide an OpenAI API key to enable AI responses.",
                "sources": []
            }
        
        client = OpenAI(api_key=api_key)
        
        # Get relevant sources
        relevant_sources = get_relevant_sources(prompt, api_key, top_k=2)
        
        # Build context
        context = "You are a disability science research assistant."
        
        if relevant_sources:
            context += f"\n\nMost relevant sources for this query:"
            for name, similarity in relevant_sources:
                context += f"\n- {name} (relevance: {similarity:.2f})"
        
        # Handle model-specific requirements
        if model == "gpt-4o-mini-search-preview":
            # No temperature parameter, uses chat completions
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=800
            )
            content = response.choices[0].message.content
            
        elif model == "o4-mini":
            # Uses max_completion_tokens instead of max_tokens
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=800
            )
            content = response.choices[0].message.content
            
        elif model == "o4-mini-deep-research":
            # Uses v1/responses endpoint instead of chat completions
            # Combine context and prompt for responses endpoint
            full_prompt = f"{context}\n\nUser Query: {prompt}"
            
            try:
                # Try the responses endpoint
                response = client.responses.create(
                    model=model,
                    prompt=full_prompt,
                    temperature=0.3,
                    max_tokens=800
                )
                content = response.choices[0].text
            except AttributeError:
                # If responses endpoint doesn't exist in client, use completions
                response = client.completions.create(
                    model=model,
                    prompt=full_prompt,
                    temperature=0.3,
                    max_tokens=800
                )
                content = response.choices[0].text
                
        else:
            # Default case: gpt-4.1-nano and other standard models
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )
            content = response.choices[0].message.content
        
        # Create sources list
        sources = []
        for name, similarity in relevant_sources:
            if name in DISABILITY_DATASETS:
                sources.append({
                    "name": name, 
                    "url": DISABILITY_DATASETS[name],
                    "relevance": f"{similarity:.2f}"
                })
        
        return {
            "content": content,
            "sources": sources
        }
        
    except Exception as e:
        return {
            "content": f"Error generating response: {str(e)}",
            "sources": []
        }
        
def get_model_config(model: str) -> dict:
    """Get configuration details for each model."""
    model_configs = {
        "gpt-4.1-nano": {
            "supports_temperature": True,
            "token_parameter": "max_tokens",
            "endpoint": "chat/completions",
            "description": "✅ Full chat features"
        },
        "gpt-4o-mini-search-preview": {
            "supports_temperature": False,
            "token_parameter": "max_tokens", 
            "endpoint": "chat/completions",
            "description": "🔍 Search-optimized (no temperature)"
        },
        "o4-mini": {
            "supports_temperature": True,
            "token_parameter": "max_completion_tokens",
            "endpoint": "chat/completions", 
            "description": "⚡ Uses max_completion_tokens"
        },
        "o4-mini-deep-research": {
            "supports_temperature": True,
            "token_parameter": "max_tokens",
            "endpoint": "responses",
            "description": "🔬 Research model (responses API)"
        }
    }
    
    return model_configs.get(model, {
        "supports_temperature": True,
        "token_parameter": "max_tokens",
        "endpoint": "chat/completions",
        "description": "❓ Unknown model"
    })

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
        st.subheader("🤖 Model Settings")

        # Model options with compatibility info
        model_options = ["gpt-4.1-nano", "o4-mini", "o4-mini-deep-research", "gpt-4o-mini-search-preview"]

        # Create selection with descriptions
        current_model = st.session_state.llm_model
        default_index = model_options.index(current_model) if current_model in model_options else 0

        selected_model = st.selectbox(
            "LLM Model", 
            model_options, 
            index=default_index,
            format_func=lambda x: f"{x}"
        )

        # Show model info
        if selected_model:
            config = get_model_config(selected_model)
            st.caption(config["description"])
            
            with st.expander("Model Details"):
                st.write(f"**Endpoint:** {config['endpoint']}")
                st.write(f"**Temperature:** {'✅ Supported' if config['supports_temperature'] else '❌ Not supported'}")
                st.write(f"**Token Parameter:** {config['token_parameter']}")

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
        # Simple status check:
        st.subheader("🧠 Smart Source Analysis")

        if st.button("🔄 Test Source Retrieval"):
            with st.spinner("Testing source analysis..."):
                try:
                    texts = fetch_dataset_texts()
                    successful = sum(1 for text in texts.values() if not text.startswith("Error"))
                    st.success(f"✅ Successfully analyzed {successful}/{len(texts)} sources")
                    
                    with st.expander("Source Status"):
                        for name, text in texts.items():
                            if text.startswith("Error"):
                                st.error(f"❌ {name}: Failed to fetch")
                            else:
                                st.success(f"✅ {name}: {len(text)} characters")
                except Exception as e:
                    st.error(f"❌ Error: {e}")


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