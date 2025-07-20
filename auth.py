import streamlit as st
import streamlit_authenticator as stauth
import bcrypt

def initialize_authenticator():
    """Initialize the authenticator with enhanced error handling."""
    
    try:
        # Debug: Check if secrets are accessible
        st.write("🔍 Checking secrets access...")
        
        # Load credentials with explicit error checking
        if "special_user" not in st.secrets:
            st.error("❌ 'special_user' section not found in secrets")
            return None, None
            
        if "auth" not in st.secrets:
            st.error("❌ 'auth' section not found in secrets")
            return None, None
        
        username = st.secrets["special_user"]["username"]
        name = st.secrets["special_user"]["name"]
        password = st.secrets["special_user"]["password"]
        
        st.write(f"✅ Loaded user: {username}")
        
        # Create credentials dictionary
        credentials = {
            "usernames": {
                username: {
                    "name": name,
                    "password": password
                }
            }
        }
        
        st.write("✅ Created credentials dictionary")
        
        # Try creating authenticator
        authenticator = stauth.Authenticate(
            credentials,
            st.secrets["auth"]["cookie_name"],
            st.secrets["auth"]["cookie_key"],
            cookie_expiry_days=st.secrets["auth"]["cookie_expiry_days"]
        )
        
        st.write("✅ Authenticator created successfully")
        return authenticator, username
        
    except KeyError as e:
        st.error(f"❌ Missing key in secrets: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Authentication setup error: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        return None, None

def get_special_user_api_key():
    """Get the special user's API key from secrets."""
    try:
        return st.secrets["openai"]["special_user_api_key"]
    except Exception as e:
        st.error(f"Error loading special user API key: {e}")
        return None

def hash_password(plain_password):
    """Helper function to hash a password for initial setup."""
    hashed = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())
    return hashed.decode('utf-8')
