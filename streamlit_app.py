import streamlit as st
from openai import OpenAI
import os
from typing import Dict, Set

# Page configuration
st.set_page_config(
    page_title="Disability Science Research Assistant",
    page_icon="♿",
    layout="wide"
)

# Default source configurations
DISABILITY_DATASETS = {
    "CDC Disability Datasets": "https://www.cdc.gov/dhds/datasets/index.html",
    "IncluSet": "https://incluset.com/",
    "Disability Statistics": "https://www.disabilitystatistics.org",
    "Research on Disability Compendium": "https://www.researchondisability.org/sites/default/files/media/2025-03/pdf-online_full-compendium-with-title-acknowledgement-pages.pdf",
    "Bureau of Labor Statistics": "https://data.bls.gov/dataQuery/find?removeAll=1",
    "Census Disability Characteristics": "https://data.census.gov/table/ACSST1Y2022.S1810?q=S1810:+DISABILITY+CHARACTERISTICS&g=010XX00US&moe=false",
    "NCES Postsecondary Students": "https://nces.ed.gov/datalab/powerstats/71-beginning-postsecondary-students-2012-2017/percentage-distribution",
    "NCES B&B Survey": "https://nces.ed.gov/surveys/b&b/",
    "College Students with Disabilities Stats": "https://nccsd.ici.umn.edu/clearinghouse/audience-specific-resources/researchers--policy-makers/stats-college-stds-with-disabilities",
    "NIH Researchers with Disabilities": "https://grants.nih.gov/new-to-nih/information-for/researchers/researchers-with-disabilities",
    "SSA Disability Data": "https://www.ssa.gov/disability/data/SSA-SA-MOWL.csv"
}

RESEARCH_SOURCES = {
    "PubMed": "https://pubmed.ncbi.nlm.nih.gov/",
    "arXiv": "https://arxiv.org/",
    "SERU Consortium Library": "https://www.zotero.org/groups/4116971/seru_publications/library"
}

def get_default_sources_context() -> str:
    """Generate context text with default sources for the AI assistant."""
    context = "\n\nYou have access to these authoritative disability research sources:\n\n"
    
    context += "**Government Datasets:**\n"
    for name, url in DISABILITY_DATASETS.items():
        if any(domain in url for domain in ['cdc.gov', 'bls.gov', 'census.gov', 'nces.ed.gov', 'nih.gov', 'ssa.gov']):
            context += f"- {name}: {url}\n"
    
    context += "\n**Research Organizations:**\n"
    for name, url in DISABILITY_DATASETS.items():
        if not any(domain in url for domain in ['cdc.gov', 'bls.gov', 'census.gov', 'nces.ed.gov', 'nih.gov', 'ssa.gov']):
            context += f"- {name}: {url}\n"
    
    context += "\n**Academic Databases:**\n"
    for name, url in RESEARCH_SOURCES.items():
        context += f"- {name}: {url}\n"
    
    context += "\nWhen answering questions, reference these sources when relevant and suggest specific datasets or resources that would be helpful for the user's research needs."
    
    return context

def display_sources_sidebar():
    """Display available sources in the sidebar."""
    with st.sidebar:
        st.header("📊 Available Data Sources")
        
        with st.expander("Government Datasets", expanded=False):
            for name, url in DISABILITY_DATASETS.items():
                if any(domain in url for domain in ['cdc.gov', 'bls.gov', 'census.gov', 'nces.ed.gov', 'nih.gov', 'ssa.gov']):
                    st.markdown(f"**[{name}]({url})**")
        
        with st.expander("Research Organizations", expanded=False):
            for name, url in DISABILITY_DATASETS.items():
                if not any(domain in url for domain in ['cdc.gov', 'bls.gov', 'census.gov', 'nces.ed.gov', 'nih.gov', 'ssa.gov']):
                    st.markdown(f"**[{name}]({url})**")
        
        with st.expander("Academic Databases", expanded=False):
            for name, url in RESEARCH_SOURCES.items():
                st.markdown(f"**[{name}]({url})**")

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def display_chat_history():
    """Display the chat history."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display referenced sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander(f"📚 Referenced Sources ({len(message['sources'])})", expanded=False):
                        for source in message["sources"]:
                            st.markdown(f"• **[{source['name']}]({source['url']})**")

def extract_referenced_sources(response_text: str) -> list:
    """Extract sources that were likely referenced in the response."""
    referenced_sources = []
    response_lower = response_text.lower()
    
    # Check disability datasets
    for name, url in DISABILITY_DATASETS.items():
        if any(keyword in response_lower for keyword in name.lower().split()):
            referenced_sources.append({"name": name, "url": url})
    
    # Check research sources
    for name, url in RESEARCH_SOURCES.items():
        if name.lower() in response_lower:
            referenced_sources.append({"name": name, "url": url})
    
    return referenced_sources

# Main app
def main():
    initialize_session_state()
    
    # Header
    st.title("♿ Disability Science Research Assistant")
    st.write(
        "Ask questions about disability research, policy, and data. The assistant has access to "
        "authoritative government datasets and research sources to provide evidence-based answers."
    )
    
    # Display sources sidebar
    display_sources_sidebar()
    
    # API key input
    with st.sidebar:
        st.header("🔑 Configuration")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.", icon="🗝️")
            return
    
    # Create OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask about disability research, data, or policy..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            # Create system message with source context
            system_message = {
                "role": "system",
                "content": f"""You are a disability science research assistant with expertise in disability policy, data analysis, and research methodologies. 

Provide evidence-based, accessible answers about disability research, policy, and data. When relevant, reference specific datasets, research sources, or government resources that would be helpful.

{get_default_sources_context()}

Guidelines:
- Be accurate and cite authoritative sources
- Suggest specific datasets when relevant
- Use accessible language
- Provide actionable research guidance
- Reference the most appropriate sources for each query"""
            }
            
            messages = [
                system_message,
                {"role": "user", "content": prompt}
            ]
            
            # Add recent chat history for context
            if len(st.session_state.chat_history) > 1:
                recent_messages = st.session_state.chat_history[-4:]  # Last 2 exchanges
                for msg in recent_messages[:-1]:  # Exclude the current user message
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Analyzing question and consulting research sources..."):
                try:
                    # Generate response
                    stream = client.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        stream=True,
                        temperature=0.3
                    )
                    
                    # Stream and collect response
                    response_text = st.write_stream(stream)
                    
                    # Extract referenced sources
                    referenced_sources = extract_referenced_sources(response_text)
                    
                    # Add assistant message to chat history with sources
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response_text,
                        "sources": referenced_sources
                    })
                    
                    # Display referenced sources
                    if referenced_sources:
                        with st.expander(f"📚 Referenced Sources ({len(referenced_sources)})", expanded=False):
                            for source in referenced_sources:
                                st.markdown(f"• **[{source['name']}]({source['url']})**")
                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

    # Additional information in sidebar
    with st.sidebar:
        st.header("💡 Example Questions")
        st.write("• What data sources track disability employment rates?")
        st.write("• Where can I find college disability statistics?")
        st.write("• What are the CDC's disability survey questions?")
        st.write("• How do I access disability research data?")
        
        st.header("🎯 Features")
        st.write("✅ Evidence-based responses")
        st.write("✅ Authoritative data sources")
        st.write("✅ Research methodology guidance")
        st.write("✅ Policy and data analysis")

if __name__ == "__main__":
    main()
