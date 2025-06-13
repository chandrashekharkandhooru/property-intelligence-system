import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Property Intelligence System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏠 Property Intelligence System")
    st.subheader("AI/ML Multimodal Property Intelligence - Combining Appraisal Analysis, Live Market Trends, and Conversational AI")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a module:", [
        "🏠 Home",
        "📄 OCR & Data Extraction", 
        "📰 Market Intelligence",
        "🤖 ML Predictions",
        "💬 Q&A System"
    ])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📄 OCR & Data Extraction":
        show_ocr_page()
    elif page == "📰 Market Intelligence":
        show_news_page()
    elif page == "🤖 ML Predictions":
        show_ml_page()
    elif page == "💬 Q&A System":
        show_qa_page()

def show_home_page():
    st.markdown("""
    ## Welcome to the Property Intelligence System! 🚀
    
    This system combines multiple AI technologies to provide comprehensive property analysis:
    
    ### 🎯 Key Features:
    - **📄 OCR Processing**: Extract structured data from appraisal PDFs
    - **📰 Market Intelligence**: Live news analysis for location-based insights  
    - **🤖 ML Predictions**: Property valuation and risk scoring
    - **💬 Q&A System**: Conversational interface using RAG
    
    ### 📊 Sample Data Available:
    - **Property**: Hinsdale Middle School Complex
    - **Location**: Hinsdale, Illinois  
    - **Valuation**: $7,000,000
    - **Size**: 107,500 sq ft building, 9.51 acres
    
    **👈 Use the sidebar to navigate between modules**
    """)

def show_ocr_page():
    st.header("📄 OCR & Data Extraction")
    st.info("🚧 Module under development - Coming soon!")

def show_news_page():
    st.header("📰 Market Intelligence")
    st.info("🚧 Module under development - Coming soon!")

def show_ml_page():
    st.header("🤖 ML Predictions")
    st.info("🚧 Module under development - Coming soon!")

def show_qa_page():
    st.header("💬 Q&A System")
    st.info("🚧 Module under development - Coming soon!")

if __name__ == "__main__":
    main()