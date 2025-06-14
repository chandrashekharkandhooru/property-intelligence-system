import streamlit as st
import os
from dotenv import load_dotenv
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

# Import our OCR processor
try:
    from ocr_processor import PropertyOCRProcessor
except ImportError:
    st.error("OCR processor not found. Please ensure ocr_processor.py is in the src folder.")
    PropertyOCRProcessor = None

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
    st.markdown("Upload a property appraisal PDF to extract structured data automatically.")
    
    if PropertyOCRProcessor is None:
        st.error("OCR processor not available. Please check module installation.")
        return
    
    # Initialize OCR processor
    ocr_processor = PropertyOCRProcessor()
    
    # File upload
    uploaded_file = st.file_uploader(
    "Choose a PDF or TXT file", 
    type=["pdf", "txt"],
    help="Upload a property appraisal PDF document or text file")
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Show file details
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📄 File size: {uploaded_file.size:,} bytes")
        with col2:
            st.info(f"📝 File type: {uploaded_file.type}")
        
        # Process button
        if st.button("🔍 Extract Data", type="primary"):
            with st.spinner("Extracting text from PDF..."):
                
                # Extract text based on file type
                if uploaded_file.type == "application/pdf":
                    extracted_text = ocr_processor.extract_text_from_pdf(uploaded_file)
                else:
                    extracted_text = ocr_processor.extract_text_from_txt(uploaded_file)
                
                if extracted_text:
                    st.success("✅ Text extraction successful!")
                    
                    # Show extracted text preview
                    with st.expander("📖 View Extracted Text (Preview)"):
                        st.text_area(
                            "Raw extracted text:",
                            extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,
                            height=200
                        )
                    
                    # Parse structured data
                    with st.spinner("Parsing property data..."):
                        property_data = ocr_processor.parse_property_data(extracted_text)
                    
                    st.success("✅ Data parsing complete!")
                    
                    # Display structured data
                    st.subheader("🏡 Extracted Property Information")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📍 Property Details")
                        st.write(f"**Address:** {property_data['property_address']}")
                        st.write(f"**Type:** {property_data['property_type']}")
                        st.write(f"**Square Footage:** {property_data['square_footage']} sq ft")
                        st.write(f"**Lot Size:** {property_data['lot_size']}")
                        st.write(f"**Year Built:** {property_data['year_built']}")
                    
                    with col2:
                        st.markdown("### 💰 Valuation & Features")
                        st.write(f"**Appraised Value:** {property_data['appraised_value']}")
                        st.write(f"**Bedrooms:** {property_data['bedrooms']}")
                        st.write(f"**Bathrooms:** {property_data['bathrooms']}")
                        st.write(f"**Location:** {property_data['location']}")
                        st.write(f"**Appraisal Date:** {property_data['appraisal_date']}")
                    
                    # Save data option
                    if st.button("💾 Save Extracted Data"):
                        filename = uploaded_file.name.replace('.pdf', '')
                        saved_path = ocr_processor.save_extracted_data(property_data, filename)
                        if saved_path:
                            st.success(f"✅ Data saved to: {saved_path}")
                    
                    # Download as JSON
                    import json
                    json_data = json.dumps(property_data, indent=2)
                    st.download_button(
                        label="📥 Download as JSON",
                        data=json_data,
                        file_name=f"{uploaded_file.name.replace('.pdf', '')}_extracted.json",
                        mime="application/json"
                    )
                else:
                    st.error("❌ Failed to extract text from PDF")
    
    else:
        st.info("👆 Please upload a PDF file to begin extraction")

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