import streamlit as st
import os
from dotenv import load_dotenv
import sys
import json
from datetime import datetime

# Page configuration MUST be first
st.set_page_config(
    page_title="Property Intelligence System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

# Import our modules AFTER set_page_config
try:
    from ocr_processor import PropertyOCRProcessor
except ImportError:
    st.error("OCR processor not found. Please ensure ocr_processor.py is in the src folder.")
    PropertyOCRProcessor = None

try:
    from market_intelligence import MarketIntelligence
except ImportError:
    st.error("Market intelligence module not found. Please ensure market_intelligence.py is in the src folder.")
    MarketIntelligence = None

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
        help="Upload a property appraisal PDF document or text file"
    )
    
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
                else:  # text file
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
    st.markdown("Get live real estate news, market trends, and investment insights for any location.")
    
    if MarketIntelligence is None:
        st.error("Market intelligence module not available. Please ensure market_intelligence.py is in the src folder.")
        return
    
    # Initialize market intelligence
    market_intel = MarketIntelligence()
    
    # Location input
    col1, col2 = st.columns([2, 1])
    with col1:
        location = st.text_input(
            "Enter location (city, state or leave blank for general news):",
            placeholder="e.g., Chicago, IL or Oak Brook, IL"
        )
    with col2:
        days_back = st.selectbox("News from last:", [7, 14, 30], index=0)
    
    # Search button
    if st.button("🔍 Get Market Intelligence", type="primary"):
        with st.spinner("Fetching real estate news and market data..."):
            # Get news data
            news_data = market_intel.search_real_estate_news(location, days_back)
            
            # Get market trends
            trends_data = market_intel.get_market_trends(location)
            
            # Create market summary
            market_summary = market_intel.create_market_summary(news_data, trends_data, location)
        
        st.success("✅ Market intelligence analysis complete!")
        
        # Display market score
        score = market_summary['market_score']
        st.subheader(f"📊 Market Health Score: {score}/100")
        
        # Color code the score
        if score >= 70:
            st.success(f"🟢 Strong Market Conditions ({score}/100)")
        elif score >= 50:
            st.warning(f"🟡 Moderate Market Conditions ({score}/100)")
        else:
            st.error(f"🔴 Challenging Market Conditions ({score}/100)")
        
        # Display in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📰 Latest News", "📈 Market Trends", "💡 Recommendations", "📊 Summary"])
        
        with tab1:
            st.subheader("📰 Latest Real Estate News")
            articles = news_data.get('articles', [])
            
            if articles:
                for i, article in enumerate(articles[:5]):  # Show top 5 articles
                    with st.expander(f"📄 {article.get('title', 'No title')}"):
                        st.write(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
                        st.write(f"**Published:** {article.get('publishedAt', 'Unknown')[:10]}")
                        st.write(f"**Description:** {article.get('description', 'No description available')}")
                        if article.get('url'):
                            st.markdown(f"[Read full article]({article['url']})")
            else:
                st.info("No recent news articles found for this location.")
        
        with tab2:
            st.subheader("📈 Market Trends & Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Price Change (1 Year)", f"+{trends_data.get('price_change_1year', 0)}%")
                st.metric("Days on Market", trends_data.get('days_on_market', 'N/A'))
            with col2:
                st.metric("Price Change (6 Months)", f"+{trends_data.get('price_change_6months', 0)}%")
                st.metric("Mortgage Rates", f"{trends_data.get('mortgage_rates', 0)}%")
            with col3:
                st.metric("Inventory Levels", trends_data.get('inventory_levels', 'Unknown'))
                st.metric("Market Temperature", trends_data.get('market_temperature', 'Unknown'))
            
            # Sentiment analysis
            st.subheader("📊 News Sentiment Analysis")
            sentiment = market_summary['news_summary']['sentiment_analysis']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positive", sentiment.get('positive', 0), delta=None)
            with col2:
                st.metric("Neutral", sentiment.get('neutral', 0), delta=None)
            with col3:
                st.metric("Negative", sentiment.get('negative', 0), delta=None)
        
        with tab3:
            st.subheader("💡 Investment Recommendations")
            recommendations = market_summary.get('recommendations', [])
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"**{i}.** {rec}")
            else:
                st.info("No specific recommendations available.")
        
        with tab4:
            st.subheader("📊 Complete Market Summary")
            st.json(market_summary)
            
            # Save and download options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Market Analysis"):
                    saved_path = market_intel.save_market_data(market_summary, location.replace(" ", "_") if location else "general")
                    if saved_path:
                        st.success(f"✅ Analysis saved to: {saved_path}")
            
            with col2:
                # Download as JSON
                json_data = json.dumps(market_summary, indent=2)
                st.download_button(
                    label="📥 Download Analysis",
                    data=json_data,
                    file_name=f"market_analysis_{location.replace(' ', '_') if location else 'general'}_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
    
    else:
        st.info("👆 Enter a location and click 'Get Market Intelligence' to start analysis")
        
        # Show sample data
        st.subheader("📋 Sample Analysis Preview")
        st.markdown("""
        **Example locations to try:**
        - Chicago, IL
        - Oak Brook, IL  
        - Hinsdale, IL
        - New York, NY
        - Miami, FL
        
        **The analysis includes:**
        - Latest real estate news and articles
        - Market trends and price changes
        - Sentiment analysis of market conditions
        - Investment recommendations
        - Overall market health score
        """)

def show_ml_page():
    st.header("🤖 ML Predictions")
    st.info("🚧 Module under development - Coming soon!")

def show_qa_page():
    st.header("💬 Q&A System")
    st.info("🚧 Module under development - Coming soon!")

if __name__ == "__main__":
    main()