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
try:
    from qa_system import PropertyQASystem
except ImportError:
    st.error("Q&A system module not found. Please ensure qa_system.py is in the src folder.")
    PropertyQASystem = None
try:
    from ml_predictions import PropertyMLPredictor
except ImportError:
    st.error("ML predictions module not found. Please ensure ml_predictions.py is in the src folder.")
    PropertyMLPredictor = None
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
    st.markdown("Predict property values and assess investment risk using machine learning models.")
    
    if PropertyMLPredictor is None:
        st.error("ML predictions module not available.")
        return
    
    # Initialize ML predictor
    if 'ml_predictor' not in st.session_state:
        st.session_state.ml_predictor = PropertyMLPredictor()
        # Train models with synthetic data
        with st.spinner("Training ML models..."):
            synthetic_data = st.session_state.ml_predictor.generate_synthetic_data(1000)
            st.session_state.ml_predictor.train_valuation_model(synthetic_data)
            st.session_state.ml_predictor.train_risk_model(synthetic_data)
        st.success("✅ ML models trained successfully!")
    
    ml_predictor = st.session_state.ml_predictor
    
    # Property input form
    st.subheader("🏠 Property Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        property_type = st.selectbox(
            "Property Type:",
            ['Single Family', 'Condo', 'Townhouse', 'Multi-Family', 'Commercial']
        )
        location = st.selectbox(
            "Location:",
            ['Chicago', 'Hinsdale', 'Oak Brook', 'Naperville', 'Wheaton', 'Glen Ellyn', 'Downers Grove']
        )
        sq_footage = st.number_input("Square Footage:", min_value=500, max_value=50000, value=2500)
        year_built = st.number_input("Year Built:", min_value=1900, max_value=2024, value=2000)
    
    with col2:
        bedrooms = st.number_input("Bedrooms:", min_value=0, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms:", min_value=1.0, max_value=10.0, value=2.5, step=0.5)
        lot_size = st.number_input("Lot Size (acres):", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    
    # Predict button
    if st.button("🔮 Generate Predictions", type="primary"):
        # Prepare property data
        property_data = {
            'property_type': property_type,
            'location': location,
            'sq_footage': sq_footage,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'year_built': year_built,
            'lot_size': lot_size,
            'estimated_value': 0  # Will be predicted
        }
        
        with st.spinner("Generating ML predictions..."):
            # Get predictions
            valuation_result = ml_predictor.predict_property_value(property_data)
            risk_result = ml_predictor.predict_risk_score(property_data)
            
            # Generate insights
            investment_insights = ml_predictor.generate_investment_insights(
                property_data, valuation_result, risk_result
            )
        
        st.success("✅ Predictions generated successfully!")
        
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💰 Valuation", "⚠️ Risk Analysis", "💡 Insights", "📊 Model Info"])
        
        with tab1:
            st.subheader("💰 Property Valuation")
            
            # Main prediction
            predicted_value = valuation_result['predicted_value']
            confidence = valuation_result['confidence']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Value", f"${predicted_value:,.0f}")
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            with col3:
                st.metric("Price/Sq Ft", f"${predicted_value/sq_footage:.0f}")
            
            # Price range
            price_range = valuation_result['price_range']
            st.subheader("📊 Value Range")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Low Estimate", f"${price_range['low']:,.0f}")
            with col2:
                st.metric("High Estimate", f"${price_range['high']:,.0f}")
            
            # Valuation chart
            val_chart = ml_predictor.create_valuation_chart(property_data, valuation_result)
            st.plotly_chart(val_chart, use_container_width=True)
        
        with tab2:
            st.subheader("⚠️ Investment Risk Analysis")
            
            # Risk metrics
            risk_score = risk_result['risk_score']
            risk_category = risk_result['risk_category']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Score", f"{risk_score:.1%}")
            with col2:
                st.metric("Risk Category", risk_category)
            
            # Risk gauge
            risk_chart = ml_predictor.create_risk_gauge(risk_result)
            st.plotly_chart(risk_chart, use_container_width=True)
        
        with tab3:
            st.subheader("💡 Investment Insights & Recommendations")
            
            # Investment grade
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Investment Grade", investment_insights['investment_grade'])
            with col2:
                st.metric("Overall Score", investment_insights['summary_score'])
            
            # Insights
            st.subheader("🔍 Key Insights")
            for insight in investment_insights['insights']:
                st.write(f"• {insight}")
            
            # Recommendations
            st.subheader("🎯 Recommendations")
            for rec in investment_insights['recommendations']:
                st.write(f"• {rec}")
        
        with tab4:
            st.subheader("📊 Model Performance & Feature Importance")
            
            # Feature importance chart
            importance_chart = ml_predictor.create_feature_importance_chart()
            if importance_chart:
                st.plotly_chart(importance_chart, use_container_width=True)
            
            # Model details
            st.subheader("🔧 Model Details")
            st.write("**Valuation Model:** Random Forest Regressor")
            st.write("**Risk Model:** Gradient Boosting Regressor")
            st.write("**Training Data:** 1,000 synthetic property records")
            st.write("**Features Used:** Square footage, bedrooms, bathrooms, age, location, property type")
        
        # Save results
        st.subheader("💾 Save Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Prediction"):
                results = {
                    'valuation': valuation_result,
                    'risk': risk_result,
                    'insights': investment_insights
                }
                saved_path = ml_predictor.save_prediction_results(property_data, results)
                if saved_path:
                    st.success(f"✅ Prediction saved to: {saved_path}")
        
        with col2:
            # Download results
            results_data = {
                'property': property_data,
                'valuation': valuation_result,
                'risk': risk_result,
                'insights': investment_insights
            }
            json_data = json.dumps(results_data, indent=2, default=str)
            st.download_button(
                label="📥 Download Results",
                data=json_data,
                file_name=f"ml_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        st.info("👆 Enter property details and click 'Generate Predictions' to start analysis")
        
        # Show sample predictions
        st.subheader("📋 Sample Prediction Preview")
        st.markdown("""
        **Example properties to try:**
        - **Single Family in Hinsdale**: 3,000 sq ft, 4 bed, 3 bath, built 2010
        - **Condo in Chicago**: 1,200 sq ft, 2 bed, 2 bath, built 2015  
        - **Commercial in Oak Brook**: 10,000 sq ft, 0 bed, 5 bath, built 2000
        
        **The ML analysis provides:**
        - Property valuation with confidence intervals
        - Investment risk scoring and categorization
        - Feature importance analysis
        - Actionable investment recommendations
        """)

def show_qa_page():
    st.header("💬 Q&A System")
    st.markdown("Ask questions about your property data using our conversational AI system.")
    
    if PropertyQASystem is None:
        st.error("Q&A system module not available.")
        return
    
    # Initialize Q&A system
    if 'qa_system' not in st.session_state:
        with st.spinner("Initializing Q&A system and loading knowledge base..."):
            st.session_state.qa_system = PropertyQASystem()
        st.success("✅ Q&A system ready!")
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    qa_system = st.session_state.qa_system
    
    # Conversation starters
    st.subheader("💡 Suggested Questions")
    starters = qa_system.get_conversation_starters()
    
    cols = st.columns(2)
    for i, starter in enumerate(starters):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(f"💭 {starter}", key=f"starter_{i}"):
                st.session_state.current_question = starter
    
    # Question input
    st.subheader("🗣️ Ask Your Question")
    
    # Use session state for question if set by button
    default_question = st.session_state.get('current_question', '')
    
    question = st.text_input(
        "Enter your property intelligence question:",
        value=default_question,
        placeholder="e.g., What factors affect property values in Hinsdale?"
    )
    
    # Clear the session state question after displaying
    if 'current_question' in st.session_state:
        del st.session_state.current_question
    
    # Ask button
    if st.button("🔍 Ask Question", type="primary") and question:
        with st.spinner("Searching knowledge base and generating answer..."):
            # Search for relevant context
            context_results = qa_system.search_knowledge(question, n_results=3)
            
            # Generate answer
            answer = qa_system.generate_answer(question, context_results)
            
            # Add to conversation history
            conversation_item = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'question': question,
                'answer': answer,
                'context_sources': len(context_results)
            }
            st.session_state.conversation_history.append(conversation_item)
        
        # Display answer
        st.success("✅ Answer generated!")
        
        # Create answer tabs
        tab1, tab2, tab3 = st.tabs(["💬 Answer", "📚 Sources", "🔍 Context"])
        
        with tab1:
            st.subheader("📝 Answer")
            st.markdown(answer)
            
            # Quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Helpful"):
                    st.success("Thanks for the feedback!")
            with col2:
                if st.button("👎 Not Helpful"):
                    st.info("We'll work on improving our responses!")
        
        with tab2:
            st.subheader("📚 Knowledge Sources")
            if context_results:
                for i, result in enumerate(context_results, 1):
                    with st.expander(f"Source {i}: {result['metadata'].get('topic', 'Unknown')}"):
                        st.write(f"**Category:** {result['metadata'].get('topic', 'N/A')}")
                        st.write(f"**Source:** {result['metadata'].get('source', 'N/A')}")
                        st.write(f"**Relevance Score:** {1 - result.get('distance', 0):.2f}")
                        st.text_area("Content:", result['content'], height=100, disabled=True)
            else:
                st.info("No specific sources found - using general knowledge.")
        
        with tab3:
            st.subheader("🔍 Search Context")
            st.write(f"**Question:** {question}")
            st.write(f"**Sources Found:** {len(context_results)}")
            st.write(f"**Knowledge Base Size:** {qa_system.collection.count() if qa_system.collection else 0} items")
            
            # Show knowledge base categories
            if context_results:
                categories = list(set([r['metadata'].get('topic', 'Unknown') for r in context_results]))
                st.write(f"**Relevant Categories:** {', '.join(categories)}")
    
    # Conversation History
    if st.session_state.conversation_history:
        st.subheader("📜 Conversation History")
        
        # Option to clear history
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.conversation_history = []
                st.experimental_rerun()
        
        # Display conversation
        for i, item in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):  # Show last 5
            with st.expander(f"Q{len(st.session_state.conversation_history)-i+1}: {item['question'][:60]}..."):
                st.write(f"**Time:** {item['timestamp']}")
                st.write(f"**Question:** {item['question']}")
                st.write(f"**Answer:** {item['answer'][:200]}..." if len(item['answer']) > 200 else item['answer'])
                st.write(f"**Sources Used:** {item['context_sources']}")
        
        # Save conversation
        if st.button("💾 Save Conversation"):
            saved_path = qa_system.save_conversation(st.session_state.conversation_history)
            if saved_path:
                st.success(f"✅ Conversation saved to: {saved_path}")
        
        # Download conversation
        json_data = json.dumps(st.session_state.conversation_history, indent=2, default=str)
        st.download_button(
            label="📥 Download Conversation",
            data=json_data,
            file_name=f"qa_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    else:
        st.info("👆 Ask a question to start the conversation!")
        
        # Show system capabilities
        st.subheader("🤖 System Knowledge Base")
        st.markdown("""
        **The Q&A system can answer questions about:**
        
        **🏠 Property Data:**
        - Extracted OCR data from your documents
        - Property characteristics and valuations
        - Sample data: Hinsdale Middle School Complex
        
        **📊 Market Intelligence:**
        - Market trends and analysis
        - Location-specific insights
        - Investment recommendations
        
        **🤖 ML Predictions:**
        - Property valuation methods
        - Risk assessment factors
        - Model performance and accuracy
        
        **📚 General Knowledge:**
        - Property types and characteristics
        - Valuation methodologies
        - Investment strategies
        
        **💡 Try asking about:**
        - Specific properties or locations
        - Market conditions and trends
        - System capabilities and features
        - Investment advice and risk factors
        """)

if __name__ == "__main__":
    main()