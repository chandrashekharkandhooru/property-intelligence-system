import streamlit as st
import os
import json
import glob
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import warnings
warnings.filterwarnings('ignore')

class PropertyQASystem:
    def __init__(self):
        self.embeddings_model = None
        self.chroma_client = None
        self.collection = None
        self.knowledge_base = []
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the Q&A system components"""
        try:
            # Initialize sentence transformer for embeddings
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection("property_knowledge")
            except:
                self.collection = self.chroma_client.create_collection("property_knowledge")
            
            # Load existing knowledge base
            self.load_knowledge_base()
            
        except Exception as e:
            st.error(f"Error initializing Q&A system: {str(e)}")
    
    def load_knowledge_base(self):
        """Load knowledge base from processed data files"""
        knowledge_items = []
        
        # Load sample property knowledge
        sample_knowledge = [
            {
                "id": "property_types",
                "content": "Property types include Single Family homes, Condominiums, Townhouses, Multi-Family properties, and Commercial buildings. Each type has different valuation factors and investment characteristics.",
                "category": "property_basics",
                "metadata": {"topic": "property_types", "source": "system"}
            },
            {
                "id": "location_factors",
                "content": "Location is one of the most important factors in property valuation. Premium locations like Hinsdale and Oak Brook command higher prices due to school districts, amenities, and proximity to transportation.",
                "category": "valuation",
                "metadata": {"topic": "location", "source": "system"}
            },
            {
                "id": "age_depreciation",
                "content": "Property age affects valuation through depreciation. Newer properties (0-10 years) require minimal maintenance, while older properties (30+ years) may need significant updates and renovations.",
                "category": "valuation",
                "metadata": {"topic": "age", "source": "system"}
            },
            {
                "id": "market_trends",
                "content": "Current market trends show varying price changes across different locations. Mortgage rates, inventory levels, and economic conditions all influence property values and investment timing.",
                "category": "market",
                "metadata": {"topic": "trends", "source": "system"}
            },
            {
                "id": "investment_risk",
                "content": "Investment risk factors include property age, location desirability, market conditions, and maintenance requirements. Low-risk properties are typically newer, in prime locations, with stable market conditions.",
                "category": "investment",
                "metadata": {"topic": "risk", "source": "system"}
            },
            {
                "id": "hinsdale_school",
                "content": "The Hinsdale Middle School Complex is a 107,500 square foot educational facility on 9.51 acres, valued at $7,000,000. It represents a significant institutional property investment in a premium school district.",
                "category": "case_study",
                "metadata": {"topic": "hinsdale", "source": "sample_data"}
            },
            {
                "id": "valuation_methods",
                "content": "Property valuation uses multiple approaches: comparable sales analysis, cost approach considering replacement costs, and income approach for rental properties. ML models can integrate these factors for more accurate predictions.",
                "category": "valuation",
                "metadata": {"topic": "methods", "source": "system"}
            },
            {
                "id": "chicago_suburbs",
                "content": "Chicago suburbs like Naperville, Wheaton, Glen Ellyn, and Downers Grove offer different value propositions. Each has unique characteristics affecting property values, from school ratings to commuter access.",
                "category": "locations",
                "metadata": {"topic": "suburbs", "source": "system"}
            }
        ]
        
        # Add sample knowledge to collection if empty
        if self.collection.count() == 0:
            self.add_knowledge_batch(sample_knowledge)
        
        # Load from processed data files
        self.load_from_data_files()
    
    def load_from_data_files(self):
        """Load knowledge from processed data files"""
        try:
            # Load OCR extracted data
            ocr_files = glob.glob("data/processed/ocr_*.json")
            for file_path in ocr_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.add_property_data_to_knowledge(data, file_path)
            
            # Load market intelligence data
            market_files = glob.glob("data/processed/market_analysis_*.json")
            for file_path in market_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.add_market_data_to_knowledge(data, file_path)
            
            # Load ML prediction data
            ml_files = glob.glob("data/processed/ml_prediction_*.json")
            for file_path in ml_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.add_ml_data_to_knowledge(data, file_path)
                    
        except Exception as e:
            st.warning(f"Could not load some data files: {str(e)}")
    
    def add_property_data_to_knowledge(self, data, source_file):
        """Add property OCR data to knowledge base"""
        try:
            content = f"""
            Property Address: {data.get('property_address', 'N/A')}
            Property Type: {data.get('property_type', 'N/A')}
            Square Footage: {data.get('square_footage', 'N/A')} sq ft
            Bedrooms: {data.get('bedrooms', 'N/A')}
            Bathrooms: {data.get('bathrooms', 'N/A')}
            Year Built: {data.get('year_built', 'N/A')}
            Lot Size: {data.get('lot_size', 'N/A')}
            Appraised Value: {data.get('appraised_value', 'N/A')}
            Location: {data.get('location', 'N/A')}
            Appraisal Date: {data.get('appraisal_date', 'N/A')}
            """
            
            knowledge_item = {
                "id": f"property_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "content": content.strip(),
                "category": "property_data",
                "metadata": {
                    "topic": "property_details",
                    "source": source_file,
                    "address": data.get('property_address', ''),
                    "type": data.get('property_type', '')
                }
            }
            
            self.add_knowledge_item(knowledge_item)
            
        except Exception as e:
            st.warning(f"Error adding property data: {str(e)}")
    
    def add_market_data_to_knowledge(self, data, source_file):
        """Add market intelligence data to knowledge base"""
        try:
            market_score = data.get('market_score', 'N/A')
            location = data.get('location', 'General Market')
            
            content = f"""
            Market Analysis for {location}:
            Market Health Score: {market_score}/100
            Analysis Date: {data.get('timestamp', 'N/A')}
            
            Key Recommendations:
            {chr(10).join(data.get('recommendations', []))}
            
            Market Trends: The analysis shows current market conditions and sentiment for the {location} area.
            """
            
            knowledge_item = {
                "id": f"market_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "content": content.strip(),
                "category": "market_intelligence",
                "metadata": {
                    "topic": "market_analysis",
                    "source": source_file,
                    "location": location,
                    "score": market_score
                }
            }
            
            self.add_knowledge_item(knowledge_item)
            
        except Exception as e:
            st.warning(f"Error adding market data: {str(e)}")
    
    def add_ml_data_to_knowledge(self, data, source_file):
        """Add ML prediction data to knowledge base"""
        try:
            property_data = data.get('property', {})
            valuation = data.get('valuation', {})
            risk = data.get('risk', {})
            
            predicted_value = valuation.get('predicted_value', 0)
            risk_score = risk.get('risk_score', 0)
            
            content = f"""
            ML Prediction Analysis:
            Property: {property_data.get('property_type', 'N/A')} in {property_data.get('location', 'N/A')}
            Square Footage: {property_data.get('sq_footage', 'N/A')} sq ft
            Predicted Value: ${predicted_value:,.0f}
            Confidence: {valuation.get('confidence', 0):.1%}
            Investment Risk Score: {risk_score:.1%}
            Risk Category: {risk.get('risk_category', 'N/A')}
            
            This ML analysis provides data-driven insights for property valuation and investment decisions.
            """
            
            knowledge_item = {
                "id": f"ml_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "content": content.strip(),
                "category": "ml_predictions",
                "metadata": {
                    "topic": "ml_analysis",
                    "source": source_file,
                    "predicted_value": predicted_value,
                    "risk_category": risk.get('risk_category', '')
                }
            }
            
            self.add_knowledge_item(knowledge_item)
            
        except Exception as e:
            st.warning(f"Error adding ML data: {str(e)}")
    
    def add_knowledge_item(self, knowledge_item):
        """Add a single knowledge item to the vector database"""
        try:
            # Generate embedding
            embedding = self.embeddings_model.encode([knowledge_item['content']])[0].tolist()
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[knowledge_item['content']],
                metadatas=[knowledge_item['metadata']],
                ids=[knowledge_item['id']]
            )
            
        except Exception as e:
            st.warning(f"Error adding knowledge item: {str(e)}")
    
    def add_knowledge_batch(self, knowledge_items):
        """Add multiple knowledge items to the vector database"""
        try:
            if not knowledge_items:
                return
            
            contents = [item['content'] for item in knowledge_items]
            embeddings = self.embeddings_model.encode(contents).tolist()
            
            self.collection.add(
                embeddings=embeddings,
                documents=contents,
                metadatas=[item['metadata'] for item in knowledge_items],
                ids=[item['id'] for item in knowledge_items]
            )
            
        except Exception as e:
            st.error(f"Error adding knowledge batch: {str(e)}")
    
    def search_knowledge(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search the knowledge base for relevant information"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query])[0].tolist()
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else 0
                })
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def generate_answer(self, question: str, context_results: List[Dict]) -> str:
        """Generate an answer based on the question and retrieved context"""
        # Create context from search results
        context_text = "\n\n".join([result['content'] for result in context_results])
        
        # Simple rule-based response generation (in production, you'd use OpenAI/LLM here)
        answer = self.create_rule_based_answer(question, context_text, context_results)
        
        return answer
    
    def create_rule_based_answer(self, question: str, context: str, results: List[Dict]) -> str:
        """Create a rule-based answer (simulating LLM response)"""
        question_lower = question.lower()
        
        # Property value questions
        if any(word in question_lower for word in ['value', 'price', 'worth', 'cost']):
            if 'hinsdale' in question_lower:
                return f"""Based on the available data, the Hinsdale Middle School Complex is valued at $7,000,000. This is a 107,500 square foot educational facility on 9.51 acres.

For other properties in Hinsdale, values typically range higher due to the premium location and excellent school district. The ML prediction model shows that location is one of the most important factors in valuation.

**Key factors affecting Hinsdale property values:**
- Premium school district
- Desirable suburban location
- Strong property appreciation trends
- Low investment risk profile

Would you like a specific valuation for a particular property? I can help you with ML-based predictions."""

            elif any(loc in question_lower for loc in ['chicago', 'oak brook', 'naperville']):
                return f"""Property values vary significantly by location in the Chicago area. Based on our market intelligence and ML models:

**General Value Ranges by Location:**
- **Hinsdale**: Premium pricing (highest values)
- **Oak Brook**: High-end suburban market
- **Naperville**: Strong family-oriented market
- **Chicago**: Varies widely by neighborhood

**Factors influencing values:**
- Square footage and lot size
- Property age and condition
- Local school ratings
- Market conditions and trends

For specific valuations, I recommend using our ML Prediction module with your property details."""

            else:
                return f"""Property valuation depends on multiple factors that our system analyzes:

**Key Valuation Factors:**
1. **Location** - Most important factor (premium areas like Hinsdale command higher prices)
2. **Size** - Square footage and lot size
3. **Age** - Newer properties typically valued higher
4. **Property Type** - Single family, condo, townhouse, etc.
5. **Market Conditions** - Current trends and economic factors

**Our System's Approach:**
- ML models trained on market data
- Comparable sales analysis
- Risk assessment integration
- Market intelligence insights

Use our ML Predictions module for specific property valuations with confidence intervals."""

        # Location questions
        elif any(word in question_lower for word in ['location', 'area', 'neighborhood', 'where']):
            return f"""Our system covers multiple Chicago-area locations, each with distinct characteristics:

**Premium Locations:**
- **Hinsdale**: Top-tier school district, luxury properties, low risk
- **Oak Brook**: Corporate center, upscale suburban community

**Strong Family Markets:**
- **Naperville**: Excellent schools, family-friendly amenities
- **Wheaton**: Historic charm, good value proposition
- **Glen Ellyn**: Quiet suburban setting, stable market
- **Downers Grove**: Transit access, diverse housing options

**Location Impact on Investment:**
- Premium locations show stronger appreciation
- School district quality heavily influences values
- Transportation access affects desirability
- Market stability varies by area

**Sample Data Available:**
Our system includes detailed analysis for the Hinsdale Middle School Complex and can generate predictions for properties in all listed locations."""

        # Risk questions
        elif any(word in question_lower for word in ['risk', 'safe', 'investment', 'returns']):
            return f"""Investment risk assessment is a key feature of our system. Here's what we analyze:

**Risk Factors Evaluated:**
1. **Property Age** - Older properties (30+ years) carry higher risk
2. **Location Stability** - Premium areas like Hinsdale are lower risk
3. **Market Conditions** - Current trends and economic indicators
4. **Maintenance Requirements** - Age and condition impact

**Risk Categories:**
- **Low Risk (0-30%)**: New properties, prime locations, stable markets
- **Medium Risk (30-60%)**: Moderate age/location factors
- **High Risk (60%+)**: Older properties, challenging markets

**Investment Recommendations:**
- Hinsdale area properties typically score as low-risk
- Consider renovation costs for older properties
- Monitor market trends through our Market Intelligence module
- Use ML predictions for data-driven decisions

Our ML model provides specific risk scores with detailed analysis for any property you evaluate."""

        # System capabilities questions
        elif any(word in question_lower for word in ['system', 'features', 'modules', 'what can', 'how does']):
            return f"""The Property Intelligence System provides comprehensive AI-powered property analysis:

**🏠 Core Modules:**

**1. OCR & Data Extraction**
- Extract structured data from property documents
- Parse appraisal PDFs and text files
- Automatic data organization and export

**2. Market Intelligence**
- Live news analysis for any location
- Market trends and sentiment analysis
- Investment recommendations and market health scoring

**3. ML Predictions**
- Property valuation with confidence intervals
- Investment risk scoring and categorization
- Feature importance analysis and insights

**4. Q&A System (Current)**
- Conversational interface for property questions
- Context-aware responses using your data
- Integration with all other modules

**📊 Sample Data:**
- Hinsdale Middle School Complex ($7M, 107,500 sq ft)
- Chicago suburbs market analysis
- ML models trained on 1,000+ property records

**🚀 Key Benefits:**
- Data-driven property decisions
- Automated document processing
- Real-time market insights
- Comprehensive risk assessment"""

        # Sample data questions
        elif any(word in question_lower for word in ['sample', 'example', 'hinsdale', 'school']):
            return f"""**Sample Property: Hinsdale Middle School Complex**

**Property Details:**
- **Address**: Hinsdale, Illinois
- **Type**: Educational/Institutional
- **Size**: 107,500 square feet
- **Land**: 9.51 acres
- **Valuation**: $7,000,000
- **Location**: Premium Hinsdale school district

**Why This Example:**
- Represents significant institutional property investment
- Located in one of Illinois' top school districts
- Demonstrates large-scale property valuation
- Shows premium location value impact

**Analysis Capabilities:**
- OCR processing of appraisal documents
- Market intelligence for Hinsdale area
- ML predictions for similar properties
- Risk assessment for institutional investments

**Additional Samples:**
You can test the system with various property types:
- Single family homes (1,000-5,000 sq ft)
- Condominiums and townhouses
- Commercial properties
- Multi-family investments

Each analysis provides comprehensive insights across all four modules."""

        # ML and predictions questions
        elif any(word in question_lower for word in ['predict', 'model', 'machine learning', 'ml', 'algorithm']):
            return f"""**ML Prediction System Overview:**

**🤖 Model Architecture:**
- **Valuation Model**: Random Forest Regressor
- **Risk Model**: Gradient Boosting Regressor
- **Training Data**: 1,000+ synthetic property records
- **Accuracy**: High confidence with ensemble methods

**📊 Input Features:**
1. Square footage (most important)
2. Location (major value driver)
3. Property type and age
4. Bedrooms, bathrooms, lot size
5. Market conditions and trends

**🎯 Outputs Provided:**
- **Property Value**: Prediction with confidence interval
- **Risk Score**: 0-100% investment risk assessment
- **Feature Importance**: Which factors matter most
- **Investment Grade**: Recommendation category

**⚡ Capabilities:**
- Real-time predictions for any property
- Confidence scoring for reliability
- Comparative analysis across locations
- Integration with market intelligence

**🔍 Model Performance:**
- Feature importance analysis shows location and size as top factors
- Confidence intervals provide realistic value ranges
- Risk scoring helps with investment decisions
- Continuous learning from new data

Try the ML Predictions module with your property details for instant analysis!"""

        # General help or unclear questions
        else:
            return f"""I'm here to help with property intelligence questions! Here are some examples of what I can assist with:

**🏠 Property Valuation:**
- "What's the value of a 3,000 sq ft home in Hinsdale?"
- "How does location affect property prices?"
- "What factors influence property values?"

**📊 Market Analysis:**
- "What are the market trends in Oak Brook?"
- "Is now a good time to invest in Chicago suburbs?"
- "How do I assess market conditions?"

**⚠️ Investment Risk:**
- "What makes a property high-risk?"
- "How do you calculate investment risk?"
- "Which locations are safest for investment?"

**🤖 System Features:**
- "What can this system do?"
- "How do the ML predictions work?"
- "What sample data is available?"

**💡 Specific Examples:**
- Ask about the Hinsdale Middle School Complex
- Request valuations for different property types
- Inquire about Chicago suburb characteristics

Feel free to ask more specific questions about properties, markets, or system capabilities!"""

        return answer

    def get_conversation_starters(self) -> List[str]:
        """Get suggested conversation starters"""
        return [
            "What's the value of the Hinsdale Middle School Complex?",
            "How do property values compare across Chicago suburbs?",
            "What factors affect investment risk in real estate?",
            "What can this Property Intelligence System do?",
            "How accurate are the ML property predictions?",
            "What makes Hinsdale properties valuable?",
            "How do you assess market conditions?",
            "What's the difference between property types for investment?"
        ]
    
    def save_conversation(self, conversation_history):
        """Save conversation for future reference"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qa_conversation_{timestamp}.json"
            filepath = f"data/processed/{filename}"
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(conversation_history, f, indent=2, default=str)
            
            return filepath
            
        except Exception as e:
            st.error(f"Error saving conversation: {str(e)}")
            return None