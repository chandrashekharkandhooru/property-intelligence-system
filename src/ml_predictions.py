import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PropertyMLPredictor:
    def __init__(self):
        self.valuation_model = None
        self.risk_model = None
        self.scaler = StandardScaler()
        self.le_property_type = LabelEncoder()
        self.le_location = LabelEncoder()
        self.feature_importance = None
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic property data for training"""
        np.random.seed(42)
        
        # Property types and locations
        property_types = ['Single Family', 'Condo', 'Townhouse', 'Multi-Family', 'Commercial']
        locations = ['Chicago', 'Hinsdale', 'Oak Brook', 'Naperville', 'Wheaton', 'Glen Ellyn', 'Downers Grove']
        
        data = []
        for i in range(n_samples):
            # Basic features
            property_type = np.random.choice(property_types)
            location = np.random.choice(locations)
            year_built = np.random.randint(1950, 2024)
            
            # Size features
            if property_type == 'Commercial':
                sq_footage = np.random.randint(5000, 50000)
                bedrooms = 0
                bathrooms = np.random.randint(2, 10)
                lot_size = np.random.uniform(0.5, 5.0)
            else:
                sq_footage = np.random.randint(800, 5000)
                bedrooms = np.random.randint(1, 6)
                bathrooms = np.random.uniform(1, 4)
                lot_size = np.random.uniform(0.1, 2.0)
            
            # Market factors
            age = 2024 - year_built
            location_premium = {
                'Hinsdale': 1.4, 'Oak Brook': 1.3, 'Naperville': 1.2,
                'Wheaton': 1.1, 'Glen Ellyn': 1.1, 'Downers Grove': 1.05, 'Chicago': 1.0
            }[location]
            
            # Property type multipliers
            type_multiplier = {
                'Single Family': 1.0, 'Townhouse': 0.85, 'Condo': 0.75,
                'Multi-Family': 1.2, 'Commercial': 0.8
            }[property_type]
            
            # Calculate base price
            base_price_per_sqft = np.random.uniform(150, 400)
            
            # Price calculation with noise
            estimated_value = (
                base_price_per_sqft * sq_footage * 
                location_premium * type_multiplier * 
                (1 - age * 0.01) *  # Depreciation
                np.random.uniform(0.9, 1.1)  # Market noise
            )
            
            # Risk factors
            risk_score = (
                (age / 100) * 0.3 +  # Age risk
                (1 / location_premium) * 0.2 +  # Location risk
                np.random.uniform(0, 0.5)  # Random market risk
            )
            risk_score = min(risk_score, 1.0)
            
            data.append({
                'property_type': property_type,
                'location': location,
                'sq_footage': sq_footage,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'year_built': year_built,
                'lot_size': lot_size,
                'estimated_value': estimated_value,
                'risk_score': risk_score
            })
        
        return pd.DataFrame(data)
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Calculate additional features
        df_processed['age'] = 2024 - df_processed['year_built']
        df_processed['price_per_sqft'] = df_processed['estimated_value'] / df_processed['sq_footage']
        df_processed['sqft_per_bedroom'] = df_processed['sq_footage'] / (df_processed['bedrooms'] + 1)
        
        # Encode categorical variables
        df_processed['property_type_encoded'] = self.le_property_type.fit_transform(df_processed['property_type'])
        df_processed['location_encoded'] = self.le_location.fit_transform(df_processed['location'])
        
        # Select features for model
        feature_columns = [
            'sq_footage', 'bedrooms', 'bathrooms', 'lot_size', 'age',
            'property_type_encoded', 'location_encoded', 'sqft_per_bedroom'
        ]
        
        return df_processed, feature_columns
    
    def train_valuation_model(self, df):
        """Train property valuation model"""
        df_processed, feature_columns = self.prepare_features(df)
        
        X = df_processed[feature_columns]
        y = df_processed['estimated_value']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.valuation_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.valuation_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.valuation_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.valuation_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'mae': mae,
            'r2': r2,
            'feature_importance': self.feature_importance
        }
    
    def train_risk_model(self, df):
        """Train property risk scoring model"""
        df_processed, feature_columns = self.prepare_features(df)
        
        X = df_processed[feature_columns]
        y = df_processed['risk_score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)  # Use same scaler from valuation
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.risk_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.risk_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.risk_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {'mae': mae, 'r2': r2}
    
    def predict_property_value(self, property_data):
        """Predict property value for new property"""
        if self.valuation_model is None:
            return None
        
        # Convert input to DataFrame
        df = pd.DataFrame([property_data])
        df_processed, feature_columns = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.transform(df_processed[feature_columns])
        
        # Predict
        predicted_value = self.valuation_model.predict(X_scaled)[0]
        
        # Get prediction confidence (simplified)
        # Use ensemble predictions to estimate confidence
        tree_predictions = [tree.predict(X_scaled)[0] for tree in self.valuation_model.estimators_]
        confidence = 1 - (np.std(tree_predictions) / np.mean(tree_predictions))
        
        return {
            'predicted_value': predicted_value,
            'confidence': confidence,
            'price_range': {
                'low': predicted_value * 0.9,
                'high': predicted_value * 1.1
            }
        }
    
    def predict_risk_score(self, property_data):
        """Predict risk score for property"""
        if self.risk_model is None:
            return None
        
        # Convert input to DataFrame
        df = pd.DataFrame([property_data])
        df_processed, feature_columns = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.transform(df_processed[feature_columns])
        
        # Predict
        risk_score = self.risk_model.predict(X_scaled)[0]
        risk_score = max(0, min(1, risk_score))  # Clamp between 0 and 1
        
        # Risk categories
        if risk_score < 0.3:
            risk_category = "Low Risk"
            risk_color = "green"
        elif risk_score < 0.6:
            risk_category = "Medium Risk"
            risk_color = "orange"
        else:
            risk_category = "High Risk"
            risk_color = "red"
        
        return {
            'risk_score': risk_score,
            'risk_category': risk_category,
            'risk_color': risk_color
        }
    
    def create_valuation_chart(self, property_data, prediction_result):
        """Create property valuation visualization"""
        # Create comparison chart
        fig = go.Figure()
        
        predicted_value = prediction_result['predicted_value']
        price_range = prediction_result['price_range']
        
        # Add predicted value
        fig.add_trace(go.Bar(
            x=['Predicted Value'],
            y=[predicted_value],
            name='Predicted Value',
            marker_color='lightblue'
        ))
        
        # Add confidence range
        fig.add_trace(go.Scatter(
            x=['Predicted Value', 'Predicted Value'],
            y=[price_range['low'], price_range['high']],
            mode='lines+markers',
            name='Confidence Range',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"Property Valuation: ${predicted_value:,.0f}",
            yaxis_title="Value ($)",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_feature_importance_chart(self):
        """Create feature importance visualization"""
        if self.feature_importance is None:
            return None
        
        fig = px.bar(
            self.feature_importance.head(8),
            x='importance',
            y='feature',
            orientation='h',
            title='Most Important Factors for Property Valuation'
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_risk_gauge(self, risk_result):
        """Create risk score gauge chart"""
        risk_score = risk_result['risk_score']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Investment Risk Score"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_result['risk_color']},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        return fig
    
    def generate_investment_insights(self, property_data, valuation_result, risk_result):
        """Generate investment insights and recommendations"""
        insights = []
        
        # Valuation insights
        predicted_value = valuation_result['predicted_value']
        confidence = valuation_result['confidence']
        
        if confidence > 0.8:
            insights.append(f"✅ High confidence prediction (confidence: {confidence:.1%})")
        elif confidence > 0.6:
            insights.append(f"⚠️ Moderate confidence prediction (confidence: {confidence:.1%})")
        else:
            insights.append(f"❌ Low confidence prediction (confidence: {confidence:.1%})")
        
        # Price per square foot analysis
        price_per_sqft = predicted_value / property_data['sq_footage']
        if price_per_sqft > 300:
            insights.append("💰 Premium price per square foot - luxury market segment")
        elif price_per_sqft > 200:
            insights.append("🏠 Moderate price per square foot - mid-market segment")
        else:
            insights.append("💸 Affordable price per square foot - value segment")
        
        # Age analysis
        age = 2024 - property_data['year_built']
        if age < 10:
            insights.append("🆕 New construction - minimal maintenance concerns")
        elif age < 30:
            insights.append("🔧 Modern property - standard maintenance expected")
        else:
            insights.append("🏚️ Older property - consider renovation costs")
        
        # Risk insights
        risk_score = risk_result['risk_score']
        if risk_score < 0.3:
            insights.append("🛡️ Low investment risk - stable asset")
        elif risk_score < 0.6:
            insights.append("⚖️ Moderate risk - balanced risk-reward profile")
        else:
            insights.append("⚠️ High risk - requires careful consideration")
        
        # Market recommendations
        recommendations = []
        
        if risk_score < 0.4 and confidence > 0.7:
            recommendations.append("🎯 RECOMMENDED: Strong investment opportunity")
        elif risk_score > 0.7:
            recommendations.append("🚨 CAUTION: High-risk investment - thorough due diligence required")
        else:
            recommendations.append("🤔 NEUTRAL: Average investment potential")
        
        if price_per_sqft < 200:
            recommendations.append("💡 Consider for value investing strategy")
        
        if age > 50:
            recommendations.append("🔨 Budget for potential renovations and updates")
        
        return {
            'insights': insights,
            'recommendations': recommendations,
            'summary_score': f"{(1-risk_score)*100:.0f}/100",
            'investment_grade': risk_result['risk_category']
        }
    
    def save_prediction_results(self, property_data, results):
        """Save prediction results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml_prediction_{timestamp}.json"
            filepath = f"data/processed/{filename}"
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            save_data = {
                'timestamp': timestamp,
                'property_data': property_data,
                'predictions': results
            }
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            return filepath
            
        except Exception as e:
            st.error(f"Error saving prediction results: {str(e)}")
            return None