import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
import os

class MarketIntelligence:
    def __init__(self):
        self.base_url = "https://newsapi.org/v2/"
        # Note: For production, get a free API key from https://newsapi.org/
        self.api_key = os.getenv("NEWS_API_KEY", "demo_key")
        
    def search_real_estate_news(self, location="", days_back=7):
        """Search for real estate news in specific location"""
        try:
            # Build search query
            if location:
                query = f"real estate {location} property market housing"
            else:
                query = "real estate property market housing prices"
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # API parameters
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'pageSize': 20
            }
            
            # For demo purposes, return mock data if no API key
            if self.api_key == "demo_key":
                return self._get_mock_news_data(location)
            
            # Make API request
            url = f"{self.base_url}everything"
            headers = {'X-API-Key': self.api_key}
            
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"News API error: {response.status_code}")
                return self._get_mock_news_data(location)
                
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return self._get_mock_news_data(location)
    
    def _get_mock_news_data(self, location=""):
        """Return mock news data for demonstration"""
        location_text = f" in {location}" if location else ""
        
        mock_articles = [
            {
                'title': f'Housing Market Trends{location_text} Show Continued Growth',
                'description': f'Recent analysis shows property values{location_text} have increased by 8.5% over the past year, with strong demand continuing.',
                'url': 'https://example.com/housing-trends',
                'publishedAt': '2024-06-10T10:30:00Z',
                'source': {'name': 'Real Estate Weekly'},
                'sentiment': 'positive'
            },
            {
                'title': f'Commercial Real Estate Investment{location_text} Reaches New Heights',
                'description': f'Investors are showing increased interest in commercial properties{location_text}, with transaction volumes up 15%.',
                'url': 'https://example.com/commercial-investment',
                'publishedAt': '2024-06-09T14:15:00Z',
                'source': {'name': 'Property Investment News'},
                'sentiment': 'positive'
            },
            {
                'title': f'Market Analysis: Residential Property Outlook{location_text}',
                'description': f'Experts predict moderate growth in residential property values{location_text} for the remainder of 2024.',
                'url': 'https://example.com/market-analysis',
                'publishedAt': '2024-06-08T09:45:00Z',
                'source': {'name': 'Market Research Today'},
                'sentiment': 'neutral'
            },
            {
                'title': f'Interest Rate Impact on Real Estate{location_text}',
                'description': f'Recent interest rate changes are affecting mortgage applications and property sales{location_text}.',
                'url': 'https://example.com/interest-rates',
                'publishedAt': '2024-06-07T16:20:00Z',
                'source': {'name': 'Financial Times Real Estate'},
                'sentiment': 'neutral'
            }
        ]
        
        return {
            'status': 'ok',
            'totalResults': len(mock_articles),
            'articles': mock_articles
        }
    
    def analyze_market_sentiment(self, articles):
        """Analyze sentiment of market news"""
        if not articles:
            return {'positive': 0, 'neutral': 0, 'negative': 0}
        
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for article in articles:
            # Simple keyword-based sentiment analysis
            title_desc = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            
            positive_words = ['growth', 'increase', 'strong', 'surge', 'boom', 'rising', 'gains', 'opportunity']
            negative_words = ['decline', 'fall', 'drop', 'crash', 'downturn', 'crisis', 'concern', 'risk']
            
            positive_score = sum(1 for word in positive_words if word in title_desc)
            negative_score = sum(1 for word in negative_words if word in title_desc)
            
            if positive_score > negative_score:
                sentiment_counts['positive'] += 1
            elif negative_score > positive_score:
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1
        
        return sentiment_counts
    
    def get_market_trends(self, location=""):
        """Get market trends and insights"""
        try:
            # Mock market data for demonstration
            trends_data = {
                'price_change_1year': 8.5,
                'price_change_6months': 4.2,
                'inventory_levels': 'Low',
                'days_on_market': 23,
                'mortgage_rates': 7.25,
                'market_temperature': 'Hot',
                'demand_supply_ratio': 1.8,
                'forecast_6months': 'Continued Growth'
            }
            
            return trends_data
            
        except Exception as e:
            st.error(f"Error fetching market trends: {str(e)}")
            return {}
    
    def save_market_data(self, data, location="general"):
        """Save market intelligence data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_intelligence_{location}_{timestamp}.json"
            filepath = f"data/processed/{filename}"
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return filepath
            
        except Exception as e:
            st.error(f"Error saving market data: {str(e)}")
            return None
    
    def create_market_summary(self, news_data, trends_data, location=""):
        """Create a comprehensive market summary"""
        location_text = f" for {location}" if location else ""
        
        summary = {
            'location': location,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'news_summary': {
                'total_articles': news_data.get('totalResults', 0),
                'sentiment_analysis': self.analyze_market_sentiment(news_data.get('articles', [])),
                'key_headlines': [article.get('title', '') for article in news_data.get('articles', [])[:3]]
            },
            'market_trends': trends_data,
            'market_score': self._calculate_market_score(news_data, trends_data),
            'recommendations': self._generate_recommendations(news_data, trends_data)
        }
        
        return summary
    
    def _calculate_market_score(self, news_data, trends_data):
        """Calculate overall market health score (0-100)"""
        score = 50  # Base score
        
        # Adjust based on sentiment
        sentiment = self.analyze_market_sentiment(news_data.get('articles', []))
        total_articles = sum(sentiment.values())
        if total_articles > 0:
            positive_ratio = sentiment['positive'] / total_articles
            score += (positive_ratio - 0.5) * 40  # +/- 20 points based on sentiment
        
        # Adjust based on price trends
        price_change = trends_data.get('price_change_1year', 0)
        if price_change > 5:
            score += 15
        elif price_change < -5:
            score -= 15
        
        # Adjust based on market temperature
        market_temp = trends_data.get('market_temperature', 'Neutral')
        if market_temp == 'Hot':
            score += 10
        elif market_temp == 'Cold':
            score -= 10
        
        return max(0, min(100, int(score)))
    
    def _generate_recommendations(self, news_data, trends_data):
        """Generate investment recommendations based on data"""
        recommendations = []
        
        # Based on price trends
        price_change = trends_data.get('price_change_1year', 0)
        if price_change > 10:
            recommendations.append("Strong market growth - Consider strategic property investments")
        elif price_change < -5:
            recommendations.append("Market correction - Potential buying opportunities for long-term investors")
        else:
            recommendations.append("Stable market conditions - Good for steady investment strategies")
        
        # Based on inventory
        inventory = trends_data.get('inventory_levels', 'Normal')
        if inventory == 'Low':
            recommendations.append("Low inventory levels - Sellers market, expect competitive pricing")
        elif inventory == 'High':
            recommendations.append("High inventory - Buyers market, more negotiation opportunities")
        
        # Based on interest rates
        mortgage_rate = trends_data.get('mortgage_rates', 7.0)
        if mortgage_rate > 7.5:
            recommendations.append("High interest rates - Consider all-cash offers or adjustable rate mortgages")
        elif mortgage_rate < 5.0:
            recommendations.append("Favorable interest rates - Good time for leveraged investments")
        
        return recommendations