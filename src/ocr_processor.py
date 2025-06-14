import streamlit as st
import PyPDF2
from PIL import Image
import pandas as pd
import json
import os
from datetime import datetime

class PropertyOCRProcessor:
    def __init__(self):
        self.extracted_data = {}
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return None
    
    def parse_property_data(self, text):
        """Parse property information from extracted text"""
        data = {
            "property_address": self._extract_address(text),
            "property_type": self._extract_property_type(text),
            "square_footage": self._extract_square_footage(text),
            "lot_size": self._extract_lot_size(text),
            "year_built": self._extract_year_built(text),
            "appraised_value": self._extract_appraised_value(text),
            "bedrooms": self._extract_bedrooms(text),
            "bathrooms": self._extract_bathrooms(text),
            "location": self._extract_location(text),
            "appraisal_date": self._extract_appraisal_date(text)
        }
        return data
    
    def _extract_address(self, text):
        """Extract property address from text"""
        # Simple pattern matching - can be enhanced with regex
        lines = text.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['subject property', 'property address', 'address:']):
                return line.strip()
        return "Address not found"
    
    def _extract_property_type(self, text):
        """Extract property type"""
        property_types = ['single family', 'condo', 'townhouse', 'multi-family', 'commercial']
        text_lower = text.lower()
        for prop_type in property_types:
            if prop_type in text_lower:
                return prop_type.title()
        return "Unknown"
    
    def _extract_square_footage(self, text):
        """Extract square footage"""
        import re
        patterns = [r'(\d{1,3}(?:,\d{3})*)\s*(?:sq\.?\s*ft\.?|square feet)', 
                   r'(\d{1,3}(?:,\d{3})*)\s*sf']
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).replace(',', '')
        return "Not found"
    
    def _extract_lot_size(self, text):
        """Extract lot size"""
        import re
        patterns = [r'(\d+\.?\d*)\s*acres?', r'lot size[:\s]*(\d+\.?\d*)\s*acres?']
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)} acres"
        return "Not found"
    
    def _extract_year_built(self, text):
        """Extract year built"""
        import re
        pattern = r'(?:year built|built in|construction)[:\s]*(\d{4})'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return "Not found"
    
    def _extract_appraised_value(self, text):
        """Extract appraised value"""
        import re
        patterns = [r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', 
                   r'appraised value[:\s]*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)']
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Return the largest value found (likely the appraised value)
                values = [float(match.replace(',', '')) for match in matches]
                return f"${max(values):,.2f}"
        return "Not found"
    
    def _extract_bedrooms(self, text):
        """Extract number of bedrooms"""
        import re
        pattern = r'(\d+)\s*bed(?:room)?s?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return "Not found"
    
    def _extract_bathrooms(self, text):
        """Extract number of bathrooms"""
        import re
        pattern = r'(\d+(?:\.\d+)?)\s*bath(?:room)?s?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return "Not found"
    
    def _extract_location(self, text):
        """Extract city/state information"""
        import re
        # Look for city, state pattern
        pattern = r'([A-Za-z\s]+),\s*([A-Z]{2})'
        match = re.search(pattern, text)
        if match:
            return f"{match.group(1)}, {match.group(2)}"
        return "Location not found"
    
    def _extract_appraisal_date(self, text):
        """Extract appraisal date"""
        import re
        patterns = [r'(?:appraisal date|date of appraisal)[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
                   r'(\d{1,2}/\d{1,2}/\d{4})']
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return "Date not found"
    
    def extract_text_from_txt(self, txt_file):
        """Extract text from uploaded TXT file (for testing)"""
        try:
            # Read the text file content
            content = txt_file.read().decode('utf-8')
            return content
        except Exception as e:
            st.error(f"Error reading text file: {str(e)}")
            return None
    
    def save_extracted_data(self, data, filename):
        """Save extracted data to JSON file"""
        try:
            filepath = f"data/processed/{filename}_extracted.json"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return filepath
        except Exception as e:
            st.error(f"Error saving data: {str(e)}")
            return None