import requests
import streamlit as st
import json
from datetime import datetime, timedelta

# Function to check API connection and get data
def get_api_data(keywords, days_back=2):
    try:
        url = 'https://magicloops.dev/api/loop/930723d7-bf89-4df4-8672-124b44072b80/run'
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        payload = {
            "keywords": keywords,
            "from": start_date.strftime('%Y-%m-%d'),
            "to": end_date.strftime('%Y-%m-%d')
        }
        
        # Make API request
        response = requests.get(url, json=payload)
        
        # Check if request was successful
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API request failed with status code: {response.status_code}"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from API"}

# Streamlit interface
def main():
    # Set page configuration
    st.set_page_config(page_title="Stock Prediction App", page_icon="📈")
    
    # Title and description
    st.title("Stock Market Prediction")
    st.write("Enter keywords to analyze stock market trends")
    
    # Input form
    with st.form(key='prediction_form'):
        keywords = st.text_input("Keywords", value="stock market, financial news")
        submit_button = st.form_submit_button(label='Get Prediction')
    
    # Process API request when form is submitted
    if submit_button:
        with st.spinner('Fetching data...'):
            # Get API data
            result = get_api_data(keywords)
            
            # Display results
            if "error" not in result:
                st.success("API connection successful!")
                st.subheader("API Response:")
                st.json(result)
                
                # Add your stock prediction logic here
                # For example:
                # prediction = your_prediction_function(result)
                # st.write("Stock Prediction:", prediction)
                
            else:
                st.error(f"Error: {result['error']}")
    
    # Add some basic styling
    st.markdown("""
        <style>
        .reportview-container {
            background: #f0f2f6
        }
        .main {
            background: #ffffff;
            padding: 20px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()