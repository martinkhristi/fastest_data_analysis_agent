import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.langchain import LangchainLLM
from langchain_openai import ChatOpenAI
import time

# Set up Streamlit page
st.set_page_config(
    page_title="The World's Fastest Advanced Data Analysis Agent with SambaNova API and PandasAI",
    layout="wide",
)

# Add custom styling
st.markdown(
    """
    <style>
    body {
        background-color: #F5F5F5;
    }
    .css-1bc7jzt {
        color: #0072C6;
    }
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
        color: #333333;
    }
    .stButton button {
        background-color: #0072C6;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add a powerful introductory prompt
st.markdown(
    """
    <div style='background-color: #0072C6; padding: 20px; border-radius: 10px; color: #FFFFFF; text-align: center;'>
        <h2>Welcome to Smart Data Analysis Platform</h2>
        <p>I am an experienced Python expert and Data Scientist here to assist you in analyzing and visualizing your data effortlessly. 
        Upload your CSV or Excel file, ask questions in plain English, and let the power of AI provide insights!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Function to authenticate SambaNova API using Langchain
@st.cache_resource
def authenticate_sambanova(api_key):
    return ChatOpenAI(
        base_url="https://api.sambanova.ai/v1/",
        api_key=api_key,
        streaming=False,
        model="Meta-Llama-3.1-70B-Instruct",
    )

# Sidebar for SambaNova API authentication
st.sidebar.title("Authentication")
sambanova_api_key = st.sidebar.text_input(
    "Enter your SambaNova API Key:", type="password"
)

if sambanova_api_key:
    try:
        # Authenticate with the provided key
        sambanova_llm = authenticate_sambanova(sambanova_api_key)
        langchain_llm = LangchainLLM(sambanova_llm)  # Wrap SambaNova LLM for PandasAI
        st.sidebar.success("Successfully authenticated!")
    except Exception as e:
        st.sidebar.error(f"Authentication failed: {e}")
        st.stop()
else:
    st.sidebar.info("Please enter your SambaNova API key to proceed.")
    st.stop()

# Sidebar for file upload
st.sidebar.title("Upload File")
file_type = st.sidebar.radio("Select file type", ("CSV", "Excel"))
uploaded_file = st.sidebar.file_uploader(f"Upload a {file_type} file", type=["csv", "xls", "xlsx"])

# Main application content
if uploaded_file is not None:
    try:
        # Load data based on file type
        if file_type == "CSV":
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file, engine="openpyxl")

        # Display data preview
        st.subheader("Data Preview")
        st.write(data.head())

        # Display general information about the data
        st.subheader("General Information")
        st.write(f"Shape of the dataset: {data.shape}")
        st.write(f"Data Types:\n{data.dtypes}")
        st.write(f"Memory Usage: {data.memory_usage(deep=True).sum()} bytes")

        # Initialize SmartDataframe with SambaNova LLM via PandasAI
        df_smart = SmartDataframe(data, config={"llm": langchain_llm})

        # User input for natural language query
        query = st.text_input(
            "Ask me anything about your data, and I'll provide clear insights or visualizations:",
            placeholder="For example, 'Show me the average sales per month' or 'Visualize the top 5 products by revenue.'"
        )

        if query:
            try:
                # Query processing
                start_time = time.time()
                response = df_smart.chat(query)  # PandasAI handles the query
                end_time = time.time()

                # Display response
                st.subheader("AI Response")
                st.write(response)
                st.success(f"Query processed in {end_time - start_time:.2f} seconds.")
            except Exception as e:
                st.error(f"An error occurred during query processing: {e}")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Footer
st.markdown(
    """
    <footer style='text-align: center; padding: 10px; background-color: #0072C6; color: #FFFFFF;'>
       Powered by SambaNova API, PandasAI, and Open Source Tools<br>
        Made with ❤️ by Martin Khristi
    </footer>
    """,
    unsafe_allow_html=True,
)


