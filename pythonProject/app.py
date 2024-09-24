from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe


def chat_with_csv(df, query):
    llm = LocalLLM(
        api_base="http://localhost:11434/v1",
        model="llama3"
    )
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    result1 = pandas_ai.chat(query)
    return result1


st.set_page_config(layout='wide')

st.title("Multiple-CSV ChatApp powered by LLM")

# Upload CSV files
input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# Check if any CSVs are uploaded
if input_csvs:
    # Select file from uploaded CSVs
    selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)

    st.info("CSV uploaded successfully")

    # Read the selected CSV file
    data = pd.read_csv(input_csvs[selected_index])

    # Show first 3 rows of the selected CSV
    st.dataframe(data.head(3), use_container_width=True)

    # Input for query
    st.info("Chat Below")
    input_text = st.text_area("Enter the query")

    if input_text:
        if st.button("Chat with CSV"):
            st.info("Your Query: " + input_text)
            result = chat_with_csv(data, input_text)
            st.success(result)
else:
    st.warning("Please upload at least one CSV file.")
