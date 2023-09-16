import streamlit as st
import plotly.express as px
import pandas as pd

# Load the data
def load_data():
    return pd.read_excel('https://raw.githubusercontent.com/seokjinwoo/budget_2024/master/budget_2024_treemap.xlsx')




df = load_data()

# Create a treemap
fig = px.treemap(df, path=['department','program'], values='budget2024')



# Display the treemap in Streamlit
st.write(fig)