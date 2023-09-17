import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('bmh')
plt.rcParams['font.family'] = 'NanumGothic'

# Load the data
def load_data(url):
    return pd.read_excel(url)  # 수정된 부분

url1 = 'https://raw.githubusercontent.com/seokjinwoo/budget_2024/master/budget_2024_treemap.xlsx'
df1 = load_data(url1)

url2 = 'https://raw.githubusercontent.com/seokjinwoo/streamlit-example/master/income_data.xlsx'
df2 = load_data(url2)

# Preprocessing
df2['ISMOK_NM'] = df2['ISMOK_NM'].str.strip()
df2 = df2.rename(columns={  # 수정된 부분
    'OJ_YY': 'year',
    'OJ_M': 'month',
    'ISMOK_NM': 'cat',
    'OUT_RT': 'pro'
})

# 페이지 선택을 위한 사이드바 메뉴 생성
page = st.sidebar.selectbox(
    "예산 혹은 국세 진도율 선택",
    ["2024년 예산 현황", "2023년 국세 진도율"]
)

# 선택된 페이지에 따라 내용 표시
if page == "2024년 예산 현황":
    st.title("2024년 예산 현황")
    st.subheader("소관(부처)-프로그램-세부사업 순")
    st.text("출처: 열린재정, 단위: 천원")
    fig = px.treemap(df1, path=['department','program','policy'], values='budget2024')
    st.write(fig)

elif page == "2023년 국세 진도율":
    st.title("국세 진도율에 대한 산포도")
    selected_cat = st.selectbox("세목:", df2['cat'].unique())
    filtered_data = df2[df2['cat'] == selected_cat]
    fig, ax = plt.subplots()
    ax.scatter(np.array(filtered_data['month']), np.array(filtered_data['pro']), alpha=0.3, label='Scatterplot')
    data_2023 = filtered_data[filtered_data['year'] == 2023]
    ax.plot(np.array(data_2023['month']), np.array(data_2023['pro']), color='magenta', label='2023')
    avg_pro_before_2023 = filtered_data[filtered_data['year'] <= 2022].groupby('month')['pro'].mean()
    ax.plot(np.array(avg_pro_before_2023.index), np.array(avg_pro_before_2023.values), 'b--', label='Average (2014-2022)')
    months_abbrev = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months_abbrev, rotation=45)
    ax.set_xlabel('')
    ax.set_ylabel('Revenue progress rate (%)')
    ax.legend()
    st.pyplot(fig)
