import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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


url2 = 'https://raw.githubusercontent.com/seokjinwoo/streamlit-example/master/income_data.xlsx'
df3 = load_data(url2)

df3 = df3.rename(columns={
    'OJ_YY': 'year',
    'OJ_M': 'month',
    'ISMOK_NM': 'cat',
    'RV_AGGR_AMT': 'amount'
})


# 페이지 선택을 위한 사이드바 메뉴 생성
page = st.sidebar.selectbox(
    "예산 혹은 국세 진도율 선택",
    ["2024년 예산 현황", "2023년 국세 진도율", "2023년 국세 수입 금액(3D)"]
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
    
elif page == "2023년 국세 수입 금액(3D)":    
    st.title("국세 수입 추이")
    selected_cat = st.selectbox("세목:", df3['cat'].unique())
    filtered_data = df3[df3['cat'] == selected_cat]
    
    # year가 2023이 아닌 데이터에 대한 라인
    fig = px.line_3d(filtered_data[filtered_data['year'] != 2023], 
                 x='year', 
                 y='month', 
                 z='amount', 
                 color='year')

    # year가 2023인 데이터에 대한 라인 (눈에 띄게 표시)
    fig.add_trace(go.Scatter3d(
    x=filtered_data[filtered_data['year'] == 2023]['year'],
    y=filtered_data[filtered_data['year'] == 2023]['month'],
    z=filtered_data[filtered_data['year'] == 2023]['amount'],
    mode='lines',
    line=dict(color='red', width=5),
    name='2023'
    ))

    # 그래프 제목 및 축 이름 설정 및 그래프 사이즈 조정
    fig.update_layout(
        title=f"3D Line Graph for {selected_cat}",
        scene=dict(
            xaxis_title='Year',
            yaxis_title='Month',
            zaxis_title='Amount'
        ),
        width=1000,  # 그래프 너비
        height=600   # 그래프 높이
    )

    # Streamlit에서 그래프 표시
    st.plotly_chart(fig)
