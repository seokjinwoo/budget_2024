import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt

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

url4 = 'https://raw.githubusercontent.com/seokjinwoo/budget_2024/master/Tbill_202309.xlsx'
df4 = load_data(url4)


# 페이지 선택을 위한 사이드바 메뉴 생성
page = st.sidebar.selectbox(
    "예산/국세 진도율(%)/국세 수입(조원)/재정증권",
    ["재정증권", "2023년 국세 진도율", "2023년 국세 수입 금액(3D)", "2024년 예산 현황" ]
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
    
    fig = go.Figure()

    # year별로 데이터를 필터링하여 각각의 라인을 그림
    for year in filtered_data['year'].unique():
        if year == 2023:
            fig.add_trace(go.Scatter3d(
                x=filtered_data[filtered_data['year'] == year]['year'],
                y=filtered_data[filtered_data['year'] == year]['month'],
                z=filtered_data[filtered_data['year'] == year]['amount'],
                mode='lines',
                name=str(year),
                line=dict(color='red', width=3)
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=filtered_data[filtered_data['year'] == year]['year'],
                y=filtered_data[filtered_data['year'] == year]['month'],
                z=filtered_data[filtered_data['year'] == year]['amount'],
                mode='lines',
                name=str(year),
                line=dict(color=f'rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.5)', width=2)  # 랜덤 색상 with 50% 투명도
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

elif page == "재정증권":    
        
    # Streamlit app
    st.title('연도-월 별 재정증권 발행 현황')
    st.markdown('''재정증권은 기재부가 단기차입을 목적으로 발행하는 만기 63일물 채권입니다.
        2019년 이전에는 대략 월 4조원, 코로나 기간에는 약 4.7조원, 2023년에는 약 5.3조원 발행하고 있습니다.''')

    
    # Dropdown menu for selecting y-axis variable
    y_axis_choice = st.selectbox("발행량 혹은 금리", ["발행액", "발행금리"])
    
    # Data preprocessing
    if y_axis_choice == "발행액":
        df4['year-month'] = df4['year'].astype(str) + '-' + df4['month'].astype(str).str.zfill(2)
        grouped = df4.groupby('year-month')['amount_issue'].sum().reset_index()
        grouped['amount_issue'] = grouped['amount_issue'] / 1e12
        grouped['year-month-num'] = (grouped['year-month'].str.split('-').str[0].astype(int) - 2011) * 12 + grouped['year-month'].str.split('-').str[1].astype(int)
        
        # Slider for selecting date range in numerical format
        start_num, end_num = st.slider('기간(2011년 1월-2023년 6월)', min(grouped['year-month-num']), max(grouped['year-month-num']), (min(grouped['year-month-num']), max(grouped['year-month-num'])))
    
        # Convert selected numerical range back to 'year-month' format
        filtered_data = grouped[(grouped['year-month-num'] >= start_num) & (grouped['year-month-num'] <= end_num)]
    
        # Altair Area Chart with Custom Style
        chart = alt.Chart(filtered_data).mark_area(
            line={'color':'darkgreen'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='white', offset=0),
                    alt.GradientStop(color='darkgreen', offset=1)],
                x1=1,
                x2=1,
                y1=1,
                y2=0
            )
        ).encode(
            alt.X('year-month:T', title="연도-월"),
            alt.Y('amount_issue:Q', title="발행량(조원)"),
            tooltip=['year-month', 'amount_issue']
        ).properties(
            width=700,
            height=400,
            title="Amount Issued by Year-Month"
        )
    
        st.altair_chart(chart, use_container_width=True)
        
    elif y_axis_choice == "발행금리":
      
        df4['year-month'] = df4['year'].astype(str) + '-' + df4['month'].astype(str).str.zfill(2)
        grouped = df4.groupby('year-month')['rate_issue'].mean().reset_index()
        grouped['year-month-num'] = (grouped['year-month'].str.split('-').str[0].astype(int) - 2011) * 12 + grouped['year-month'].str.split('-').str[1].astype(int)    
    
        # Slider for selecting date range in numerical format
        start_num, end_num = st.slider('기간(2011년 1월-2023년 6월)', min(grouped['year-month-num']), max(grouped['year-month-num']), (min(grouped['year-month-num']), max(grouped['year-month-num'])))
    
        # Convert selected numerical range back to 'year-month' format
        filtered_data = grouped[(grouped['year-month-num'] >= start_num) & (grouped['year-month-num'] <= end_num)]
    
        # Altair Area Chart with Custom Style
        chart = alt.Chart(filtered_data).mark_area(
            line={'color':'darkblue'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='white', offset=0),
                    alt.GradientStop(color='darkblue', offset=1)],
                x1=1,
                x2=1,
                y1=1,
                y2=0
            )
        ).encode(
            alt.X('year-month:T', title="연도-월"),
            alt.Y('rate_issue:Q', title="금리(%)"),
            tooltip=['year-month', 'rate_issue']
        ).properties(
            width=700,
            height=400,
            title="Rate Issued by Year-Month"
        )
    
        st.altair_chart(chart, use_container_width=True)
    
