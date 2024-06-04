import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt

plt.style.use('bmh')
# plt.rcParams['font.family'] = 'NanumGothic'

# Load the data
def load_data(url):
    return pd.read_excel(url)

url1 = 'https://raw.githubusercontent.com/seokjinwoo/budget_2024/master/budget_2024_treemap.xlsx'
df1 = load_data(url1)

url2 = 'https://raw.githubusercontent.com/seokjinwoo/budget_2024/master/income_data.xlsx'
df2 = load_data(url2)

# Preprocessing
df2['ISMOK_NM'] = df2['ISMOK_NM'].str.strip()
df2 = df2.rename(columns={
    'OJ_YY': 'year',
    'OJ_M': 'month',
    'ISMOK_NM': 'cat',
    'OUT_RT': 'pro',
    'RV_AGGR_AMT': 'amount'
})

url3 = 'https://raw.githubusercontent.com/seokjinwoo/budget_2024/master/income_data.xlsx'
df3 = load_data(url3)

df3 = df3.rename(columns={
    'OJ_YY': 'year',
    'OJ_M': 'month',
    'ISMOK_NM': 'cat',
    'RV_AGGR_AMT': 'amount'
})

url4 = 'https://raw.githubusercontent.com/seokjinwoo/budget_2024/master/Tbill_202309.xlsx'
df4 = load_data(url4)

# Sidebar menu for page selection
page = st.sidebar.selectbox(
    "국세 진도율(%)/국세 수입(조원)/예산/재정증권",
    ["2024년 국세 진도율", "2024년 국세 수입 금액(3D)", "2024년 예산 현황", "재정증권"]
)

# Page content based on selection
if page == "2024년 예산 현황":
    st.title("2024년 예산 현황")
    st.subheader("소관(부처)-프로그램-세부사업 순")
    st.text("출처: 열린재정, 단위: 천원")
    fig = px.treemap(df1, path=['department', 'program', 'policy'], values='budget2024')
    st.write(fig)

elif page == "2024년 국세 진도율":
    st.title("국세 진도율")
    selected_cat = st.selectbox("세목:", df2['cat'].unique())
    filtered_data = df2[df2['cat'] == selected_cat]

    st.markdown("## 진도율(%)")
    st.markdown('''<span style='color:red'>진도율</span>은 예산 대비 얼마나 거쳤는지를 보는 지표입니다. 
    평균적으로 연말이 되면 100%를 초과합니다. 이는 세수 추계가 보수적으로 이루어지기 때문입니다. 
    남은 돈은 세계잉여금의 형태로 처리됩니다.''', unsafe_allow_html=True)
    
    jitter_strength = 0.1  # Adjust this value to increase or decrease the jitter
    jittered_month = filtered_data['month'] + np.random.normal(0, jitter_strength, size=len(filtered_data))

    fig, ax = plt.subplots()
    ax.scatter(jittered_month, np.array(filtered_data['pro']), alpha=0.3, label='Observed', color='#6B8E23')
    data_2024 = filtered_data[filtered_data['year'] == 2024]
    ax.plot(np.array(data_2024['month']), np.array(data_2024['pro']), color='#FF6F61', label='2024')
    avg_pro_before_2024 = filtered_data[filtered_data['year'] <= 2023].groupby('month')['pro'].mean()
    ax.plot(np.array(avg_pro_before_2024.index), np.array(avg_pro_before_2024.values), 'b--', label='Average (2014-2023)')
    months_abbrev = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months_abbrev, rotation=10)
    ax.set_xlabel('')
    ax.set_ylabel('Revenue progress rate (%)')
    ax.legend()
    st.pyplot(fig)

    
    # 구분선 추가
    st.markdown("---")
    
    # 2024년에 관측치가 있는 마지막 달 찾기
    last_month_2024 = filtered_data[filtered_data['year'] == 2024]['month'].max()
    
    # 수입 섹션
    st.markdown(f"## 연도별 {last_month_2024}월 {selected_cat} 세수(조원)")
    
    # 2024년에 관측치가 있는 마지막 달 기준 연도별 해당 세목의 수입을 막대그래프로 표현
    last_month_data = filtered_data[(filtered_data['month'] == last_month_2024)]
    
    # Create the bar plot
    fig, ax = plt.subplots()
    bars = ax.bar(last_month_data['year'], last_month_data['amount'], color=[(0.1, 0.2, 0.5, 0.1) if year != 2024 else (0.1, 0.2, 0.5, 1.0) for year in last_month_data['year']])
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Set year ticks from 2014 to 2024
    ax.set_xticks(range(2014, 2025))
    ax.set_xticklabels(range(2014, 2025), rotation=10)
    ax.set_ylabel('Amount')
    st.pyplot(fig)



elif page == "2024년 국세 수입 금액(3D)":
    st.title("국세 수입 추이")
    selected_cat = st.selectbox("세목:", df3['cat'].unique())
    filtered_data = df3[df3['cat'] == selected_cat]

    fig = go.Figure()

    # Filter data by year and plot each line
    for year in filtered_data['year'].unique():
        if year == 2024:
            fig.add_trace(go.Scatter3d(
                x=filtered_data[filtered_data['year'] == year]['year'],
                y=filtered_data[filtered_data['year'] == year]['month'],
                z=filtered_data[filtered_data['year'] == year]['amount'],
                mode='lines',
                name=str(year),
                line=dict(color='#FF6F61', width=3)
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=filtered_data[filtered_data['year'] == year]['year'],
                y=filtered_data[filtered_data['year'] == year]['month'],
                z=filtered_data[filtered_data['year'] == year]['amount'],
                mode='lines',
                name=str(year),
                line=dict(color=f'rgba({np.random.randint(50, 150)}, {np.random.randint(50, 150)}, {np.random.randint(50, 150)}, 0.5)', width=2)
            ))

    # Update layout
    fig.update_layout(
        title=f"3D Line Graph for {selected_cat}",
        scene=dict(
            xaxis_title='Year',
            yaxis_title='Month',
            zaxis_title='Amount'
        ),
        width=1000,
        height=600
    )

    st.plotly_chart(fig)

elif page == "재정증권":
    st.title('연도-월 별 재정증권 발행 현황')
    st.markdown('''<span style='color: #FF6F61;'>재정증권</span>은 기재부가 단기차입을 목적으로 발행하는 만기 63일물 채권입니다.
        2019년 이전에는 대략 월 4조원, 코로나 기간에는 약 4.7조원, 2023년에는 약 5.3조원 발행하고 있습니다.''', unsafe_allow_html=True)

    y_axis_choice = st.selectbox("발행량 혹은 금리", ["발행액", "발행금리"])

    if y_axis_choice == "발행액":
        df4['year-month'] = df4['year'].astype(str) + '-' + df4['month'].astype(str).str.zfill(2)
        grouped = df4.groupby('year-month')['amount_issue'].sum().reset_index()
        grouped['amount_issue'] = grouped['amount_issue'] / 1e12
        grouped['year-month-num'] = (grouped['year-month'].str.split('-').str[0].astype(int) - 2011) * 12 + grouped['year-month'].str.split('-').str[1].astype(int)

        start_num, end_num = st.slider('기간(2011년 1월-2023년 6월)', min(grouped['year-month-num']), max(grouped['year-month-num']), (min(grouped['year-month-num']), max(grouped['year-month-num'])))

        filtered_data = grouped[(grouped['year-month-num'] >= start_num) & (grouped['year-month-num'] <= end_num)]

        chart = alt.Chart(filtered_data).mark_area(
            line={'color': 'darkgreen'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='darkgreen', offset=1)],
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

        start_num, end_num = st.slider('기간(2011년 1월-2023년 6월)', min(grouped['year-month-num']), max(grouped['year-month-num']), (min(grouped['year-month-num']), max(grouped['year-month-num'])))

        filtered_data = grouped[(grouped['year-month-num'] >= start_num) & (grouped['year-month-num'] <= end_num)]

        chart = alt.Chart(filtered_data).mark_area(
            line={'color': 'darkblue'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='darkblue', offset=1)],
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
