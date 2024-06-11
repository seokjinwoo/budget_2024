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
    'OUT_RT': 'pro',
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


    # 2024년도에 관측된 월 리스트 생성
    months_2024 = sorted(df2[df2['year'] == 2024]['month'].unique(), reverse=True)


    # Streamlit 앱
    st.title("국세 진도율")

    st.markdown('''<span style='color:red'>진도율</span>은 예산 대비 얼마나 거쳤는지를 보는 지표입니다. 
    평균적으로 연말이 되면 100%를 초과합니다. 이는 세수 추계가 보수적으로 이루어지기 때문입니다. 
    남은 돈은 세계잉여금의 형태로 처리됩니다.''', unsafe_allow_html=True)
    
    # 세목 선택
    selected_cat = st.selectbox("세목:", df2['cat'].unique())
    filtered_data = df2[df2['cat'] == selected_cat]

    # 2024년도에 관측된 월 선택
    selected_month = st.selectbox("2024년도 월을 선택하세요:", months_2024)


    # 평균 진도율과 2024년 진도율 계산
    avg_progress_rate = filtered_data[(filtered_data['year'] <= 2023) & (filtered_data['month'] == selected_month)]['pro'].mean()
    progress_rate_2024 = filtered_data[(filtered_data['year'] == 2024) & (filtered_data['month'] == selected_month)]['pro'].mean()
    
    # 평균 진도율과 2024년 진도율을 큰 글씨로 표시
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background-color: #B0C4DE; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: #2F4F4F;">{selected_month}월 평균 진도율</h2>
            <p style="font-size: 30px; color: #2F4F4F;">{avg_progress_rate:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background-color: #FFD700; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: #8B0000;">2024년 {selected_month}월 진도율</h2>
            <p style="font-size: 30px; color: #8B0000;">{progress_rate_2024:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)


    
    st.markdown(f"## {selected_cat } 진도율(%)")


    # Adding jitter to the month values for better visualization
    jitter_strength = 0.1
    jittered_month = filtered_data['month'] + np.random.normal(0, jitter_strength, size=len(filtered_data))

    fig = go.Figure()

    # Scatter plot for observed data with jitter
    fig.add_trace(go.Scatter(
        x=jittered_month,
        y=filtered_data['pro'],
        mode='markers',
        marker=dict(color='rgba(107, 142, 35, 0.3)', size=13, line=dict(width=0, color='DarkSlateGrey')),
        name='Observed'
    ))

    # Line plot for 2024 data up to the selected month
    data_2024 = filtered_data[(filtered_data['year'] == 2024) & (filtered_data['month'] <= selected_month)]
    fig.add_trace(go.Scatter(
        x=data_2024['month'],
        y=data_2024['pro'],
        mode='lines+markers',
        line=dict(color='red', width=3),
        name='2024'
    ))

    # Average progress rate from previous years (2014-2023)
    avg_pro_before_2024 = filtered_data[filtered_data['year'] <= 2023].groupby('month')['pro'].mean()
    fig.add_trace(go.Scatter(
        x=avg_pro_before_2024.index,
        y=avg_pro_before_2024.values,
        mode='lines',
        line=dict(color='blue', dash='dash', width=2),
        name='Average (2014-2023)'
    ))

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ),
        yaxis_title='Revenue progress rate (%)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=600,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)


    # 구분선 추가
    st.markdown("---")

    # 수입 섹션
    st.markdown(f"## 연도별 {selected_month}월 {selected_cat} 세수(조원)")

    # 선택된 월 기준 연도별 해당 세목의 수입을 막대그래프로 표현
    last_month_data = df2[(df2['month'] == selected_month) & (df2['cat'] == selected_cat)]

    # Create the bar plot
    colors = ['#AEC6CF', '#FFB7B2', '#C1E1C1', '#FFDAC1', '#B39EB5', '#FFDFD3', '#C6E2FF', '#E6E6FA', '#FFD1DC', '#F5E6CC']

    bar_fig = go.Figure()

    bar_fig.add_trace(go.Bar(
        x=last_month_data['year'],
        y=last_month_data['amount'],
        marker=dict(
            color=[colors[i % len(colors)] for i in range(len(last_month_data))]
        ),
        opacity=1.0
    ))

    # Add value labels on the bars
    for i, row in last_month_data.iterrows():
        bar_fig.add_annotation(
            x=row['year'],
            y=row['amount'],
            text=f'{row["amount"]:.1f}',
            showarrow=False,
            yshift=10,
            font=dict(color='black')
        )

    bar_fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(2014, 2025)),
            ticktext=list(range(2014, 2025)),
            title='Year'
        ),
        yaxis_title='Amount',
        showlegend=False,
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="DarkSlateGrey"
        )
    )

    st.plotly_chart(bar_fig, use_container_width=True)


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
