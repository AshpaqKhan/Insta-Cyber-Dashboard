import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def run_insta_dashboard():
    # === PAGE CONFIG ===
    st.markdown("""
        <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .main {
            background-color: #0e1117;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # === TITLE ===
    st.markdown("""
        <div style='background-color: #1f2937; padding: 20px; border-radius: 10px; text-align: center;'>
            <h1 style='color: #ffffff;'>📸 Instagram Influencer Analysis</h1>
            <p style='color: #d1d5db;'>Visualize top influencer stats, trends, and engagement prediction</p>
        </div>
    """, unsafe_allow_html=True)

    # === FILE UPLOAD ===
    uploaded_file = st.file_uploader(" Upload Cleaned Influencer CSV", type=["csv"], key="insta")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # CLEANING
        def convert(val):
            val = str(val).strip().upper().replace(",", "")
            if 'M' in val:
                return float(val.replace('M', '')) * 1_000_000
            elif 'K' in val:
                return float(val.replace('K', '')) * 1_000
            else:
                try:
                    return float(val)
                except:
                    return None

        df['followers'] = df['followers'].apply(convert)
        df['avg_likes'] = df['avg_likes'].apply(convert)
        df['new_post_avg_like'] = df['new_post_avg_like'].apply(convert)
        df['influence_score'] = df['influence_score'].apply(convert)
        df['posts'] = df['posts'].apply(convert)

        df['engagement_rate'] = df['60_day_eng_rate'].str.replace('%', '').astype(float) / 100

        # KPI METRICS
        col1, col2, col3 = st.columns(3)
        col1.metric(" Total Influencers", len(df))
        col2.metric(" Avg Followers", f"{int(df['followers'].mean()):,}")
        col3.metric(" Avg Engagement", f"{round(df['engagement_rate'].mean()*100, 2)}%")

        st.markdown("---")

        # CHARTS
        col4, col5 = st.columns([2, 1])

        with col4:
            fig1 = px.scatter(df, x="followers", y="engagement_rate", color="country",
                             size='avg_likes', hover_name='channel_info', log_x=True,
                             title="Followers vs Engagement Rate (Bubble by Likes)",
                             height=450)
            st.plotly_chart(fig1, use_container_width=True)

        with col5:
            top_countries = df['country'].value_counts().dropna().head(7)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.barplot(y=top_countries.index, x=top_countries.values, palette="coolwarm", ax=ax2)
            ax2.set_title("Top Countries by Influencers")
            ax2.set_xlabel("Number of Influencers")
            ax2.set_ylabel("Country")
            st.pyplot(fig2)

        st.markdown("---")

        # PIE CHART + BOX PLOT
        df['engagement_label'] = df['engagement_rate'].apply(lambda x: 'High (>5%)' if x > 0.05 else 'Low (<=5%)')
        col6, col7 = st.columns(2)

        with col6:
            fig3 = px.pie(df, names='engagement_label', title='Engagement Rate Classification')
            st.plotly_chart(fig3, use_container_width=True)

        with col7:
            top_country_list = df['country'].value_counts().head(6).index
            fig4 = px.box(df[df['country'].isin(top_country_list)], x='country', y='engagement_rate', color='country',
                         title='Engagement Rate by Country')
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")

        # RAW TABLE
        st.subheader(" Data Preview")
        st.dataframe(df.head(20))
