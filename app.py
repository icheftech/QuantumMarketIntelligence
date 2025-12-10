# app.py
"""
================================================================================
QUANTUM MARKET INTELLIGENCE PLATFORM - MVP UI
================================================================================
Streamlit front-end for financial news sentiment + LLM insights.
"""

import streamlit as st
import pandas as pd

import config
from sentiment_engine import FinancialSentimentAnalyzer, get_sentiment_color
import data_ingestion
import llm_engine


@st.cache_resource
def load_sentiment_engine():
    engine = FinancialSentimentAnalyzer()
    if not engine.is_trained:
        engine.train_model()
    return engine


def derive_signal(score: float) -> str:
    t = config.SIGNAL_THRESHOLDS
    if score <= t["strong_sell"]:
        return "STRONG SELL"
    if score <= t["sell"]:
        return "SELL"
    if score >= t["strong_buy"]:
        return "STRONG BUY"
    if score >= t["buy"]:
        return "BUY"
    return "HOLD"


def main():
    st.set_page_config(
        page_title="Quantum Market Intelligence Platform",
        layout="wide",
    )

    st.title("📊 Quantum Market Intelligence Platform")
    st.caption(
        "Institutional-grade market intelligence prototype – built for Module 04: AI in Finance"
    )

    with st.expander("📌 Business Use Case & Value Proposition", expanded=False):
        st.markdown(
            """
**Target Client:**  
World's largest hedge funds & institutional investors seeking edge in global markets.

**Challenge:**  
Traditional research is slow, narrow, expensive, and biased. Human analysts cannot
monitor every asset, every region, and every news event in real time.

**Our Solution – Quantum Market Intelligence Platform**  

- **Omniscient Data Collection:** Aggregates global financial news, macro data, and sentiment.
- **AI-Powered Real-Time Analysis:** Hybrid NLP + ML sentiment at scale.
- **Autonomous Decision Intelligence:** Converts signals into actionable buy/sell/hold views.
- **Continuous Learning:** System improves as more market cycles pass.
"""
        )

    st.markdown("### 1️⃣ Load Market News")

    col_left, col_right = st.columns([1.2, 1.8])

    with col_left:
        query = st.text_input(
            "News keyword (for live API; offline samples if no key):",
            value="markets",
        )
        n_articles = st.slider("Number of articles (if using API)", 3, 50, 10)
        refresh = st.button("🔄 Fetch Latest News")

    if refresh:
        st.session_state["news_df"] = data_ingestion.fetch_news_from_rss(query, n_articles)

    news_df: pd.DataFrame = st.session_state.get(
        "news_df", data_ingestion.fetch_news_from_rss(query, n_articles)
    )

    with col_right:
        st.subheader("News Feed")
        if not news_df.empty:
            st.dataframe(
                news_df[["source", "title", "published_at"]],
                use_container_width=True,
                height=250,
            )
        else:
            st.warning("No news articles available.")

    st.markdown("---")
    st.markdown("### 2️⃣ Sentiment Analysis & Trading Signal")

    engine = load_sentiment_engine()

    if not news_df.empty:
        titles = news_df["title"].tolist()
        selected_idx = st.selectbox(
            "Select an article to analyze:",
            options=list(range(len(titles))),
            format_func=lambda i: titles[i][:150],
        )
        selected_row = news_df.iloc[selected_idx]
        article_text = (
            selected_row.get("content") or selected_row.get("description") or selected_row.get("title")
        )
    else:
        selected_row = None
        article_text = ""

    col_a, col_b = st.columns([1.4, 1.6])

    with col_a:
        st.subheader("Article Text")
        st.text_area(
            "You can also edit the article text before analysis:",
            value=article_text,
            height=200,
            key="article_text_input",
        )
        analyze_btn = st.button("🔍 Run Sentiment & Signal")

    with col_b:
        if analyze_btn and st.session_state.get("article_text_input", "").strip():
            text = st.session_state["article_text_input"]
            result = engine.analyze(text)

            score = result["score"]
            label = result["label"]
            emoji = result["emoji"]
            method = result["method"]

            signal = derive_signal(score)
            color = get_sentiment_color(score)

            st.markdown("#### Sentiment Result")
            st.markdown(
                f"""
<div style="border-radius:10px;padding:1rem;border:1px solid #ddd;
background:##000000;">
<b>Sentiment:</b> {emoji} <b>{label.upper()}</b><br>;color:#FFFFFF
<b>Score:</b> {score:.3f}<br>;color:#FFFFFF
<b>Method:</b> {method}<br>;color:#FFFFFF
<b>Signal:</b> <span style="color:{color};font-weight:bold;color:#FFFFFF;">{signal}</span>
</div>
""",
                unsafe_allow_html=True,
            )

            st.caption(
            "<span style='color:#CCCCCC;'>⚠️ Signal is a simple mapping of sentiment score → STRONG SELL / SELL / HOLD / BUY / STRONG BUY "                "for demo purposes (not financial advice)."
            "for demo purposes (not financial advice).</span>",
    st.markdown("### 3️⃣ LLM Insights: Summary & Q&A")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📌 LLM Summary")
        if analyze_btn and st.session_state.get("article_text_input", "").strip():
            with st.spinner("Calling LLM for summary..."):
                summary = llm_engine.generate_summary(st.session_state["article_text_input"])
            st.write(summary)
        else:
            st.info("Run sentiment analysis first to generate a summary.")

    with col2:
        st.markdown("#### ❓ Ask a Question About This Article")
        question = st.text_input(
            "Example: 'What is the overall outlook on this company?' or 'What risks are mentioned?'",
            value="",
        )
        ask_btn = st.button("🧠 Ask LLM")

        if ask_btn:
            if not st.session_state.get("article_text_input", "").strip():
                st.warning("Please analyze an article first.")
            elif not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Calling LLM for Q&A..."):
                    answer = llm_engine.answer_question(
                        st.session_state["article_text_input"], question
                    )
                st.write(answer)

    st.markdown("---")
    st.markdown("### 4️⃣ Regulatory & Risk Disclaimer")
    st.info(config.DISCLAIMER)


if __name__ == "__main__":
    main()
