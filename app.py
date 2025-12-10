# app.py
"""
================================================================================
QUANTUM MARKET INTELLIGENCE PLATFORM - MVP UI
================================================================================
Netflix-quality Streamlit front-end for financial news sentiment + LLM insights.
"""

import streamlit as st
import pandas as pd
import re

import config
from sentiment_engine import FinancialSentimentAnalyzer, get_sentiment_color, extract_financial_entities
import data_ingestion
import llm_engine

# Custom CSS for Netflix-quality UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    /* Card styling with glassmorphism */
    .sentiment-card {
        background: rgba(30, 35, 55, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    
    .sentiment-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Signal badge styling */
    .signal-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #00C853 0%, #00E676 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(0, 200, 83, 0.4);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #FF1744 0%, #FF5252 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(255, 23, 68, 0.4);
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #FFB300 0%, #FFC107 100%);
        color: #000;
        box-shadow: 0 4px 20px rgba(255, 179, 0, 0.4);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Entity badges */
    .entity-badge {
        display: inline-block;
        background: rgba(33, 150, 243, 0.15);
        border: 1px solid rgba(33, 150, 243, 0.3);
        color: #64B5F6;
        padding: 4px 12px;
        border-radius: 12px;
        margin: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Progress bar styling */
    .confidence-bar {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 8px;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #00C853 0%, #64DD17 100%);
        transition: width 0.8s ease;
        box-shadow: 0 0 10px rgba(0, 200, 83, 0.5);
    }
    
    /* Section headers */
    .section-header {
        font-size: 24px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 16px;
    }
    
    /* News table hover effect */
    .stDataFrame tbody tr:hover {
        background: rgba(255, 255, 255, 0.05);
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

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

def get_signal_class(signal: str) -> str:
    """Get CSS class for signal badge"""
    if "BUY" in signal:
        return "signal-buy"
    elif "SELL" in signal:
        return "signal-sell"
    else:
        return "signal-hold"

def main():
    st.set_page_config(
        page_title="Quantum Market Intelligence Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Header with gradient
    st.markdown('<h1 style="text-align:center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 48px; font-weight: 800; margin-bottom: 8px;">📊 Quantum Market Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color: #9CA3AF; font-size: 16px; margin-bottom: 32px;">Institutional-grade market intelligence powered by AI • Built for Module 04: AI in Finance</p>', unsafe_allow_html=True)
    
    # Business use case expandable
    with st.expander("🎯 Platform Overview & Value Proposition", expanded=False):
        st.markdown("""
        **Target:** Hedge funds, trading desks, and institutional investors seeking real-time market intelligence
        
        **Core Capabilities:**
        - 📰 Multi-source financial news aggregation
        - 🧠 Hybrid sentiment analysis (VADER + ML)
        - 🎯 Entity extraction & contextual recommendations
        - 💡 LLM-powered actionable insights
        - ⚡ Real-time signal generation
        
        **Competitive Edge:** 
        Analyzes every article in seconds, identifies specific trading opportunities, and delivers institutional-grade insights at 90% lower cost than traditional analyst teams.
        """)
    
    # Load sentiment engine
    engine = load_sentiment_engine()
    
    # Section 1: News Feed
    st.markdown('<p class="section-header">1️⃣ Market News Feed</p>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.2, 1.8])
    
    with col_left:
        query = st.text_input("News keyword:", value="markets", help="Search term for live API; uses offline samples if no key")
        n_articles = st.slider("Articles:", 3, 50, 10)
        refresh = st.button("🔄 Fetch Latest News", type="primary")
    
    if refresh or "news_df" not in st.session_state:
        st.session_state["news_df"] = data_ingestion.get_financial_news(query, n_articles)
    
    news_df = st.session_state.get("news_df", data_ingestion.get_financial_news(query, n_articles))
    
    with col_right:
        if not news_df.empty:
            st.dataframe(
                news_df[["source", "title", "published_at"]], 
                use_container_width=True,
                height=280
            )
        else:
            st.warning("No news available.")
    
    st.markdown("---")
    
    # Section 2: Sentiment Analysis
    st.markdown('<p class="section-header">2️⃣ Sentiment Analysis & Trading Signal</p>', unsafe_allow_html=True)
    
    if not news_df.empty:
        titles = news_df["title"].tolist()
        selected_idx = st.selectbox(
            "Select article to analyze:",
            options=list(range(len(titles))),
            format_func=lambda i: titles[i][:120],
        )
        selected_row = news_df.iloc[selected_idx]
        article_text = (
            selected_row.get("content") or 
            selected_row.get("description") or 
            selected_row.get("title")
        )
    else:
        article_text = ""
    
    col_a, col_b = st.columns([1.2, 1.8])
    
    with col_a:
        st.subheader("Article Text")
        edited_text = st.text_area(
            "Edit article text if needed:",
            value=article_text,
            height=200,
            key="article_text_input"
        )
        analyze_btn = st.button("🔍 Run Sentiment & Signal", type="primary")
    
    with col_b:
        if analyze_btn and st.session_state.get("article_text_input", "").strip():
            text = st.session_state["article_text_input"]
            
            # Run sentiment analysis
            result = engine.analyze(text)
            
            # Extract entities
            entities = extract_financial_entities(text)
            
            # Store in session state
            st.session_state["sentiment_result"] = result
            st.session_state["entities"] = entities
            st.session_state["article_analyzed"] = text
        
        # Display results from session state (persists)
        if "sentiment_result" in st.session_state:
            result = st.session_state["sentiment_result"]
            entities = st.session_state.get("entities", {})
            
            score = result["score"]
            label = result["label"]
            emoji = result["emoji"]
            method = result["method"]
            confidence = result["confidence"]
            signal = derive_signal(score)
            signal_class = get_signal_class(signal)
            
            # Sentiment card with Netflix-style design
            st.markdown(f"""
            <div class="sentiment-card">
                <h3 style="margin-top:0;">Sentiment Analysis Results</h3>
                <p style="font-size:18px;margin-bottom:8px;">
                    <strong>Sentiment:</strong> {emoji} <span style="color:{get_sentiment_color(score)};font-weight:700;font-size:20px;">{label.upper()}</span>
                </p>
                <p><strong>Score:</strong> {score:.3f} | <strong>Method:</strong> {method}</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{confidence*100}%;"></div>
                </div>
                <p style="font-size:12px;color:#9CA3AF;margin-top:4px;">Confidence: {confidence*100:.1f}%</p>
                
                <div style="margin-top:20px;">
                    <p style="font-size:14px;margin-bottom:8px;"><strong>Trading Signal:</strong></p>
                    <span class="signal-badge {signal_class}">{signal}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Entity display
            if any(entities.values()):
                st.markdown("**🏢 Identified Entities:**")
                entity_html = ""
                if entities.get("companies"):
                    entity_html += "".join([f'<span class="entity-badge">🏢 {c}</span>' for c in entities["companies"][:5]])
                if entities.get("tickers"):
                    entity_html += "".join([f'<span class="entity-badge">📈 {t}</span>' for t in entities["tickers"][:5]])
                if entities.get("currencies"):
                    entity_html += "".join([f'<span class="entity-badge">💱 {c}</span>' for c in entities["currencies"]])
                if entities.get("sectors"):
                    entity_html += "".join([f'<span class="entity-badge">🏭 {s}</span>' for s in entities["sectors"]])
                
                st.markdown(entity_html, unsafe_allow_html=True)
            
            st.caption("⚠️ Trading signals are algorithmic outputs for educational purposes only. Not financial advice.")
        
        elif analyze_btn:
            st.warning("Please paste or select an article before clicking *Analyze*.")
    
    st.markdown("---")
    
    # Section 3: LLM Insights
    st.markdown('<p class="section-header">3️⃣ AI-Powered Trading Insights</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📌 Article Summary")
        if st.button("✨ Generate Summary"):
            if "article_analyzed" in st.session_state:
                with st.spinner("Generating summary..."):
                    summary = llm_engine.generate_summary(st.session_state["article_analyzed"])                    summary = llm_engine.generate_summary(st.session_state["article_analyzed"])
                    st.session_state["summary"] = summary
            
            if "summary" in st.session_state:
                st.info(st.session_state["summary"])
            else:
                st.write("Click button above to generate AI summary")
    
    with col2:
        st.markdown("#### 💡 Trading Recommendations")
        if st.button("🎯 Generate Trading Ideas"):
            if "sentiment_result" in st.session_state and "entities" in st.session_state:
                with st.spinner("Generating recommendations..."):
                    suggestions = llm_engine.generate_trading_suggestions(
                        st.session_state["article_analyzed"],
                        st.session_state["sentiment_result"]["score"],
                        st.session_state["entities"]
                    )
                    st.session_state["suggestions"] = suggestions
            
            if "suggestions" in st.session_state:
                st.markdown(st.session_state["suggestions"])
            else:
                st.write("Click button above to get AI-powered trading ideas")
    
    # Q&A Section
    st.markdown("---")
    st.markdown("#### ❓ Ask Questions About This Article")
    
    question = st.text_input(
        "Your question:",
        placeholder="e.g., 'What companies are mentioned?' or 'What are the key risks?'"
    )
    
    if st.button("🧠 Ask AI") and question.strip():
        if "article_analyzed" in st.session_state:
            with st.spinner("Analyzing..."):
                answer = llm_engine.answer_question(
                    st.session_state["article_analyzed"],
                    question
                )
                st.session_state["last_qa"] = {"q": question, "a": answer}
        else:
            st.warning("Please analyze an article first.")
    
    if "last_qa" in st.session_state:
        qa = st.session_state["last_qa"]
        st.markdown(f"**Q:** {qa['q']}")
        st.markdown(f"**A:** {qa['a']}")
    
    st.markdown("---")
    
    # Disclaimer
    st.markdown('<p class="section-header">4️⃣ Regulatory & Risk Disclaimer</p>', unsafe_allow_html=True)
    st.warning(config.DISCLAIMER)

if __name__ == "__main__":
    main()

