# sentiment_engine.py
"""
================================================================================
SENTIMENT ANALYSIS ENGINE
================================================================================
Hybrid VADER + ML sentiment for financial text
"""

import re
import string

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import config

# ---------------------------------------------------------------------
# NLTK downloads (safe if called multiple times)
# ---------------------------------------------------------------------
try:
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception:
    pass


class TextPreprocessor:
    """Advanced text preprocessing for financial news."""

    def __init__(self):
        try:
            self.stopwords = set(nltk.corpus.stopwords.words("english"))
        except LookupError:
            # If stopwords not downloaded, just skip removal
            self.stopwords = set()
        self.finance_keeps = {"up", "down", "above", "below", "over", "under", "more", "less"}
        self.stopwords = self.stopwords - self.finance_keeps

    def clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Remove emails
        text = re.sub(r"\S+@\S+", "", text)
        # Keep letters, digits, %, $, and whitespace
        text = re.sub(r"[^0-9a-zA-Z%\$€£¥\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def tokenize(self, text):
        try:
            return nltk.word_tokenize(text)
        except LookupError:
            # If punkt isn’t available, fall back to simple split
            return text.split()

    def remove_stopwords(self, tokens):
        if not self.stopwords:
            return tokens
        return [t for t in tokens if t not in self.stopwords]

    def preprocess(self, text):
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return " ".join(tokens)


class FinancialSentimentAnalyzer:
    """
    Hybrid sentiment engine:
    - VADER (rule-based)
    - Naive Bayes classifier (TF-IDF) when trained
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vader = SentimentIntensityAnalyzer()
        self.ml_model = None
        self.vectorizer = None
        self.is_trained = False
        self._load_model()

    def _load_model(self):
        """Try loading existing model/vectorizer; fall back to VADER only."""
        try:
            self.ml_model = joblib.load(config.SENTIMENT_MODEL_PATH)
            self.vectorizer = joblib.load(config.VECTORIZER_PATH)
            self.is_trained = True
            print("[Sentiment] Loaded pre-trained model.")
        except Exception:
            print("[Sentiment] No pre-trained model found. VADER-only mode.")
            self.is_trained = False

    def train_model(self, training_data_path=None):
        """
        Train the ML sentiment model.
        If training_data.csv is missing or broken, create sample data.
        """
        if training_data_path is None:
            training_data_path = config.TRAINING_DATA_PATH

        try:
            df = pd.read_csv(training_data_path)
            if df.empty or "text" not in df.columns or "label" not in df.columns:
                raise ValueError("training_data.csv is empty or malformed.")
        except Exception:
            print("[Sentiment] No valid training_data.csv found, creating sample training data.")
            df = create_sample_training_data()
            df.to_csv(training_data_path, index=False)

        df["cleaned"] = df["text"].apply(self.preprocessor.preprocess)

        X = df["cleaned"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        self.ml_model = MultinomialNB(alpha=0.1)
        self.ml_model.fit(X_train_tfidf, y_train)

        y_pred = self.ml_model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"[Sentiment] Model trained. Accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))

        joblib.dump(self.ml_model, config.SENTIMENT_MODEL_PATH)
        joblib.dump(self.vectorizer, config.VECTORIZER_PATH)
        self.is_trained = True
        return accuracy

    def analyze_vader(self, text):
        scores = self.vader.polarity_scores(text)
        return scores["compound"]

    def analyze_ml(self, text):
        if not self.is_trained:
            return None, 0.0

        cleaned = self.preprocessor.preprocess(text)
        vectorized = self.vectorizer.transform([cleaned])

        prediction = self.ml_model.predict(vectorized)[0]
        proba = self.ml_model.predict_proba(vectorized)[0]
        confidence = float(np.max(proba))

        score_map = {"negative": -0.8, "neutral": 0.0, "positive": 0.8}
        score = score_map.get(prediction, 0.0)
        return score, confidence

    def analyze(self, text):
        """Return dict with sentiment score + label + method."""

        vader_score = self.analyze_vader(text)
        ml_score, ml_confidence = self.analyze_ml(text)

        if ml_score is not None:
            final_score = 0.4 * vader_score + 0.6 * ml_score
            method = "hybrid"
            confidence = ml_confidence
        else:
            final_score = vader_score
            method = "vader"
            confidence = abs(vader_score)

        if final_score < config.SENTIMENT_THRESHOLDS["negative"]:
            label = "negative"
            emoji = "📉"
        elif final_score > config.SENTIMENT_THRESHOLDS["positive"]:
            label = "positive"
            emoji = "📈"
        else:
            label = "neutral"
            emoji = "➡️"

        return {
            "score": float(final_score),
            "label": label,
            "emoji": emoji,
            "confidence": float(confidence),
            "method": method,
            "vader_score": float(vader_score),
            "ml_score": None if ml_score is None else float(ml_score),
        }

    def batch_analyze(self, texts):
        return [self.analyze(t) for t in texts]


def get_sentiment_color(score):
    if score < -0.2:
        return config.COLOR_NEGATIVE
    if score > 0.2:
        return config.COLOR_POSITIVE
    return config.COLOR_NEUTRAL


def create_sample_training_data():
    """Small labeled dataset for demo / assignment."""
    texts = [
        "Stock market rallies to record highs on strong earnings.",
        "Markets plunge amid deep recession fears.",
        "Federal Reserve holds interest rates steady.",
        "Tech stocks surge as AI demand increases.",
        "Banking crisis triggers broad market selloff.",
        "Dollar strengthens against major currencies.",
        "Unemployment rate falls to lowest level in decades.",
        "Inflation concerns weigh heavily on investor sentiment.",
        "Merger announcement boosts both companies' share prices.",
        "Profit warnings from major firms drag down indices.",
        "Central bank maintains accommodative policy stance.",
        "Corporate profits soar amid economic recovery.",
        "Supply chain disruptions weigh on manufacturing sector.",
        "Housing market shows signs of cooling after rapid growth.",
        "Consumer confidence hits multi-year high.",
        "Energy prices surge on geopolitical tensions.",
        "Labor market tightens as unemployment drops.",
        "Bond yields rise on inflation expectations.",
        "Equity markets retreat from record highs.",
        "Trade negotiations progress between major economies.",
        "Cryptocurrency rally sparks regulatory concerns.",
        "Retail sales exceed forecasts during holiday season.",
        "Manufacturing PMI indicates expansion continues.",
        "Financial sector faces headwinds from new regulations.",
        "Agricultural commodities decline on favorable weather.",
        "Real estate investment trusts attract strong demand.",
        "Global economic outlook remains uncertain.",
        "Central bank signals potential policy shift.",
        "Tech sector leads market gains on innovation.",
        "Emerging markets show resilience amid volatility.",
        "Tech giants report better-than-expected quarterly revenue growth.",
        "Small cap stocks outperform amid economic optimism.",
        "Market volatility increases on regulatory uncertainty.",
        "Oil prices stabilize after weeks of dramatic swings.",
        "Banking sector rebounds following stress test results.",
        "Semiconductor shortage eases, boosting production forecasts.",
        "Gold prices drop as investors move to riskier assets.",
        "Healthcare stocks climb on breakthrough drug approval.",
        "Retail earnings disappoint, signaling consumer pullback.",
        "Currency markets react to unexpected policy announcement.",
        "Renewable energy investments surge to record levels.",
        "Corporate debt concerns weigh on credit markets.",
        "Dividend-paying stocks attract safety-seeking investors.",
        "Merger activity accelerates in telecommunications sector.",
        "Export growth slows amid trade policy changes.",
        "Inflation data comes in line with expectations.",
        "Emerging markets show mixed performance this quarter.",
        "Insurance companies face challenges from natural disasters.",
        "Logistics companies benefit from e-commerce boom.",
        "Construction spending increases on infrastructure projects.",
        "Pharmaceutical stocks rally on positive trial results.",
        "Commercial real estate market shows signs of stabilization.",
        "Federal deficit projections raise long-term growth concerns.",
        "Automotive industry struggles with supply chain disruptions.",
        "Financial technology adoption accelerates across industries.",
        "Consumer spending remains robust despite economic headwinds.",
        "Interest rate speculation drives bond market movements.",
        "Manufacturing output contracts for second consecutive month.",
        "Private equity activity reaches multi-year peak.",
        "Venture capital funding flows to artificial intelligence startups.",
        "Luxury goods sector experiences unexpected demand surge.",
        "Airline industry recovery continues at moderate pace.",
        "Gaming and entertainment stocks benefit from digital trends.",
        "Biotech sector faces headwinds from regulatory changes.",
        "Regional banks outperform national competitors this quarter.",
        "Food and beverage companies report steady sales growth.",
        "Telecommunications infrastructure investments increase significantly.",
        "Carbon credit markets expand amid climate policy push.",
        "Shipping costs decline as capacity constraints ease.",
        "Defense contractors secure major government contracts.",
        "Precious metals market reacts to inflation concerns.",
        "Electric vehicle sales exceed industry projections.",
        "Cloud computing revenue growth remains strong.",
        "Agricultural commodity prices fluctuate on weather patterns.",
        "Cybersecurity spending increases across all sectors.",
        "Pension funds adjust strategies amid rate environment.",
        "Media companies navigate changing consumption habits.",
        "Industrial metals demand softens in key markets.",
        "Specialty chemicals sector reports mixed results.",
        "Water infrastructure investments gain political support.",
        "Consumer electronics demand moderates after pandemic surge.",
        "Payment processing companies expand market share.",
        "Recreational vehicle sales decline from peak levels.",
        "Satellite communications market attracts new entrants.",
        "Dental and medical device makers report steady performance.",
        "Fast food chains announce expansion plans.",
        "Fertilizer prices stabilize after volatile period.",
        "Data center construction accelerates worldwide.",
        "Apparel retailers face margin pressure from costs.",
        "Alternative energy storage solutions gain traction.",
        "Packaging industry adapts to sustainability demands.",
        "Hotel occupancy rates improve but remain below pre-pandemic.",
        "Copper demand strengthens on infrastructure buildout.",
        "Software-as-a-service subscriptions continue upward trend.",
        "Plastics manufacturers face regulatory headwinds.",
        "Ride-sharing platforms report improving profitability metrics.",
        "Timber prices moderate after period of rapid appreciation.",
        "Casino gaming revenues recover in major markets.",
        "Frozen food sales normalize after pandemic stockpiling.",
        "3D printing technology adoption expands in manufacturing.",
    ]
    labels = [
        "positive", "negative", "neutral", "positive", "negative",
        "neutral", "positive", "negative", "positive", "negative",
        "neutral", "positive", "negative", "neutral", "positive",
        "negative", "positive", "negative", "negative", "neutral",
        "positive", "positive", "positive", "negative", "negative",
        "positive", "neutral", "neutral", "positive", "positive",
        "positive", "positive", "neutral", "neutral", "positive",
        "positive", "negative", "positive", "negative", "neutral",
        "positive", "negative", "positive", "positive", "negative",
        "neutral", "neutral", "negative", "positive", "positive",
        "positive", "neutral", "negative", "negative", "positive",
        "positive", "neutral", "negative", "positive", "positive",
        "positive", "neutral", "positive", "negative", "positive",
        "positive", "positive", "positive", "neutral", "positive",
        "neutral", "positive", "positive", "neutral", "positive",
        "neutral", "neutral", "negative", "neutral", "positive",
        "neutral", "positive", "negative", "positive", "neutral",
        "positive", "neutral", "positive", "negative", "positive",
        "neutral", "neutral", "positive", "positive", "negative",
        "positive", "neutral", "positive", "neutral", "positive",
    ]
    # Sanity check lengths
    assert len(texts) == len(labels), "texts and labels must have same length"
    return pd.DataFrame({"text": texts, "label": labels})


if __name__ == "__main__":
    analyzer = FinancialSentimentAnalyzer()
    acc = analyzer.train_model()
    print(f"Trained with accuracy: {acc}")
