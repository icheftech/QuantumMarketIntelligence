# sentiment_engine.py
"""
================================================================================
SENTIMENT ANALYSIS ENGINE
================================================================================
Hybrid VADER + ML sentiment for financial text
"""

import re
import string
from datetime import datetime

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

# Download NLTK data
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
            self.stopwords = set()
        self.finance_keeps = {"up", "down", "above", "below", "over", "under", "more", "less"}
        self.stopwords = self.stopwords - self.finance_keeps

    def clean_text(self, text: str) -> str:
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

    def tokenize(self, text: str):
        try:
            return nltk.word_tokenize(text)
        except LookupError:
            return text.split()

    def remove_stopwords(self, tokens):
        if not self.stopwords:
            return tokens
        return [t for t in tokens if t not in self.stopwords]

    def preprocess(self, text: str) -> str:
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
        try:
            self.ml_model = joblib.load(config.SENTIMENT_MODEL_PATH)
            self.vectorizer = joblib.load(config.VECTORIZER_PATH)
            self.is_trained = True
            print("[Sentiment] Loaded pre-trained model.")
        except Exception:
            print("[Sentiment] No pre-trained model found. VADER-only mode.")
            self.is_trained = False

    def train_model(self, training_data_path: str | None = None):
        if training_data_path is None:
            training_data_path = config.TRAINING_DATA_PATH

        try:
            df = pd.read_csv(training_data_path)
        except FileNotFoundError:
            print("[Sentiment] No training_data.csv found, creating sample training data.")
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

    def analyze_vader(self, text: str) -> float:
        scores = self.vader.polarity_scores(text)
        return scores["compound"]

    def analyze_ml(self, text: str):
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

    def analyze(self, text: str) -> dict:
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

    def batch_analyze(self, texts: list[str]) -> list[dict]:
        return [self.analyze(t) for t in texts]


def get_sentiment_color(score: float) -> str:
    if score < -0.2:
        return config.COLOR_NEGATIVE
    if score > 0.2:
        return config.COLOR_POSITIVE
    return config.COLOR_NEUTRAL


def create_sample_training_data() -> pd.DataFrame:
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
    ]
    labels = [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "positive",
        "negative",
    ]
    return pd.DataFrame({"text": texts, "label": labels})


if __name__ == "__main__":
    analyzer = FinancialSentimentAnalyzer()
    acc = analyzer.train_model()
    print(f"Trained with accuracy: {acc}")
