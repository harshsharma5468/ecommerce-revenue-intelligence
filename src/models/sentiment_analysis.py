"""
Review Sentiment Analysis Module
NLP-based sentiment scoring for customer reviews.

Features:
- Rule-based sentiment scoring (no external dependencies)
- Optional VADER sentiment analysis (if nltk installed)
- Sentiment trends over time
- Keyword extraction for positive/negative reviews

Usage:
    from src.models.sentiment_analysis import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_reviews(reviews_df)
    
    # Get sentiment trend
    trend = analyzer.get_sentiment_trend()
    
    # Get top keywords
    keywords = analyzer.get_top_keywords("positive")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)


# Sentiment lexicons (Portuguese/English mixed for Olist dataset)
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "perfect", "love", "loved", "best", "happy", "satisfied", "recommend",
    "quality", "fast", "quick", "easy", "beautiful", "nice", "super",
    "bom", "ótimo", "excelente", "maravilhoso", "perfeito", "amo",
    "melhor", "feliz", "satisfeito", "recomendo", "qualidade", "rápido",
    "lindo", "incrível", "fantástico", "top", "show", "demais",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "hate", "disappointed",
    "poor", "slow", "late", "damaged", "broken", "wrong", "never", "not",
    "ruim", "terrível", "horrível", "pior", "ódio", "odeio", "decepcionado",
    "pobre", "lento", "atrasado", "danificado", "quebrado", "errado",
    "nunca", "não", "problema", "defeito", "atraso", "demora",
}

INTENSIFIERS = {
    "very": 1.5, "really": 1.5, "extremely": 2.0, "absolutely": 2.0,
    "muito": 1.5, "extremamente": 2.0, "totalmente": 2.0, "demais": 1.5,
}

NEGATORS = {"not", "no", "never", "don't", "doesn't", "won't", "não", "nunca", "jamais"}


class SentimentAnalyzer:
    """
    Sentiment analyzer for customer reviews.
    """
    
    def __init__(self, language: str = "mixed"):
        """
        Initialize sentiment analyzer.
        
        Args:
            language: Language mode ('portuguese', 'english', 'mixed')
        """
        self.language = language
        self.results_df: Optional[pd.DataFrame] = None
        self._try_import_vader()
    
    def _try_import_vader(self):
        """Try to import VADER for enhanced sentiment analysis."""
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            import nltk
            
            # Download VADER lexicon if not present
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            
            self.vader = SentimentIntensityAnalyzer()
            self.vader_available = True
            logger.info("VADER sentiment analyzer loaded")
        except Exception:
            self.vader = None
            self.vader_available = False
            logger.info("Using rule-based sentiment (VADER not available)")
    
    def analyze_reviews(self, df: pd.DataFrame,
                        text_cols: List[str] = None,
                        score_col: str = "review_score") -> pd.DataFrame:
        """
        Analyze sentiment for review DataFrame.
        
        Args:
            df: DataFrame with reviews
            text_cols: Columns containing review text
            score_col: Column with numeric review score (1-5)
            
        Returns:
            DataFrame with sentiment scores added
        """
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        self.results_df = df.copy()
        
        # Calculate sentiment from score (1-5 to -1 to 1)
        if score_col in self.results_df.columns:
            self.results_df["score_sentiment"] = (
                self.results_df[score_col] - 3
            ) / 2
        else:
            self.results_df["score_sentiment"] = 0
        
        # Analyze text sentiment
        if text_cols is None:
            text_cols = [c for c in self.results_df.columns
                        if "comment" in c.lower() or "message" in c.lower() or "title" in c.lower()]
        
        if text_cols:
            combined_text = self.results_df[text_cols].fillna("").agg(" ".join, axis=1)
            self.results_df["text_sentiment"] = combined_text.apply(self._analyze_text)
            
            # Combine score and text sentiment
            self.results_df["combined_sentiment"] = (
                0.7 * self.results_df["score_sentiment"] +
                0.3 * self.results_df["text_sentiment"]
            )
        else:
            self.results_df["text_sentiment"] = 0
            self.results_df["combined_sentiment"] = self.results_df["score_sentiment"]
        
        # Categorize sentiment
        self.results_df["sentiment_category"] = self.results_df["combined_sentiment"].apply(
            self._categorize_sentiment
        )
        
        logger.info(f"Analyzed sentiment for {len(self.results_df)} reviews")
        return self.results_df
    
    def _analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of text string.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between -1 and 1
        """
        if pd.isna(text) or not text.strip():
            return 0
        
        # Use VADER if available
        if self.vader_available:
            scores = self.vader.polarity_scores(str(text))
            return scores["compound"]
        
        # Rule-based fallback
        return self._rule_based_sentiment(str(text).lower())
    
    def _rule_based_sentiment(self, text: str) -> float:
        """
        Rule-based sentiment analysis.
        
        Args:
            text: Lowercase text to analyze
            
        Returns:
            Sentiment score between -1 and 1
        """
        # Tokenize
        words = re.findall(r'\b\w+\b', text)
        
        if not words:
            return 0
        
        score = 0
        intensifier = 1.0
        negated = False
        
        for i, word in enumerate(words):
            # Check for intensifiers
            if word in INTENSIFIERS:
                intensifier = INTENSIFIERS[word]
                continue
            
            # Check for negators
            if word in NEGATORS:
                negated = True
                continue
            
            # Calculate sentiment
            if word in POSITIVE_WORDS:
                word_score = 1.0 * intensifier
                if negated:
                    word_score = -word_score * 0.5
                score += word_score
            elif word in NEGATIVE_WORDS:
                word_score = -1.0 * intensifier
                if negated:
                    word_score = -word_score * 0.5
                score += word_score
            
            # Reset modifiers
            intensifier = 1.0
            negated = False
        
        # Normalize by text length
        if len(words) > 0:
            score = score / max(np.sqrt(len(words)), 1)
        
        # Clamp to [-1, 1]
        return max(-1, min(1, score))
    
    def _categorize_sentiment(self, score: float) -> str:
        """Categorize sentiment score."""
        if score >= 0.5:
            return "Very Positive"
        elif score >= 0.2:
            return "Positive"
        elif score >= -0.2:
            return "Neutral"
        elif score >= -0.5:
            return "Negative"
        else:
            return "Very Negative"
    
    def get_sentiment_trend(self, freq: str = "W") -> pd.DataFrame:
        """
        Get sentiment trend over time.
        
        Args:
            freq: Frequency for resampling ('D', 'W', 'M')
            
        Returns:
            DataFrame with sentiment trend
        """
        if self.results_df is None:
            return pd.DataFrame()
        
        df = self.results_df.copy()
        
        # Find date column
        date_col = None
        for col in ["review_creation_date", "date", "created_at", "timestamp"]:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            logger.warning("No date column found for trend analysis")
            return pd.DataFrame()
        
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        
        if len(df) == 0:
            return pd.DataFrame()
        
        df.set_index(date_col, inplace=True)
        
        # Resample
        trend = df.resample(freq).agg({
            "combined_sentiment": "mean",
            "score_sentiment": "mean",
            "text_sentiment": "mean",
        }).dropna()
        
        trend.columns = ["avg_sentiment", "avg_score_sentiment", "avg_text_sentiment"]
        trend["review_count"] = df.resample(freq).size()
        
        return trend.reset_index()
    
    def get_top_keywords(self, sentiment_type: str = "positive",
                         top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Get top keywords from positive or negative reviews.
        
        Args:
            sentiment_type: 'positive' or 'negative'
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, count) tuples
        """
        if self.results_df is None:
            return []
        
        # Filter by sentiment
        if sentiment_type == "positive":
            filtered = self.results_df[self.results_df["combined_sentiment"] > 0.3]
        else:
            filtered = self.results_df[self.results_df["combined_sentiment"] < -0.3]
        
        if len(filtered) == 0:
            return []
        
        # Combine all text
        text_cols = [c for c in filtered.columns
                    if "comment" in c.lower() or "message" in c.lower() or "title" in c.lower()]
        
        if not text_cols:
            return []
        
        all_text = " ".join(filtered[text_cols].fillna("").agg(" ".join, axis=1))
        
        # Tokenize and count
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        
        # Remove stopwords
        stopwords = {
            "this", "that", "with", "have", "from", "they", "will", "would",
            "there", "their", "what", "about", "more", "when", "were", "very",
            "como", "para", "com", "não", "tem", "muito", "foi", "vou", "faz",
        }
        words = [w for w in words if w not in stopwords]
        
        # Also remove sentiment words (already counted)
        sentiment_words = POSITIVE_WORDS | NEGATIVE_WORDS
        words = [w for w in words if w not in sentiment_words]
        
        return Counter(words).most_common(top_n)
    
    def get_sentiment_summary(self) -> Dict:
        """
        Get summary statistics for sentiment analysis.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.results_df is None:
            return {}
        
        df = self.results_df
        
        return {
            "total_reviews": len(df),
            "avg_sentiment": round(df["combined_sentiment"].mean(), 4),
            "std_sentiment": round(df["combined_sentiment"].std(), 4),
            "positive_pct": round((df["combined_sentiment"] > 0.2).mean() * 100, 1),
            "negative_pct": round((df["combined_sentiment"] < -0.2).mean() * 100, 1),
            "neutral_pct": round(
                ((df["combined_sentiment"] >= -0.2) & (df["combined_sentiment"] <= 0.2)).mean() * 100, 1
            ),
            "category_distribution": df["sentiment_category"].value_counts().to_dict(),
        }
    
    def plot_sentiment_distribution(self) -> "go.Figure":
        """
        Create sentiment distribution plot.
        
        Returns:
            Plotly figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly not available")
            return None
        
        if self.results_df is None:
            return go.Figure().add_annotation(text="No data analyzed")
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=self.results_df["combined_sentiment"],
            nbinsx=50,
            marker_color="#6C63FF",
            opacity=0.7,
        ))
        
        # Vertical lines for categories
        fig.add_vline(x=-0.5, line_dash="dash", line_color="#E74C3C", opacity=0.5)
        fig.add_vline(x=-0.2, line_dash="dash", line_color="#F39C12", opacity=0.5)
        fig.add_vline(x=0.2, line_dash="dash", line_color="#2ECC71", opacity=0.5)
        fig.add_vline(x=0.5, line_dash="dash", line_color="#27AE60", opacity=0.5)
        
        fig.update_layout(
            title="Sentiment Score Distribution",
            xaxis_title="Sentiment Score (-1 to 1)",
            yaxis_title="Count",
            showlegend=False,
        )
        
        return fig
    
    def plot_sentiment_trend(self, freq: str = "W") -> "go.Figure":
        """
        Create sentiment trend over time plot.
        
        Args:
            freq: Frequency for resampling
            
        Returns:
            Plotly figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly not available")
            return None
        
        trend = self.get_sentiment_trend(freq)
        
        if len(trend) == 0:
            return go.Figure().add_annotation(text="No trend data")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trend[trend.columns[0]],  # Date column
            y=trend["avg_sentiment"],
            name="Avg Sentiment",
            mode="lines+markers",
            line=dict(color="#6C63FF", width=2),
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=0.5, line_dash="dash", line_color="#2ECC71", opacity=0.3)
        fig.add_hline(y=-0.5, line_dash="dash", line_color="#E74C3C", opacity=0.3)
        
        fig.update_layout(
            title="Sentiment Trend Over Time",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            showlegend=True,
        )
        
        return fig
    
    def save_results(self, path: str = "data/processed/sentiment_analysis.csv"):
        """Save sentiment analysis results."""
        if self.results_df is None:
            logger.warning("No results to save")
            return
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.results_df.to_csv(save_path, index=False)
        logger.info(f"Sentiment results saved to {path}")
    
    @classmethod
    def load_results(cls, path: str = "data/processed/sentiment_analysis.csv") -> "SentimentAnalyzer":
        """Load sentiment analysis results."""
        analyzer = cls()
        
        load_path = Path(path)
        if load_path.exists():
            analyzer.results_df = pd.read_csv(load_path)
            logger.info(f"Sentiment results loaded from {path}")
        
        return analyzer


# Global analyzer instance
analyzer = SentimentAnalyzer()


def get_analyzer() -> SentimentAnalyzer:
    """Get global analyzer instance."""
    return analyzer
