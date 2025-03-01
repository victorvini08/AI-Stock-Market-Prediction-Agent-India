import yfinance as yf
import json
from langchain_core.messages import HumanMessage
from utils import get_structured_news
from utils import summarize
from state import State

def news_sentiment_analysis_agent(state: State):
    """
    Fetches latest news data for a stock and evaluates it to generate trade signals.

    Args:
    - state (): The state shared between the graph.

    Returns:
    - dict: A dictionary containing summary of latest news, trade signal (sentiment), and confidence score.
    """
    tickers = state["data"]["tickers"]
    from transformers import pipeline

    # Financial-specific model
    sentiment_analyzer = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )
    news_analysis_signals = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        company_name = stock.info['longName']
        news_items = get_structured_news(company_name)
        full_news_text = " "
        positives = []
        negatives = []
        for news in news_items:
            # Combine title and snippet for analysis
            text = f"{news['title']}: {news['snippet']}"
            # Get sentiment prediction
            sentiment = sentiment_analyzer(text)[0]
            full_news_text += text
            if sentiment['label'] == 'positive':
                positives.append(sentiment)
            elif sentiment['label'] == 'negative':
                negatives.append(sentiment)

        # Determine dominant sentiment
        if len(positives) > len(negatives):
            confidence = round(sum(a['score'] for a in positives) / len(positives),2)
            overall_signal =  'Bullish'
        elif len(negatives) > len(positives):
            confidence = round(sum(a['score'] for a in negatives) / len(negatives),2)
            overall_signal = 'Bearish'

        summary = summarize(full_news_text)
        news_analysis_signals[ticker] = {
            "Confidence": confidence,
            "Overall Signal": overall_signal,
            "News Summary": summary
        }

    final_result_na = HumanMessage(
    content=json.dumps(news_analysis_signals),
    name="news_sentiment_analyst_agent",
    )

    state["data"]["analyst_signals"]["news_sentiment_analyst_agent"] = news_analysis_signals

    return {
        "agent_actions": state["agent_actions"] + [final_result_na],
        "data": state["data"]
    }

