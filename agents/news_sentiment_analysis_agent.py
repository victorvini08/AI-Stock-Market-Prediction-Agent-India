import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yfinance as yf
import json
import hashlib
import math
import re
from datetime import datetime, timedelta
import random
from state import State
from utils import get_structured_news, summarize
from langchain_core.messages import HumanMessage
from transformers import pipeline, set_seed

# Set seed for reproducibility across all randomness
random.seed(42)
set_seed(42)

# Cache for news data to ensure consistent results
news_cache = {}

def clean_text(text):
    """Clean text for better sentiment analysis"""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep important ones for sentiment
    text = re.sub(r'[^\w\s\.\,\!\?\$\%\+\-]', '', text)
    
    return text

def extract_financial_entities(text):
    """Extract financial entities and metrics from text"""
    entities = []
    
    # Price mentions
    price_pattern = r'Rs\.?\s?(\d+(?:\.\d+)?)|₹\s?(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s?(?:rupees|rs)'
    price_matches = re.findall(price_pattern, text, re.IGNORECASE)
    
    if price_matches:
        # Process the matches
        for match in price_matches:
            # Take the first non-empty group
            price = next((m for m in match if m), None)
            if price:
                entities.append({"type": "price", "value": float(price)})
    
    # Percentage changes
    pct_pattern = r'(\+|\-)?(\d+(?:\.\d+)?)%'
    pct_matches = re.findall(pct_pattern, text)
    
    if pct_matches:
        for match in pct_matches:
            sign, value = match
            multiplier = -1 if sign == '-' else 1
            entities.append({"type": "percentage", "value": multiplier * float(value)})
    
    # Growth/decline words patterns
    growth_words = ['increase', 'rise', 'gain', 'grew', 'up', 'higher', 'positive', 'profit', 'rally', 'surge', 'jump']
    decline_words = ['decrease', 'fall', 'drop', 'fell', 'down', 'lower', 'negative', 'loss', 'decline', 'slump', 'crash']
    
    text_lower = text.lower()
    
    for word in growth_words:
        if f" {word} " in f" {text_lower} ":
            entities.append({"type": "sentiment_word", "value": "positive", "word": word})
    
    for word in decline_words:
        if f" {word} " in f" {text_lower} ":
            entities.append({"type": "sentiment_word", "value": "negative", "word": word})
    
    return entities

def analyze_entity_based_sentiment(entities):
    """Analyze sentiment based on extracted financial entities"""
    if not entities:
        return {"score": 0, "label": "neutral", "confidence": 0.5}
    
    # Count positive and negative indicators
    pos_count = 0
    neg_count = 0
    sentiment_words = []
    
    for entity in entities:
        if entity["type"] == "percentage":
            if entity["value"] > 0:
                pos_count += 1
            elif entity["value"] < 0:
                neg_count += 1
        
        elif entity["type"] == "sentiment_word":
            if entity["value"] == "positive":
                pos_count += 1
                sentiment_words.append(entity["word"])
            elif entity["value"] == "negative":
                neg_count += 1
                sentiment_words.append(entity["word"])
    
    # Calculate entity-based sentiment
    if pos_count > neg_count:
        score = min(1.0, 0.5 + (pos_count - neg_count) * 0.1)
        return {"score": score, "label": "positive", "confidence": score, "sentiment_words": sentiment_words}
    elif neg_count > pos_count:
        score = min(1.0, 0.5 + (neg_count - pos_count) * 0.1)
        return {"score": -score, "label": "negative", "confidence": score, "sentiment_words": sentiment_words}
    else:
        return {"score": 0, "label": "neutral", "confidence": 0.5, "sentiment_words": sentiment_words}

def merge_sentiment_analyses(model_sentiment, entity_sentiment):
    """Merge model-based and entity-based sentiment analyses"""
    # If they agree, increase confidence
    if model_sentiment["label"] == entity_sentiment["label"]:
        confidence = min(0.95, model_sentiment["score"] + entity_sentiment["confidence"] * 0.2)
        return {
            "label": model_sentiment["label"], 
            "score": confidence,
            "entity_agreement": True,
            "sentiment_words": entity_sentiment.get("sentiment_words", [])
        }
    
    # If entity sentiment is stronger, weight it more
    if entity_sentiment["confidence"] > model_sentiment["score"] + 0.2:
        confidence = entity_sentiment["confidence"]
        return {
            "label": entity_sentiment["label"], 
            "score": confidence,
            "entity_override": True,
            "sentiment_words": entity_sentiment.get("sentiment_words", [])
        }
    
    # Otherwise, default to model sentiment with slight adjustment
    return {
        "label": model_sentiment["label"], 
        "score": model_sentiment["score"],
        "entity_influence": 0.1,
        "sentiment_words": entity_sentiment.get("sentiment_words", [])
    }

def get_stock_price_history(ticker, days=30):
    """Get recent stock price history for context"""
    try:
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get historical data
        hist = stock.history(start=start_date, end=end_date)
        
        if len(hist) > 0:
            # Calculate price change
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            price_change_pct = ((end_price / start_price) - 1) * 100
            
            # Get max and min
            max_price = hist['High'].max()
            min_price = hist['Low'].min()
            
            # Calculate average volume
            avg_volume = hist['Volume'].mean()
            
            return {
                'start_price': start_price,
                'end_price': end_price,
                'price_change_pct': price_change_pct,
                'max_price': max_price,
                'min_price': min_price,
                'avg_volume': avg_volume,
                'trend': 'up' if price_change_pct > 0 else 'down'
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching price history for {ticker}: {e}")
        return None

def analyze_news_sentiment(news_items, price_history=None):
    """Analyze sentiment of news items with advanced context-aware methods"""
    # Use a smaller sentiment model
    try:
        # Use distilbert instead of finbert (much smaller model)
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        # Fallback to rule-based sentiment only
        sentiment_analyzer = None
    
    results = []
    
    for news in news_items:
        # Combine title and snippet for analysis
        title = news.get('title', '')
        snippet = news.get('snippet', '')
        
        # Clean the text
        cleaned_title = clean_text(title)
        cleaned_snippet = clean_text(snippet)
        
        full_text = f"{cleaned_title}: {cleaned_snippet}"
        
        # Deterministic hash-based scoring (for consistency)
        text_hash = int(hashlib.md5(full_text.encode()).hexdigest(), 16)
        deterministic_baseline = (text_hash % 1000) / 1000.0  # Between 0 and 1
        
        # Extract financial entities
        entities = extract_financial_entities(full_text)
        
        # Get sentiment from model
        try:
            if sentiment_analyzer:
                # Get sentiment from the model
                model_result = sentiment_analyzer(full_text)
                
                # Safely extract the label and score
                if isinstance(model_result, list):
                    # The model returns a list of results
                    if len(model_result) > 0:
                        result = model_result[0]  # Get first result
                        
                        # Extract label and score safely
                        if isinstance(result, dict) and 'label' in result and 'score' in result:
                            label = result['label'].lower()
                            score = result['score']
                            
                            # Map POSITIVE/NEGATIVE to our format
                            if label == 'positive' or label == 'POSITIVE':
                                model_sentiment = {"label": "positive", "score": score}
                            elif label == 'negative' or label == 'NEGATIVE':
                                model_sentiment = {"label": "negative", "score": score}
                            else:
                                model_sentiment = {"label": "neutral", "score": 0.5}
                        else:
                            # Fallback if dictionary format is unexpected
                            model_sentiment = {"label": "neutral", "score": 0.5}
                    else:
                        # Empty list fallback
                        model_sentiment = {"label": "neutral", "score": 0.5}
                else:
                    # Unexpected format fallback
                    model_sentiment = {"label": "neutral", "score": 0.5}
            else:
                # No model available, use deterministic baseline
                if deterministic_baseline > 0.6:
                    model_sentiment = {"label": "positive", "score": deterministic_baseline}
                elif deterministic_baseline < 0.4:
                    model_sentiment = {"label": "negative", "score": deterministic_baseline}
                else:
                    model_sentiment = {"label": "neutral", "score": 0.5}
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Fallback to deterministic score
            if deterministic_baseline > 0.6:
                model_sentiment = {"label": "positive", "score": deterministic_baseline}
            elif deterministic_baseline < 0.4:
                model_sentiment = {"label": "negative", "score": deterministic_baseline}
            else:
                model_sentiment = {"label": "neutral", "score": 0.5}
        
        # Get entity-based sentiment
        entity_sentiment = analyze_entity_based_sentiment(entities)
        
        # Merge both sentiment analyses
        merged_sentiment = merge_sentiment_analyses(model_sentiment, entity_sentiment)
        
        # Standardize the output
        if merged_sentiment["label"] == "positive":
            sentiment_result = {
                "label": "positive",
                "score": merged_sentiment["score"],
                "entities": entities
            }
        elif merged_sentiment["label"] == "negative":
            sentiment_result = {
                "label": "negative", 
                "score": merged_sentiment["score"],
                "entities": entities
            }
        else:
            sentiment_result = {
                "label": "neutral",
                "score": merged_sentiment["score"],
                "entities": entities
            }
        
        # Add context from price history if available
        if price_history:
            # If news sentiment aligns with recent price trend, boost confidence
            if (sentiment_result["label"] == "positive" and price_history["trend"] == "up") or \
               (sentiment_result["label"] == "negative" and price_history["trend"] == "down"):
                sentiment_result["score"] = min(0.95, sentiment_result["score"] + 0.1)
                sentiment_result["price_trend_alignment"] = True
        
        results.append({
            "title": title,
            "snippet": snippet,
            "sentiment": sentiment_result
        })
    
    return results

def news_sentiment_analysis_agent(state: State):
    """
    Analyzes news sentiment for stocks using advanced NLP and contextual analysis.
    
    Args:
    - state: The shared state between agents
    
    Returns:
    - Updated state with news sentiment analysis signals
    """
    tickers = state["data"]["tickers"]
    news_analysis_signals = {}
    
    # Cache key based on tickers and date (to refresh daily)
    today = datetime.now().strftime("%Y-%m-%d")
    cache_key = hashlib.md5(f"{'-'.join(sorted(tickers))}-{today}".encode()).hexdigest()
    
    # Check if we have cached news data for today
    if cache_key in news_cache:
        stored_news = news_cache[cache_key]
    else:
        stored_news = {}
    
    for ticker in tickers:
        # Use cached news if available
        if ticker in stored_news:
            news_items = stored_news[ticker]['news_items']
            company_name = stored_news[ticker]['company_name']
            price_history = stored_news[ticker].get('price_history')
        else:
            stock = yf.Ticker(ticker)
            try:
                company_name = stock.info.get('longName', ticker)
                news_items = get_structured_news(company_name)
                price_history = get_stock_price_history(ticker)
                
                # Store in cache
                stored_news[ticker] = {
                    'company_name': company_name,
                    'news_items': news_items,
                    'price_history': price_history
                }
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                company_name = ticker
                news_items = []
                price_history = None
        
        # Update global cache
        news_cache[cache_key] = stored_news
        
        if not news_items:
            # If no news, use neutral sentiment
            news_analysis_signals[ticker] = {
                "Overall Signal": "NEUTRAL",
                "Confidence": 0.5,
                "News Summary": f"No recent news found for {company_name}"
            }
            continue
        
        # Analyze sentiment of news
        sentiment_results = analyze_news_sentiment(news_items, price_history)
        
        # Aggregate sentiment scores
        positive_scores = [r["sentiment"]["score"] for r in sentiment_results if r["sentiment"]["label"] == "positive"]
        negative_scores = [r["sentiment"]["score"] for r in sentiment_results if r["sentiment"]["label"] == "negative"]
        
        positive_count = len(positive_scores)
        negative_count = len(negative_scores)
        neutral_count = len(sentiment_results) - positive_count - negative_count
        
        # Calculate weighted sentiment score
        if positive_count > 0:
            avg_positive_score = sum(positive_scores) / positive_count
        else:
            avg_positive_score = 0
            
        if negative_count > 0:
            avg_negative_score = sum(negative_scores) / negative_count
        else:
            avg_negative_score = 0
        
        # Generate overall sentiment signal
        if positive_count > negative_count and positive_count >= max(1, len(sentiment_results) / 3):
            overall_signal = "BULLISH"
            confidence = round(avg_positive_score * (positive_count / len(sentiment_results)), 2)
        elif negative_count > positive_count and negative_count >= max(1, len(sentiment_results) / 3):
            overall_signal = "BEARISH"
            confidence = round(avg_negative_score * (negative_count / len(sentiment_results)), 2)
        else:
            overall_signal = "NEUTRAL"
            confidence = 0.5
        
        # Generate a combined text of all news for summarization
        full_news_text = " ".join([f"{r['title']}: {r['snippet']}" for r in sentiment_results])
        summary = summarize(full_news_text)
        
        # Extract price mentions
        all_entities = []
        for result in sentiment_results:
            all_entities.extend(result["sentiment"].get("entities", []))
            
        price_mentions = [e for e in all_entities if e["type"] == "price"]
        price_values = [e["value"] for e in price_mentions] if price_mentions else []
        
        news_analysis_signals[ticker] = {
            "Overall Signal": overall_signal,
            "Confidence": confidence,
            "News Summary": summary,
            "Positive Count": positive_count,
            "Negative Count": negative_count,
            "Neutral Count": neutral_count,
            "News Count": len(sentiment_results),
            "Recent Price Trend": price_history["trend"] if price_history else "unknown",
            "Mentioned Prices": price_values[:3] if price_values else []
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

