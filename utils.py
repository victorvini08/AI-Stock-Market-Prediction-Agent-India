import re
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from langchain_community.tools import DuckDuckGoSearchResults
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def parse_search_results(response: str) -> list[dict]:
    """
    Parses DuckDuckGoSearchResults string output into structured data
    Returns list of dictionaries with snippet, title, and link
    """
    structured_results = []

    # Split entries by 'snippet: ' and process each entry
    entries = re.split(r',\s*snippet:\s*', response)

    for entry in entries:
        try:
            # Use regex to extract components
            match = re.search(
                r'(?P<snippet>.*?),\s*title:\s*(?P<title>.*?),\s*link:\s*(?P<link>https?://[^\s,]+)',
                entry,
                re.DOTALL
            )

            if match:
                structured_results.append({
                    'snippet': match.group('snippet').strip(),
                    'title': match.group('title').strip(),
                    'link': match.group('link').strip()
                })
        except Exception as e:
            print(f"Error parsing entry: {e}")
            continue

    return structured_results

def get_structured_news(stock_name: str, max_results: int = 5) -> list[dict]:
    """
    Gets structured news for Indian stocks using DuckDuckGoSearchResults
    Returns list of dictionaries ready for sentiment analysis
    """
    search = DuckDuckGoSearchResults(backend="news", max_results=max_results)
    query = f"{stock_name} stock latest news India"
    raw_results = search.run(query)

    return parse_search_results(raw_results)[:max_results]

def summarize(news):
    model_name = "t5-small"  # Lightweight summarization model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer.encode("summarize: " + news, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(input_ids, max_length=50, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and return the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def llm_prediction(tickers: list, analyst_signals: dict, portfolio: dict, llm):
    prompt_template = """You are a sophisticated trading AI agent that takes decision of trade (Buy/Sell/Hold) and allocates capital based on multiple analysis signals.
        
        You get a list of tickers and max cash amount for each ticker as input. You also get the signals from different analyst agents.
        
        Tickers: {tickers}
        Cash (List which contains max cash allotment for each ticker): {portfolio}

        Analyze the following signals for each ticker and allocate the available cash accordingly:
        
        Technical Analysis signals: {technical_analysis_signals}
        The dictionary contains:
        - Overall Signal (BULLISH/BEARISH/NEUTRAL)
        - Confidence (numerical value between 0 and 1)
        - Technical Indicator Signals (Signals based on indicators like RSI, MACD, etc)

        Fundamental Analysis signals: {fundamental_analysis_signals}
        The dictionary contains:
        - Overall Signal (BULLISH/BEARISH/NEUTRAL)
        - Confidence (numerical value between 0 and 1)
        - Fundamental Analysis Signals (Signals based on fundamentals like Revenue growth, PE ratio, etc)

        News Analysis signals: {news_analysis_signals}
        The dictionary contains:
        - Overall Signal (BULLISH/BEARISH/NEUTRAL)
        - Confidence (numerical value between 0 and 1)
        - News Summary for the given stock

        Follow these exact steps for each ticker:
        
        1. Convert signals to numerical values:
           - BULLISH = +1
           - BEARISH = -1
           - NEUTRAL = +0.1
        
        2. Calculate the aggregate confidence score with proper weights:
           - Technical weight = 0.35
           - Fundamental weight = 0.30
           - News weight = 0.35
           
           Aggregate confidence score = (technical_weight * technical_signal_value * technical_confidence) + 
                                        (fundamental_weight * fundamental_signal_value * fundamental_confidence) + 
                                        (news_weight * news_signal_value * news_confidence)
        
        3. Determine the trading decision:
           - If aggregate confidence score > 0.2: "BUY"
           - If aggregate confidence score < -0.2: "SELL"
           - Otherwise: "HOLD"
        
        4. Calculate allocation amount:
           - For BUY: allocation = max_cash_for_stock * abs(aggregate_confidence_score)
           - For SELL or HOLD: allocation = 0
        
        5. Calculate stop loss percentage (only for BUY decisions):
           - stop_loss = 1 - (0.05 + (0.10 * (1 - technical_confidence)))
           - This represents the percentage of the original price to set as stop loss
        
        For each ticker, return a JSON object in the following format:
        {{
            "ticker": "SYMBOL",
            "decision": "BUY/HOLD/SELL",
            "allocated_amount": float,
            "confidence_score": float,
            "stop_loss": float,
            "rationale": "Detailed explanation with specific references to technical, fundamental, and news signals"
        }}

        Return a list of these JSON objects for all tickers.Do not include any string except the list of jsons. So ensure output format is a list. Make sure to show all mathematical calculations in the rationale field.
        Do not hallucinate or make up data. Only use the provided analysis signals.
        """
    
    prompt = prompt_template.format(
        tickers=tickers,
        portfolio=portfolio,
        technical_analysis_signals=analyst_signals["technical_analyst_agent"],
        fundamental_analysis_signals=analyst_signals["fundamental_analyst_agent"],
        news_analysis_signals=analyst_signals["news_sentiment_analyst_agent"]
    )

    response = llm.invoke(prompt)

    return response
    
