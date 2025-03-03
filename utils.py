# Set environment variable to disable tokenizers parallelism warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich import box
from langchain_community.tools import DuckDuckGoSearchResults
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartForConditionalGeneration

# Color styles for different signals
BUY_STYLE = Style(color="green", bold=True)
SELL_STYLE = Style(color="red", bold=True)
NEUTRAL_STYLE = Style(color="yellow", bold=True)
HEADER_STYLE = Style(color="cyan", bold=True)
PRICE_STYLE = Style(color="blue")
METRICS_STYLE = Style(color="magenta")
DATE_STYLE = Style(color="bright_black")

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
    """Summarize news text using a deterministic algorithm"""
    # Use BART model for summarization
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Set max length
    max_length = 1024
    if len(news) > max_length:
        news = news[:max_length]
    
    # Tokenize and generate summary
    inputs = tokenizer([news], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=100)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def clean_text_format(text):
    """
    Clean text formatting issues like individual character bullets
    """
    if not text:
        return text
        
    # Check for character-by-character bulleting which indicates a formatting issue
    if re.search(r'•\s\w\s•\s\w\s•\s\w', text):
        # Remove all bullets and rebuild the text
        cleaned = re.sub(r'•\s', '', text)
        return cleaned
    
    return text

def get_signal_style(signal):
    """Get the appropriate rich style based on the signal"""
    if signal in ["BUY", "STRONG BUY", "BULLISH", "SLIGHTLY BULLISH"]:
        return BUY_STYLE
    elif signal in ["SELL", "STRONG SELL", "BEARISH", "SLIGHTLY BEARISH"]:
        return SELL_STYLE
    else:
        return NEUTRAL_STYLE

def format_trade_prediction(prediction_data):
    """
    Format the prediction data for display with rich color formatting.
    
    Args:
        prediction_data: Trade prediction data
        
    Returns:
        str: Formatted trade prediction
    """
    console = Console()
    output = []
    
    # Process each ticker's prediction
    for ticker, pred_data in prediction_data.items():
        # Create ticker header
        ticker_text = Text()
        ticker_text.append("\n")
        ticker_text.append(f"===== {ticker} =====", style=HEADER_STYLE)
        
        # Add signal with appropriate coloring
        signal = pred_data.get("Overall_Signal", "NEUTRAL")
        signal_style = get_signal_style(signal)
        
        signal_text = Text()
        signal_text.append("SIGNAL: ", style=HEADER_STYLE)
        signal_text.append(signal, style=signal_style)
        
        # Add confidence
        confidence_text = Text()
        confidence_text.append("CONFIDENCE: ", style=HEADER_STYLE)
        confidence_text.append(f"{pred_data.get('Confidence', 0.5) * 100:.1f}%", style=METRICS_STYLE)
        
        # Add to output
        output.append(str(ticker_text))
        output.append(str(signal_text))
        output.append(str(confidence_text))
        output.append("")
        
        # Add price targets section if available
        if 'Price_Targets' in pred_data:
            pt = pred_data['Price_Targets']
            
            price_header = Text("PRICE TARGETS:", style=HEADER_STYLE)
            
            current_price = Text()
            current_price.append("Current: ", style=PRICE_STYLE)
            current_price.append(f"₹{pt.get('Current_Price', 0):.2f}", style=PRICE_STYLE)
            
            target_1d = Text()
            target_1d.append("1-Day:   ", style=PRICE_STYLE)
            target_1d.append(f"₹{pt.get('Target_1D', 0):.2f} ", style=PRICE_STYLE)
            target_1d.append(f"(Range: ₹{pt['CI_1D'][0]:.2f} - ₹{pt['CI_1D'][1]:.2f})", style=DATE_STYLE)
            
            target_1w = Text()
            target_1w.append("1-Week:  ", style=PRICE_STYLE)
            target_1w.append(f"₹{pt.get('Target_1W', 0):.2f} ", style=PRICE_STYLE)
            target_1w.append(f"(Range: ₹{pt['CI_1W'][0]:.2f} - ₹{pt['CI_1W'][1]:.2f})", style=DATE_STYLE)
            
            target_1m = Text()
            target_1m.append("1-Month: ", style=PRICE_STYLE)
            target_1m.append(f"₹{pt.get('Target_1M', 0):.2f} ", style=PRICE_STYLE)
            target_1m.append(f"(Range: ₹{pt['CI_1M'][0]:.2f} - ₹{pt['CI_1M'][1]:.2f})", style=DATE_STYLE)
            
            expected_move = Text()
            expected_move.append("Expected Move: ", style=METRICS_STYLE)
            move_pct = pt.get('Expected_Move_Percent', 0)
            move_style = BUY_STYLE if move_pct > 0 else SELL_STYLE if move_pct < 0 else NEUTRAL_STYLE
            expected_move.append(f"{move_pct:.2f}%", style=move_style)
            
            # Add to output
            output.append(str(price_header))
            output.append(str(current_price))
            output.append(str(target_1d))
            output.append(str(target_1w))
            output.append(str(target_1m))
            output.append(str(expected_move))
            output.append("")
        
        # Add rationale and risk factors
        if 'Rationale' in pred_data:
            rationale = clean_text_format(pred_data['Rationale'])
            rationale_header = Text("RATIONALE:", style=HEADER_STYLE)
            
            # Add to output
            output.append(str(rationale_header))
            output.append(rationale)
            output.append("")
            
        if 'Risk_Factors' in pred_data and pred_data['Risk_Factors']:
            risks_header = Text("RISK FACTORS:", style=HEADER_STYLE)
            output.append(str(risks_header))
            
            # Handle different risk factor formats
            risk_factors = pred_data['Risk_Factors']
            
            if isinstance(risk_factors, list):
                for risk in risk_factors:
                    # Clean up any formatting issues
                    clean_risk = clean_text_format(risk)
                    if clean_risk and len(clean_risk.strip()) > 0:
                        risk_text = Text("• ", style=SELL_STYLE)
                        risk_text.append(clean_risk)
                        output.append(str(risk_text))
            elif isinstance(risk_factors, str):
                # If it's a single string, split by newlines or periods
                if '\n' in risk_factors:
                    for risk in risk_factors.split('\n'):
                        if risk.strip():
                            risk_text = Text("• ", style=SELL_STYLE)
                            risk_text.append(risk.strip())
                            output.append(str(risk_text))
                else:
                    # Clean up the text first
                    clean_risks = clean_text_format(risk_factors)
                    # Split by periods for sentence-by-sentence risks
                    sentences = [s.strip() for s in re.split(r'\.(?=\s|$)', clean_risks) if s.strip()]
                    for risk in sentences:
                        if risk:
                            risk_text = Text("• ", style=SELL_STYLE)
                            risk_text.append(risk)
                            output.append(str(risk_text))
            
            output.append("")
            
        # Add allocation recommendation if available
        if 'Suggested_Allocation' in pred_data:
            allocation_header = Text("SUGGESTED ALLOCATION: ", style=HEADER_STYLE)
            
            # Handle different formats of allocation
            allocation = pred_data['Suggested_Allocation']
            if isinstance(allocation, (int, float)):
                allocation_header.append(f"{allocation:.1f}%", style=METRICS_STYLE)
            else:
                # Try to extract a number from text
                allocation_text = clean_text_format(str(allocation))
                number_match = re.search(r'(\d+(?:\.\d+)?)', allocation_text)
                if number_match:
                    allocation_header.append(f"{float(number_match.group(1)):.1f}%", style=METRICS_STYLE)
                else:
                    allocation_header.append(allocation_text)
            
            output.append(str(allocation_header))
            output.append("")
            
        # Add which signals contributed most to the prediction
        if all(k in pred_data for k in ['Technical_Weight', 'Fundamental_Weight', 'News_Weight']):
            weights_header = Text("SIGNAL WEIGHTS:", style=HEADER_STYLE)
            
            tech_weight = Text()
            tech_weight.append("Technical:    ", style=METRICS_STYLE)
            tech_weight.append(f"{pred_data['Technical_Weight'] * 100:.1f}%")
            
            fund_weight = Text()
            fund_weight.append("Fundamental:  ", style=METRICS_STYLE)
            fund_weight.append(f"{pred_data['Fundamental_Weight'] * 100:.1f}%")
            
            news_weight = Text()
            news_weight.append("News:         ", style=METRICS_STYLE)
            news_weight.append(f"{pred_data['News_Weight'] * 100:.1f}%")
            
            agreement = Text()
            agreement.append("Agreement:    ", style=METRICS_STYLE)
            agreement.append(f"{pred_data.get('Agent_Agreement', 0) * 100:.1f}%")
            
            output.append(str(weights_header))
            output.append(str(tech_weight))
            output.append(str(fund_weight))
            output.append(str(news_weight))
            output.append(str(agreement))
            output.append("")
        
        # Add any additional insights
        if 'Additional_Insights' in pred_data and pred_data['Additional_Insights']:
            insights = clean_text_format(pred_data['Additional_Insights'])
            insights_header = Text("ADDITIONAL INSIGHTS:", style=HEADER_STYLE)
            
            output.append(str(insights_header))
            output.append(insights)
            output.append("")
            
        # Add timestamp
        if 'Prediction_Time' in pred_data:
            time_text = Text()
            time_text.append("Generated: ", style=DATE_STYLE)
            time_text.append(pred_data['Prediction_Time'], style=DATE_STYLE)
            output.append(str(time_text))
            
        # Add divider
        divider = Text("=" * 40, style=HEADER_STYLE)
        output.append(str(divider))
        output.append("")
    
    # If we're in a non-terminal environment, fall back to plain text
    try:
        console.print("\n".join(output))
        return ""  # Return empty since we've already printed
    except:
        # If rich printing fails, return the plain text version
        return "\n".join(output)
    
