import numpy as np
import json
import random
import re
import scipy.stats as stats
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from datetime import datetime
from langchain_core.messages import HumanMessage
from state import State
from langchain_groq import ChatGroq

# Set fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def normalize_score(score, min_val=0, max_val=1):
    """Normalize score to be between 0 and 1"""
    return max(min_val, min(max_val, score))

def confidence_to_weight(confidence):
    """Convert confidence score to weight for ensemble"""
    # Apply sigmoid-like function to confidence
    return 1 / (1 + np.exp(-10 * (confidence - 0.5)))

def parse_signal_value(signal):
    """Parse signal string to numeric value"""
    signal_map = {
        "STRONG BUY": 1.0,
        "BUY": 0.75,
        "BULLISH": 0.75,
        "SLIGHTLY BULLISH": 0.6,
        "NEUTRAL": 0.5,
        "SLIGHTLY BEARISH": 0.4,
        "BEARISH": 0.25,
        "SELL": 0.25,
        "STRONG SELL": 0.0,
    }
    # Case-insensitive matching
    signal_upper = signal.upper()
    for key, value in signal_map.items():
        if key in signal_upper:
            return value
    return 0.5  # Default neutral

def create_feature_vector(signals_data, ticker):
    """Create a feature vector from all agent signals for machine learning"""
    features = []
    
    # Technical analysis features
    if "technical_analysis_agent" in signals_data:
        ta_data = signals_data["technical_analysis_agent"].get(ticker, {})
        # Get values with fallbacks for any missing fields
        features.extend([
            parse_signal_value(ta_data.get("Overall Signal", "NEUTRAL")),
            normalize_score(ta_data.get("Confidence", 0.5)),
            ta_data.get("Above_200d_SMA", 0),
            normalize_score(ta_data.get("RSI_value", ta_data.get("RSI", 50)) / 100),
            # Get MACD signal - might be a value or classification
            normalize_score(ta_data.get("MACD_Signal", 0) if isinstance(ta_data.get("MACD_Signal", 0), float) 
                           else 0.75 if ta_data.get("MACD_Signal", 0) == 1 
                           else 0.25 if ta_data.get("MACD_Signal", 0) == -1 
                           else 0.5),
            normalize_score(ta_data.get("Bullish_Indicators", 0) / max(1, ta_data.get("Bullish_Indicators", 0) + ta_data.get("Bearish_Indicators", 0)))
        ])
    else:
        features.extend([0.5, 0.5, 0, 0.5, 0.5, 0.5])  # Default values
    
    # Fundamental analysis features
    if "fundamental_analysis_agent" in signals_data:
        fa_data = signals_data["fundamental_analysis_agent"].get(ticker, {})
        features.extend([
            parse_signal_value(fa_data.get("Overall Signal", "NEUTRAL")),
            normalize_score(fa_data.get("Confidence", 0.5)),
            normalize_score(fa_data.get("PE_Ratio_Score", 0.5)),
            normalize_score(fa_data.get("Growth_Score", 0.5)),
            normalize_score(fa_data.get("Value_Score", 0.5)),
            1 if fa_data.get("Undervalued", False) else 0
        ])
    else:
        features.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0])  # Default values
    
    # News sentiment features
    if "news_sentiment_analyst_agent" in signals_data:
        ns_data = signals_data["news_sentiment_analyst_agent"].get(ticker, {})
        features.extend([
            parse_signal_value(ns_data.get("Overall Signal", "NEUTRAL")),
            normalize_score(ns_data.get("Confidence", 0.5)),
            ns_data.get("Positive Count", 0) / max(1, ns_data.get("News Count", 1)),
            ns_data.get("Negative Count", 0) / max(1, ns_data.get("News Count", 1)),
            1 if ns_data.get("Recent Price Trend", "") == "up" else 0
        ])
    else:
        features.extend([0.5, 0.5, 0, 0, 0])  # Default values
    
    return np.array(features)

def ensemble_prediction(signals_data, ticker):
    """Use ensemble methods to generate a more robust prediction"""
    # Get signal weights based on confidence
    weights = {}
    
    # Technical analysis
    if "technical_analysis_agent" in signals_data:
        ta_data = signals_data["technical_analysis_agent"].get(ticker, {})
        ta_signal = parse_signal_value(ta_data.get("Overall Signal", "NEUTRAL"))
        ta_confidence = normalize_score(ta_data.get("Confidence", 0.5))
        weights["technical"] = confidence_to_weight(ta_confidence)
    else:
        ta_signal = 0.5
        weights["technical"] = 0.0
    
    # Fundamental analysis
    if "fundamental_analysis_agent" in signals_data:
        fa_data = signals_data["fundamental_analysis_agent"].get(ticker, {})
        fa_signal = parse_signal_value(fa_data.get("Overall Signal", "NEUTRAL"))
        fa_confidence = normalize_score(fa_data.get("Confidence", 0.5))
        weights["fundamental"] = confidence_to_weight(fa_confidence)
    else:
        fa_signal = 0.5
        weights["fundamental"] = 0.0
    
    # News sentiment
    if "news_sentiment_analyst_agent" in signals_data:
        ns_data = signals_data["news_sentiment_analyst_agent"].get(ticker, {})
        ns_signal = parse_signal_value(ns_data.get("Overall Signal", "NEUTRAL"))
        ns_confidence = normalize_score(ns_data.get("Confidence", 0.5))
        weights["news"] = confidence_to_weight(ns_confidence)
    else:
        ns_signal = 0.5
        weights["news"] = 0.0
    
    # Calculate weighted average signal
    total_weight = sum(weights.values())
    if total_weight > 0:
        weighted_signal = (
            weights["technical"] * ta_signal +
            weights["fundamental"] * fa_signal +
            weights["news"] * ns_signal
        ) / total_weight
    else:
        weighted_signal = 0.5  # Default neutral
    
    # Calculate dispersion/agreement using standard deviation
    signals = [ta_signal, fa_signal, ns_signal]
    signal_std = np.std(signals)
    
    # Adjust confidence based on agreement (lower std = higher confidence)
    agreement_factor = 1 - min(1, signal_std * 2)
    
    # Calculate overall confidence
    overall_confidence = min(0.95, np.mean([
        weights["technical"],
        weights["fundamental"],
        weights["news"]
    ]) * agreement_factor)
    
    # Apply machine learning model to refine prediction
    feature_vector = create_feature_vector(signals_data, ticker)
    
    # Random Forest classifier (simplified)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    # Training data would normally come from historical signals and outcomes
    # Here we use a simple synthetic example
    X_synthetic = np.random.rand(100, len(feature_vector))
    y_synthetic = (X_synthetic.mean(axis=1) > 0.5).astype(int)
    rf.fit(X_synthetic, y_synthetic)
    
    # Get ML confidence
    ml_proba = rf.predict_proba([feature_vector])[0]
    ml_confidence = max(ml_proba)
    ml_signal = 1.0 if ml_proba[1] > 0.5 else 0.0
    
    # Blend weighted signal with ML signal
    final_signal = 0.7 * weighted_signal + 0.3 * ml_signal
    final_confidence = 0.7 * overall_confidence + 0.3 * ml_confidence
    
    # Determine signal label based on final signal value
    if final_signal >= 0.75:
        signal_label = "STRONG BUY"
    elif final_signal >= 0.6:
        signal_label = "BUY"
    elif final_signal >= 0.55:
        signal_label = "SLIGHTLY BULLISH"
    elif final_signal >= 0.45:
        signal_label = "NEUTRAL"
    elif final_signal >= 0.4:
        signal_label = "SLIGHTLY BEARISH"
    elif final_signal >= 0.25:
        signal_label = "SELL"
    else:
        signal_label = "STRONG SELL"
    
    return {
        "Signal": signal_label,
        "Confidence": round(final_confidence, 2),
        "Signal_Value": round(final_signal, 2),
        "Technical_Weight": round(weights["technical"], 2),
        "Fundamental_Weight": round(weights["fundamental"], 2),
        "News_Weight": round(weights["news"], 2),
        "Agent_Agreement": round(agreement_factor, 2),
    }

def estimate_price_range(signals_data, ticker, current_price):
    """Estimate price ranges based on signals and market data"""
    # Default volatility
    volatility = 0.02  # 2% daily volatility
    
    # Try to get actual volatility from technical data
    if "technical_analysis_agent" in signals_data:
        ta_data = signals_data["technical_analysis_agent"].get(ticker, {})
        if "ATR_Percent" in ta_data:
            volatility = ta_data["ATR_Percent"] / 100
        elif "Volatility" in ta_data:
            volatility = ta_data["Volatility"] / 100
        elif "ATR" in ta_data:
            # Normalize ATR to percentage if only raw ATR is available
            volatility = ta_data["ATR"] / current_price
    
    # Get signal strength
    prediction = ensemble_prediction(signals_data, ticker)
    signal_value = prediction["Signal_Value"]
    confidence = prediction["Confidence"]
    
    # Signal strength from 0 to 1, where 0.5 is neutral
    signal_strength = abs(signal_value - 0.5) * 2
    
    # Calculate expected movement based on signal and volatility
    # Scale by confidence
    expected_move_percent = volatility * signal_strength * confidence
    
    # Direction based on signal
    direction = 1 if signal_value > 0.5 else -1
    
    # Calculate price targets
    expected_move = current_price * expected_move_percent * direction
    
    # Base case: neutral signal
    if 0.45 <= signal_value <= 0.55:
        target_1d = current_price * (1 + direction * volatility)
        target_1w = current_price * (1 + direction * volatility * 2)
        target_1m = current_price * (1 + direction * volatility * 3)
    else:
        # Stronger signals lead to larger expected moves
        target_1d = current_price * (1 + expected_move_percent * direction)
        target_1w = current_price * (1 + expected_move_percent * direction * 2.5)
        target_1m = current_price * (1 + expected_move_percent * direction * 5)
    
    # Calculate confidence intervals
    ci_lower_1d = current_price * (1 - volatility)
    ci_upper_1d = current_price * (1 + volatility)
    
    ci_lower_1w = current_price * (1 - volatility * 2)
    ci_upper_1w = current_price * (1 + volatility * 2)
    
    ci_lower_1m = current_price * (1 - volatility * 3.5)
    ci_upper_1m = current_price * (1 + volatility * 3.5)
    
    return {
        "Current_Price": round(current_price, 2),
        "Target_1D": round(target_1d, 2),
        "Target_1W": round(target_1w, 2),
        "Target_1M": round(target_1m, 2),
        "CI_1D": [round(ci_lower_1d, 2), round(ci_upper_1d, 2)],
        "CI_1W": [round(ci_lower_1w, 2), round(ci_upper_1w, 2)],
        "CI_1M": [round(ci_lower_1m, 2), round(ci_upper_1m, 2)],
        "Expected_Move_Percent": round(expected_move_percent * 100, 2),
        "Volatility": round(volatility * 100, 2),
    }

def extract_evidence(signals_data, ticker):
    """Extract evidence for prediction rationale"""
    evidence = []
    
    # Technical evidence
    if "technical_analysis_agent" in signals_data:
        ta_data = signals_data["technical_analysis_agent"].get(ticker, {})
        if ta_data:
            # Extract key technical indicators - handle different possible naming conventions
            rsi_value = ta_data.get('RSI_value', ta_data.get('RSI', 'N/A'))
            macd_signal = ta_data.get('MACD_Signal', 'N/A')
            
            # Get trend information from various possible sources
            trend = ta_data.get('Trend', 'Unknown')
            if trend == 'Unknown' and 'Above_200d_SMA' in ta_data:
                trend = "Bullish" if ta_data['Above_200d_SMA'] == 1 else "Bearish"
            
            # Get counts of bullish/bearish indicators
            bullish = ta_data.get('Bullish_Indicators', 0)
            bearish = ta_data.get('Bearish_Indicators', 0)
            
            # Create a more comprehensive technical points list
            key_points = [f"Trend: {trend}"]
            
            # Add RSI information
            if rsi_value != 'N/A':
                rsi_value = float(rsi_value) if not isinstance(rsi_value, str) else 50
                rsi_condition = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                key_points.append(f"RSI: {rsi_value} ({rsi_condition})")
            else:
                key_points.append("RSI: N/A")
                
            # Add MACD information
            if macd_signal != 'N/A':
                if isinstance(macd_signal, (int, float)):
                    if macd_signal == 1:
                        macd_text = "Bullish"
                    elif macd_signal == -1:
                        macd_text = "Bearish"
                    else:
                        macd_text = "Neutral"
                    key_points.append(f"MACD: {macd_text}")
                else:
                    key_points.append(f"MACD: {macd_signal}")
            else:
                key_points.append("MACD: N/A")
            
            # Add indicator counts
            key_points.append(f"Bullish Indicators: {bullish}")
            key_points.append(f"Bearish Indicators: {bearish}")
            
            # Add volatility if available
            if 'ATR_Percent' in ta_data:
                key_points.append(f"Volatility (ATR): {ta_data['ATR_Percent']}%")
            elif 'Volatility' in ta_data:
                key_points.append(f"Volatility: {ta_data['Volatility']}%")
            
            # Add any pattern recognition findings
            patterns = []
            if ta_data.get('Doji', 0) > 0:
                patterns.append(f"Doji patterns: {ta_data['Doji']}")
            if ta_data.get('Hammer', 0) > 0:
                patterns.append(f"Hammer patterns: {ta_data['Hammer']}")
            if ta_data.get('Engulfing', 0) > 0:
                patterns.append(f"Engulfing patterns: {ta_data['Engulfing']}")
                
            if patterns:
                key_points.append("Price patterns: " + ", ".join(patterns))
            
            evidence.append({
                "source": "Technical Analysis",
                "signal": ta_data.get("Overall Signal", "NEUTRAL"),
                "confidence": ta_data.get("Confidence", 0.5),
                "key_points": key_points
            })
    
    # Fundamental evidence
    if "fundamental_analysis_agent" in signals_data:
        fa_data = signals_data["fundamental_analysis_agent"].get(ticker, {})
        if fa_data:
            evidence.append({
                "source": "Fundamental Analysis",
                "signal": fa_data.get("Overall Signal", "NEUTRAL"),
                "confidence": fa_data.get("Confidence", 0.5),
                "key_points": [
                    f"Valuation: {fa_data.get('Valuation_Summary', 'Unknown')}",
                    f"Growth: {fa_data.get('Growth_Score', 'N/A')}",
                    f"Value: {fa_data.get('Value_Score', 'N/A')}",
                    f"Sector Performance: {fa_data.get('Sector_Performance', 'Unknown')}"
                ]
            })
    
    # News evidence
    if "news_sentiment_analyst_agent" in signals_data:
        ns_data = signals_data["news_sentiment_analyst_agent"].get(ticker, {})
        if ns_data:
            evidence.append({
                "source": "News Sentiment",
                "signal": ns_data.get("Overall Signal", "NEUTRAL"),
                "confidence": ns_data.get("Confidence", 0.5),
                "key_points": [
                    f"Recent Trend: {ns_data.get('Recent Price Trend', 'Unknown')}",
                    f"Positive News: {ns_data.get('Positive Count', 0)}",
                    f"Negative News: {ns_data.get('Negative Count', 0)}",
                    f"Summary: {ns_data.get('News Summary', '')[:100]}..."
                ]
            })
    
    return evidence

def create_llm_prompt(ticker, base_prediction, price_estimates, evidence, portfolio_allocation=0):
    """Create a detailed prompt for LLM to generate qualitative insights"""
    prompt = f"""
You are a sophisticated financial analyst providing trade predictions for {ticker}. 
Our quantitative models have generated the following prediction:

Signal: {base_prediction["Signal"]}
Confidence: {base_prediction["Confidence"]} (0-1 scale)

=== PRICE TARGETS ===
Current Price: ₹{price_estimates["Current_Price"]}
1-Day Target: ₹{price_estimates["Target_1D"]} (Range: ₹{price_estimates["CI_1D"][0]} - ₹{price_estimates["CI_1D"][1]})
1-Week Target: ₹{price_estimates["Target_1W"]} (Range: ₹{price_estimates["CI_1W"][0]} - ₹{price_estimates["CI_1W"][1]})
1-Month Target: ₹{price_estimates["Target_1M"]} (Range: ₹{price_estimates["CI_1M"][0]} - ₹{price_estimates["CI_1M"][1]})
Expected Move: {price_estimates["Expected_Move_Percent"]}%
Volatility: {price_estimates["Volatility"]}%

=== EVIDENCE ===
"""

    # Add technical analysis evidence
    tech_evidence = next((e for e in evidence if e["source"] == "Technical Analysis"), None)
    if tech_evidence:
        prompt += f"TECHNICAL ANALYSIS (Signal: {tech_evidence['signal']}, Confidence: {tech_evidence['confidence']}):\n"
        for point in tech_evidence["key_points"]:
            prompt += f"- {point}\n"
        prompt += "\n"
    
    # Add fundamental analysis evidence
    fund_evidence = next((e for e in evidence if e["source"] == "Fundamental Analysis"), None)
    if fund_evidence:
        prompt += f"FUNDAMENTAL ANALYSIS (Signal: {fund_evidence['signal']}, Confidence: {fund_evidence['confidence']}):\n"
        for point in fund_evidence["key_points"]:
            prompt += f"- {point}\n"
        prompt += "\n"
    
    # Add news sentiment evidence
    news_evidence = next((e for e in evidence if e["source"] == "News Sentiment"), None)
    if news_evidence:
        prompt += f"NEWS SENTIMENT (Signal: {news_evidence['signal']}, Confidence: {news_evidence['confidence']}):\n"
        for point in news_evidence["key_points"]:
            prompt += f"- {point}\n"
        prompt += "\n"
    
    # Add portfolio allocation information if available
    if portfolio_allocation > 0:
        prompt += f"Current portfolio allocation for {ticker}: ₹{portfolio_allocation}\n\n"
    
    # Add the request for insights
    prompt += """
Based on this data, please provide:
1. A coherent rationale for the prediction, synthesizing the technical, fundamental, and news evidence
2. Key risk factors that could invalidate this prediction
3. A suggested position size/allocation (as a percentage) based on the confidence and risk
4. Any additional insights or patterns you observe in the data

IMPORTANT FORMATTING INSTRUCTIONS:
- Format your response as a JSON object with these keys: "rationale", "risks", "allocation", "additional_insights"
- For "risks", provide an array of 3-5 complete sentences, each identifying a specific risk factor
- For "allocation", provide a simple numeric percentage (e.g., 5 for 5%)
- Ensure "rationale" and "additional_insights" are well-formatted paragraphs without bullet points
- DO NOT include any markdown formatting, special characters, or individual character bullet points
- Ensure your response parses as valid JSON

Example response format:
{
  "rationale": "The technical indicators suggest a bullish trend...",
  "risks": ["Unexpected negative earnings could reverse the uptrend.", "Market-wide correction could impact all stocks including this one.", "Sector rotation away from technology could reduce institutional interest."],
  "allocation": 5,
  "additional_insights": "The stock is showing higher than average volume which confirms the trend..."
}
"""
    
    return prompt

def llm_enhanced_prediction(ticker, base_prediction, price_estimates, evidence, portfolio_allocation, llm):
    """Use LLM to enhance prediction with qualitative insights"""
    try:
        # Create detailed prompt for LLM
        prompt = create_llm_prompt(ticker, base_prediction, price_estimates, evidence, portfolio_allocation)
        
        # Get LLM response
        llm_response = llm.invoke(prompt)
        
        # Parse the response to extract JSON
        try:
            # First try direct JSON parsing
            llm_insights = json.loads(llm_response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response if direct parsing fails
            json_match = re.search(r'\{.*\}', llm_response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                llm_insights = json.loads(json_str)
            else:
                # Fallback with empty insights
                llm_insights = {
                    "rationale": "Could not parse LLM response.",
                    "risks": ["Parsing error"],
                    "allocation": 0,
                    "additional_insights": "Error in LLM processing."
                }
        
        # Return enhanced prediction
        return {
            **base_prediction,
            "LLM_Enhanced": True,
            "Rationale": llm_insights.get("rationale", "No rationale provided."),
            "Risk_Factors": llm_insights.get("risks", ["No risks identified."]),
            "Suggested_Allocation": llm_insights.get("allocation", 0),
            "Additional_Insights": llm_insights.get("additional_insights", "No additional insights.")
        }
        
    except Exception as e:
        print(f"Error in LLM enhancement: {str(e)}")
        # Return base prediction with error info
        return {
            **base_prediction,
            "LLM_Enhanced": False,
            "Rationale": "Error in LLM processing.",
            "Error": str(e)
        }

def trade_prediction_agent(state: State):
    """
    Hybrid trade prediction agent that combines deterministic ensemble methods with LLM-enhanced
    qualitative insights. This approach leverages both the consistency of algorithmic predictions
    and the contextual understanding of LLMs.
    
    Args:
    - state: The shared state containing signals from all other agents
    
    Returns:
    - Updated state with trade prediction signals
    """
    tickers = state["data"]["tickers"]
    signals_data = state["data"]["analyst_signals"]
    portfolio = state["data"]["portfolio"]
    predictions = {}
    
    # Setup LLM with consistent parameters
    groq_api_key = ""
    
    # Create LLM with temperature 0.1 for slight creativity but mostly consistent outputs
    llm = ChatGroq(
        api_key=groq_api_key,
        model="mixtral-8x7b-32768",
        temperature=0.1,  # Low temperature for consistency with slight variation
        seed=42,  # Fixed seed for reproducibility
    )
    
    # For timestamp-based seed to ensure day-to-day consistency
    # but allow changes as days progress
    today = datetime.now().strftime("%Y%m%d")
    day_seed = int(today) % 10000
    random.seed(42 + day_seed)
    np.random.seed(42 + day_seed)
    
    for i, ticker in enumerate(tickers):
        # Get current price - first try from technical analysis data, then fetch directly if needed
        current_price = None
        if "technical_analysis_agent" in signals_data:
            ta_data = signals_data["technical_analysis_agent"].get(ticker, {})
            if "Current_Price" in ta_data:
                current_price = ta_data["Current_Price"]
        
        # If we couldn't get current price from technical analysis, fetch it directly
        if current_price is None:
            try:
                stock = yf.Ticker(ticker)
                history = stock.history(period="1d")
                if not history.empty:
                    # Use the most recent close price
                    current_price = float(history['Close'].iloc[-1])
                    print(f"Fetched current price for {ticker}: ₹{current_price:.2f}")
                else:
                    # Fallback to default if we can't get a price
                    current_price = 100.0
                    print(f"Warning: Using default price for {ticker}")
            except Exception as e:
                print(f"Error fetching price for {ticker}: {str(e)}")
                current_price = 100.0  # Default fallback
        
        try:
            # Step 1: Generate base prediction using deterministic ensemble methods
            base_prediction = ensemble_prediction(signals_data, ticker)
            
            # Step 2: Estimate price targets and ranges
            price_estimates = estimate_price_range(signals_data, ticker, current_price)
            
            # Step 3: Extract evidence for the prediction
            evidence = extract_evidence(signals_data, ticker)
            
            # Step 4: Use LLM to enhance prediction with qualitative insights
            portfolio_allocation = portfolio[i] if i < len(portfolio) else 0
            enhanced_prediction = llm_enhanced_prediction(
                ticker, 
                base_prediction, 
                price_estimates, 
                evidence, 
                portfolio_allocation,
                llm
            )
            
            # Step 5: Create final prediction by combining all components
            predictions[ticker] = {
                "Overall_Signal": enhanced_prediction["Signal"],
                "Confidence": enhanced_prediction["Confidence"],
                "Price_Targets": price_estimates,
                "Evidence": evidence,
                "Rationale": enhanced_prediction.get("Rationale", "No rationale available."),
                "Risk_Factors": enhanced_prediction.get("Risk_Factors", ["No risks identified."]),
                "Suggested_Allocation": enhanced_prediction.get("Suggested_Allocation", 0),
                "Additional_Insights": enhanced_prediction.get("Additional_Insights", ""),
                "Prediction_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Analysis_Summary": f"The {enhanced_prediction['Signal']} signal is based on weighted evidence from technical, fundamental, and news analysis with {enhanced_prediction['Confidence']*100:.1f}% confidence.",
                "Technical_Weight": enhanced_prediction["Technical_Weight"],
                "Fundamental_Weight": enhanced_prediction["Fundamental_Weight"],
                "News_Weight": enhanced_prediction["News_Weight"],
                "Agent_Agreement": enhanced_prediction["Agent_Agreement"],
                "LLM_Enhanced": enhanced_prediction.get("LLM_Enhanced", False)
            }
        except Exception as e:
            print(f"Error generating prediction for {ticker}: {str(e)}")
            # Fallback to a neutral prediction
            predictions[ticker] = {
                "Overall_Signal": "NEUTRAL",
                "Confidence": 0.5,
                "Error": str(e),
                "Prediction_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    final_result_tp = HumanMessage(
        content=json.dumps(predictions),
        name="trade_prediction_agent",
    )

    state["data"]["trade_predictions"] = predictions

    return {
        "agent_actions": state["agent_actions"] + [final_result_tp],
        "data": state["data"]
    }
