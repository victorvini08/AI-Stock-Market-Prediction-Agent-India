import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
from langchain_core.messages import HumanMessage
from state import State
from datetime import datetime

def calculate_advanced_indicators(data):
    """Calculate advanced technical indicators for stock data"""
    # Make a copy to avoid SettingWithCopyWarning
    df = data.copy()
    
    # Initialize indicators dictionary
    indicators = {}
    
    # Current price and basic data
    indicators['Close'] = df['Close'].iloc[-1]
    indicators['Current_Price'] = df['Close'].iloc[-1]  # Add for compatibility
    
    # Trend indicators
    try:
        # Moving Averages - handle empty dataframes gracefully
        if len(df) >= 20:
            indicators['SMA_20'] = df.ta.sma(length=20).iloc[-1]
        else:
            indicators['SMA_20'] = df['Close'].iloc[-1]
            
        if len(df) >= 50:
            indicators['SMA_50'] = df.ta.sma(length=50).iloc[-1]
        else:
            indicators['SMA_50'] = df['Close'].iloc[-1]
            
        if len(df) >= 200:
            indicators['SMA_200'] = df.ta.sma(length=200).iloc[-1]
        else:
            indicators['SMA_200'] = df['Close'].iloc[-1]
            
        if len(df) >= 20:
            indicators['EMA_20'] = df.ta.ema(length=20).iloc[-1]
        else:
            indicators['EMA_20'] = df['Close'].iloc[-1]
            
        if len(df) >= 50:
            indicators['EMA_50'] = df.ta.ema(length=50).iloc[-1]
        else:
            indicators['EMA_50'] = df['Close'].iloc[-1]
            
        # Calculate if price is above 200-day SMA
        indicators['Above_200d_SMA'] = 1 if indicators['Close'] > indicators['SMA_200'] else 0
        
    except Exception as e:
        print(f"Error calculating moving averages: {e}")
        # Fallback if pandas_ta doesn't work
        indicators['SMA_20'] = df['Close'].rolling(window=min(20, len(df))).mean().iloc[-1]
        indicators['SMA_50'] = df['Close'].rolling(window=min(50, len(df))).mean().iloc[-1]
        indicators['SMA_200'] = df['Close'].rolling(window=min(200, len(df))).mean().iloc[-1]
        indicators['EMA_20'] = df['Close'].ewm(span=min(20, len(df))).mean().iloc[-1]
        indicators['EMA_50'] = df['Close'].ewm(span=min(50, len(df))).mean().iloc[-1]
        indicators['Above_200d_SMA'] = 1 if indicators['Close'] > indicators['SMA_200'] else 0
    
    # Momentum indicators
    try:
        # RSI
        if len(df) >= 14:
            rsi = df.ta.rsi(length=14)
            indicators['RSI'] = rsi.iloc[-1]
            indicators['RSI_value'] = indicators['RSI']  # For compatibility with other code
        else:
            indicators['RSI'] = 50  # Neutral value
            indicators['RSI_value'] = 50
        
        # MACD
        if len(df) >= 26:  # Need at least 26 days for MACD
            macd = df.ta.macd(fast=12, slow=26, signal=9)
            indicators['MACD'] = macd.iloc[-1, 0]  # MACD line
            indicators['MACD_Signal'] = macd.iloc[-1, 1]  # Signal line
            indicators['MACD_Hist'] = macd.iloc[-1, 2]  # Histogram
            
            # MACD signal for compatibility
            if indicators['MACD'] > indicators['MACD_Signal']:
                indicators['MACD_Signal'] = 1  # Bullish
            elif indicators['MACD'] < indicators['MACD_Signal']:
                indicators['MACD_Signal'] = -1  # Bearish
            else:
                indicators['MACD_Signal'] = 0  # Neutral
        else:
            indicators['MACD'] = 0
            indicators['MACD_Signal'] = 0
            indicators['MACD_Hist'] = 0
        
        # Stochastic
        if len(df) >= 14:
            stoch = df.ta.stoch(k=14, d=3, smooth_k=3)
            indicators['Stochastic_K'] = stoch.iloc[-1, 0]  # %K line
            indicators['Stochastic_D'] = stoch.iloc[-1, 1]  # %D line
        else:
            indicators['Stochastic_K'] = 50
            indicators['Stochastic_D'] = 50
            
    except Exception as e:
        print(f"Error calculating momentum indicators: {e}")
        # Fallback calculations for RSI
        indicators['RSI'] = 50
        indicators['RSI_value'] = 50
        indicators['MACD'] = 0
        indicators['MACD_Signal'] = 0
        indicators['MACD_Hist'] = 0
        indicators['Stochastic_K'] = 50
        indicators['Stochastic_D'] = 50
    
    # Volatility indicators
    try:
        # ATR - Average True Range
        if len(df) >= 14:
            atr = df.ta.atr(length=14)
            indicators['ATR'] = atr.iloc[-1]
            
            # Calculate ATR as percentage of price for easier interpretation
            indicators['ATR_Percent'] = (indicators['ATR'] / indicators['Close']) * 100
        else:
            indicators['ATR'] = indicators['Close'] * 0.02  # Default 2% volatility
            indicators['ATR_Percent'] = 2.0
        
        # Bollinger Bands
        if len(df) >= 20:
            bbands = df.ta.bbands(length=20, std=2)
            # Handle NaN values in bbands
            if not bbands.iloc[-1, 0] is None and not pd.isna(bbands.iloc[-1, 0]):
                indicators['Bollinger_Upper'] = bbands.iloc[-1, 0]  # Upper band
                indicators['Bollinger_Middle'] = bbands.iloc[-1, 1]  # Middle band (SMA)
                indicators['Bollinger_Lower'] = bbands.iloc[-1, 2]  # Lower band
                
                # Bollinger Band Width - measure of volatility
                indicators['BB_Width'] = (indicators['Bollinger_Upper'] - indicators['Bollinger_Lower']) / indicators['Bollinger_Middle']
                
                # Determine if price is near bands (potential reversal)
                close_to_upper = (indicators['Close'] / indicators['Bollinger_Upper']) > 0.95
                close_to_lower = (indicators['Close'] / indicators['Bollinger_Lower']) < 1.05
                indicators['Near_Upper_Band'] = 1 if close_to_upper else 0
                indicators['Near_Lower_Band'] = 1 if close_to_lower else 0
            else:
                # Default values if bbands has NaN
                indicators['Bollinger_Upper'] = indicators['Close'] * 1.1
                indicators['Bollinger_Middle'] = indicators['Close']
                indicators['Bollinger_Lower'] = indicators['Close'] * 0.9
                indicators['BB_Width'] = 0.2
                indicators['Near_Upper_Band'] = 0
                indicators['Near_Lower_Band'] = 0
        else:
            # Default values for short time series
            indicators['Bollinger_Upper'] = indicators['Close'] * 1.1
            indicators['Bollinger_Middle'] = indicators['Close']
            indicators['Bollinger_Lower'] = indicators['Close'] * 0.9
            indicators['BB_Width'] = 0.2
            indicators['Near_Upper_Band'] = 0
            indicators['Near_Lower_Band'] = 0
            
    except Exception as e:
        print(f"Error calculating volatility indicators: {e}")
        # Fallback for Bollinger Bands
        indicators['Bollinger_Upper'] = indicators['Close'] * 1.1
        indicators['Bollinger_Middle'] = indicators['Close']
        indicators['Bollinger_Lower'] = indicators['Close'] * 0.9
        indicators['BB_Width'] = 0.2
        indicators['ATR'] = indicators['Close'] * 0.02  # Default 2% volatility
        indicators['ATR_Percent'] = 2.0
        indicators['Near_Upper_Band'] = 0
        indicators['Near_Lower_Band'] = 0
    
    # Volume indicators
    try:
        # On-Balance Volume
        if 'Volume' in df.columns and len(df) > 1:
            obv = df.ta.obv()
            indicators['OBV'] = obv.iloc[-1]
            
            # Volume trend (compare recent volume to average)
            recent_vol = df['Volume'].iloc[-5:].mean()
            avg_vol = df['Volume'].iloc[-20:].mean() if len(df) >= 20 else df['Volume'].mean()
            indicators['Volume_Ratio'] = recent_vol / avg_vol if avg_vol > 0 else 1.0
        else:
            indicators['OBV'] = 0
            indicators['Volume_Ratio'] = 1.0
    except Exception as e:
        print(f"Error calculating volume indicators: {e}")
        indicators['OBV'] = 0
        indicators['Volume_Ratio'] = 1.0
    
    # Trend strength indicator
    try:
        # ADX - Average Directional Index
        if len(df) >= 14:
            adx_df = df.ta.adx(length=14)
            if adx_df is not None and not adx_df.empty and not pd.isna(adx_df.iloc[-1, 0]):
                indicators['ADX'] = adx_df.iloc[-1, 0]  # ADX value
            else:
                indicators['ADX'] = 25  # Default value
        else:
            indicators['ADX'] = 25  # Default value for short time series
    except Exception as e:
        print(f"Error calculating ADX: {e}")
        indicators['ADX'] = 25  # Fallback default
    
    # Price patterns - pandas_ta doesn't have direct candlestick pattern recognition
    # We'll implement simplified versions
    try:
        # Check last 5 candles for patterns (or fewer if not enough data)
        max_lookback = min(5, len(df)-1)
        doji_count = 0
        hammer_count = 0
        engulfing_count = 0
        
        for i in range(-max_lookback, 0):
            try:
                # Doji - open and close are almost the same
                body = abs(df['Open'].iloc[i] - df['Close'].iloc[i])
                total_range = df['High'].iloc[i] - df['Low'].iloc[i]
                if total_range > 0 and body / total_range < 0.1:  # Body is less than 10% of total range
                    doji_count += 1
                
                # Hammer - small body at the top, long lower shadow
                if i > -max_lookback:  # Need to check the previous candle for engulfing
                    body_current = abs(df['Open'].iloc[i] - df['Close'].iloc[i])
                    upper_shadow = df['High'].iloc[i] - max(df['Open'].iloc[i], df['Close'].iloc[i])
                    lower_shadow = min(df['Open'].iloc[i], df['Close'].iloc[i]) - df['Low'].iloc[i]
                    
                    # Hammer condition
                    if (lower_shadow > 2 * body_current) and (upper_shadow < 0.2 * body_current):
                        hammer_count += 1
                    
                    # Bullish engulfing
                    if (df['Close'].iloc[i] > df['Open'].iloc[i] and  # Current is bullish
                        df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and  # Previous is bearish
                        df['Open'].iloc[i] < df['Close'].iloc[i-1] and  # Current open below previous close
                        df['Close'].iloc[i] > df['Open'].iloc[i-1]):  # Current close above previous open
                        engulfing_count += 1
            except Exception as e:
                print(f"Error in candlestick pattern calculation at index {i}: {e}")
                continue
        
        indicators['Doji'] = doji_count
        indicators['Hammer'] = hammer_count
        indicators['Engulfing'] = engulfing_count
    except Exception as e:
        print(f"Error calculating candlestick patterns: {e}")
        indicators['Doji'] = 0
        indicators['Hammer'] = 0
        indicators['Engulfing'] = 0
    
    # Calculate distance from 52-week high and low
    try:
        if len(df) >= 252:  # Approximately 1 year of trading days
            year_data = df[-252:]
            year_high = year_data['High'].max()
            year_low = year_data['Low'].min()
            current_price = df['Close'].iloc[-1]
            
            indicators['Distance_From_52W_High'] = (current_price / year_high - 1) * 100  # Negative value means below high
            indicators['Distance_From_52W_Low'] = (current_price / year_low - 1) * 100  # Positive value means above low
        else:
            # Default values for shorter time series
            indicators['Distance_From_52W_High'] = 0
            indicators['Distance_From_52W_Low'] = 0
    except Exception as e:
        print(f"Error calculating 52-week metrics: {e}")
        indicators['Distance_From_52W_High'] = 0
        indicators['Distance_From_52W_Low'] = 0
    
    # Add volatility metric for price target calculations
    indicators['Volatility'] = indicators.get('ATR_Percent', 2.0)
    
    return indicators

def analyze_multi_timeframe(ticker, start_date, end_date):
    """Analyze stock across multiple timeframes"""
    timeframes = {
        'Daily': '1d',
        'Weekly': '1wk',
        'Monthly': '1mo'
    }
    
    multi_tf_data = {}
    stock = yf.Ticker(ticker)
    
    for tf_name, tf_code in timeframes.items():
        try:
            data = stock.history(start=start_date, end=end_date, interval=tf_code)
            if len(data) > 0:
                multi_tf_data[tf_name] = calculate_advanced_indicators(data)
                multi_tf_data[tf_name]['Close'] = data['Close'].iloc[-1]
            else:
                multi_tf_data[tf_name] = {'Error': f"No data available for {tf_name} timeframe"}
        except Exception as e:
            multi_tf_data[tf_name] = {'Error': str(e)}
    
    return multi_tf_data

def generate_technical_signals(indicators, multi_tf_data):
    """Generate technical signals based on indicators"""
    signals = {}
    confidence_factors = []
    
    # Current price and basic signals
    current_price = indicators.get('Close', 0)
    signals['Current Price'] = current_price
    
    # Trend signals
    if indicators.get('SMA_20', 0) > indicators.get('SMA_50', 0):
        signals['Short_Term_Trend'] = 'BULLISH'
        confidence_factors.append(0.6)
    elif indicators.get('SMA_20', 0) < indicators.get('SMA_50', 0):
        signals['Short_Term_Trend'] = 'BEARISH'
        confidence_factors.append(-0.6)
    else:
        signals['Short_Term_Trend'] = 'NEUTRAL'
        confidence_factors.append(0.1)
    
    if indicators.get('SMA_50', 0) > indicators.get('SMA_200', 0):
        signals['Long_Term_Trend'] = 'BULLISH'  # Golden Cross
        confidence_factors.append(0.8)
    elif indicators.get('SMA_50', 0) < indicators.get('SMA_200', 0):
        signals['Long_Term_Trend'] = 'BEARISH'  # Death Cross
        confidence_factors.append(-0.8)
    else:
        signals['Long_Term_Trend'] = 'NEUTRAL'
        confidence_factors.append(0.1)
    
    # RSI signals
    rsi = indicators.get('RSI', 50)
    if rsi < 30:
        signals['RSI'] = 'BULLISH'  # Oversold
        # Higher confidence the lower RSI goes
        confidence_score = min(1.0, (30 - rsi) / 20)
        confidence_factors.append(confidence_score)
    elif rsi > 70:
        signals['RSI'] = 'BEARISH'  # Overbought
        # Higher confidence the higher RSI goes
        confidence_score = min(1.0, (rsi - 70) / 20)
        confidence_factors.append(-confidence_score)
    else:
        signals['RSI'] = 'NEUTRAL'
        confidence_factors.append(0.1)
    
    # MACD signals
    if indicators.get('MACD_Hist', 0) > 0 and indicators.get('MACD', 0) > indicators.get('MACD_Signal', 0):
        signals['MACD'] = 'BULLISH'
        confidence_factors.append(0.7)
    elif indicators.get('MACD_Hist', 0) < 0 and indicators.get('MACD', 0) < indicators.get('MACD_Signal', 0):
        signals['MACD'] = 'BEARISH'
        confidence_factors.append(-0.7)
    else:
        signals['MACD'] = 'NEUTRAL'
        confidence_factors.append(0.1)
    
    # Bollinger Bands signals
    if current_price <= indicators.get('Bollinger_Lower', 0):
        signals['Bollinger_Bands'] = 'BULLISH'  # Price at lower band - potential buy
        confidence_factors.append(0.65)
    elif current_price >= indicators.get('Bollinger_Upper', 0):
        signals['Bollinger_Bands'] = 'BEARISH'  # Price at upper band - potential sell
        confidence_factors.append(-0.65)
    else:
        # Calculate position within bands as percentage
        upper = indicators.get('Bollinger_Upper', current_price * 1.1)
        lower = indicators.get('Bollinger_Lower', current_price * 0.9)
        band_width = upper - lower
        if band_width > 0:
            position = (current_price - lower) / band_width  # 0 = at lower band, 1 = at upper band
            if position < 0.4:
                signals['Bollinger_Bands'] = 'SLIGHTLY BULLISH'
                confidence_factors.append(0.3)
            elif position > 0.6:
                signals['Bollinger_Bands'] = 'SLIGHTLY BEARISH'
                confidence_factors.append(-0.3)
            else:
                signals['Bollinger_Bands'] = 'NEUTRAL'
                confidence_factors.append(0.1)
        else:
            signals['Bollinger_Bands'] = 'NEUTRAL'
            confidence_factors.append(0.1)
    
    # ADX signals (trend strength)
    adx = indicators.get('ADX', 25)
    if adx > 25:
        signals['ADX'] = f'STRONG TREND ({adx:.1f})'
        # Don't add confidence factor here, just indicates strength of existing trend
    else:
        signals['ADX'] = f'WEAK TREND ({adx:.1f})'
    
    # Volume confirmation
    # Typically high volume confirms the trend direction
    signals['Volume'] = 'NORMAL'  # Default
    
    # Support and resistance levels based on Bollinger Bands
    signals['Support'] = indicators.get('Bollinger_Lower', current_price * 0.95)
    signals['Resistance'] = indicators.get('Bollinger_Upper', current_price * 1.05)
    
    # Multi-timeframe confluence
    # Check if signals align across timeframes for stronger conviction
    tf_trends = {}
    for tf, tf_indicators in multi_tf_data.items():
        if 'Error' not in tf_indicators:
            # Simple trend determination based on position relative to MAs
            if tf_indicators.get('SMA_20', 0) > tf_indicators.get('SMA_50', 0):
                tf_trends[tf] = 'BULLISH'
            elif tf_indicators.get('SMA_20', 0) < tf_indicators.get('SMA_50', 0):
                tf_trends[tf] = 'BEARISH'
            else:
                tf_trends[tf] = 'NEUTRAL'
    
    # Count bullish and bearish timeframes
    bullish_count = sum(1 for trend in tf_trends.values() if trend == 'BULLISH')
    bearish_count = sum(1 for trend in tf_trends.values() if trend == 'BEARISH')
    
    if bullish_count > bearish_count and bullish_count >= 2:
        signals['Multi_Timeframe'] = 'BULLISH'
        confidence_factors.append(0.75)
    elif bearish_count > bullish_count and bearish_count >= 2:
        signals['Multi_Timeframe'] = 'BEARISH'
        confidence_factors.append(-0.75)
    else:
        signals['Multi_Timeframe'] = 'MIXED/NEUTRAL'
        confidence_factors.append(0.1)
    
    # Calculate overall signal and confidence
    bullish_factors = sum(factor for factor in confidence_factors if factor > 0)
    bearish_factors = sum(factor for factor in confidence_factors if factor < 0)
    neutral_factors = sum(factor for factor in confidence_factors if factor == 0.1)
    
    # Total of absolute values of all confidence factors for normalization
    total_confidence = sum(abs(factor) for factor in confidence_factors)
    
    if total_confidence > 0:
        bullish_weight = bullish_factors / total_confidence
        bearish_weight = abs(bearish_factors) / total_confidence
        
        if bullish_weight > bearish_weight + 0.2:
            signals['Overall Signal'] = 'BULLISH'
            confidence = bullish_weight
        elif bearish_weight > bullish_weight + 0.2:
            signals['Overall Signal'] = 'BEARISH'
            confidence = bearish_weight
        else:
            signals['Overall Signal'] = 'NEUTRAL'
            confidence = max(0.3, (neutral_factors / total_confidence))
    else:
        signals['Overall Signal'] = 'NEUTRAL'
        confidence = 0.3
    
    # Add potential price targets based on technical levels
    current_price = indicators.get('Close', 0)
    
    if signals['Overall Signal'] == 'BULLISH':
        # Calculate resistance levels for price targets
        resistance1 = signals.get('Resistance', current_price * 1.05)
        signals['Price_Target_1'] = resistance1
        signals['Price_Target_2'] = resistance1 * 1.05  # 5% above resistance
    elif signals['Overall Signal'] == 'BEARISH':
        # Calculate support levels for price targets
        support1 = signals.get('Support', current_price * 0.95)
        signals['Price_Target_1'] = support1
        signals['Price_Target_2'] = support1 * 0.95  # 5% below support
    
    signals['Confidence'] = round(confidence, 2)
    
    return signals

def technical_analysis_agent(state: State):
    """
    Performs technical analysis on stock data using multiple indicators and timeframes.
    
    Args:
    - state: The shared state between agents
    
    Returns:
    - Updated state with technical analysis signals
    """
    tickers = state["data"]["tickers"]
    start_date = state["data"]["start_date"]
    end_date = state["data"]["end_date"]
    
    technical_analysis_signals = {}
    
    for ticker in tickers:
        try:
            # Fetch historical data
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if len(data) > 0:
                # Calculate indicators
                indicators = calculate_advanced_indicators(data)
                
                # Get current price for reference
                current_price = data['Close'].iloc[-1]
                indicators['Close'] = current_price
                
                # Analyze multiple timeframes
                multi_tf_data = analyze_multi_timeframe(ticker, start_date, end_date)
                
                # Generate technical signals
                signals = generate_technical_signals(indicators, multi_tf_data)
                
                technical_analysis_signals[ticker] = signals
            else:
                technical_analysis_signals[ticker] = {
                    "Error": "No data available for the specified date range",
                    "Overall Signal": "NEUTRAL",
                    "Confidence": 0.3
                }
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            technical_analysis_signals[ticker] = {
                "Error": str(e),
                "Overall Signal": "NEUTRAL",
                "Confidence": 0.3
            }
    
    # Create final message
    final_result_ta = HumanMessage(
        content=json.dumps(technical_analysis_signals),
        name="technical_analyst_agent",
    )
    
    state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis_signals
    
    return {
        "agent_actions": state["agent_actions"] + [final_result_ta],
        "data": state["data"]
    }

