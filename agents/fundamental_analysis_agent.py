import yfinance as yf
import numpy as np
import json
import pandas as pd
from datetime import datetime
from langchain_core.messages import HumanMessage
from state import State

def get_sector_performance(ticker):
    """Get industry and sector performance data"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get sector and industry
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # You could expand this to fetch actual sector performance data
        # For now, we'll return the classifications
        return {
            'sector': sector,
            'industry': industry
        }
    except Exception as e:
        print(f"Error fetching sector data: {e}")
        return {
            'sector': 'Unknown',
            'industry': 'Unknown',
            'error': str(e)
        }

def calculate_financial_ratios(stock_data):
    """Calculate advanced financial ratios"""
    ratios = {}
    info = stock_data.info
    
    # Valuation ratios
    ratios['Market Cap (in Crores)'] = info.get('marketCap', 0) / 10000000  # Convert to crores (1 crore = 10M)
    
    # P/E ratio
    if 'trailingPE' in info:
        ratios['P/E'] = info['trailingPE']
    elif 'forwardPE' in info:
        ratios['P/E (Forward)'] = info['forwardPE']
    
    # Price to Book
    if 'priceToBook' in info:
        ratios['P/B'] = info['priceToBook']
    
    # Price to Sales
    if 'priceToSalesTrailing12Months' in info:
        ratios['P/S'] = info['priceToSalesTrailing12Months']
    
    # EV/EBITDA
    if 'enterpriseToEbitda' in info:
        ratios['EV/EBITDA'] = info['enterpriseToEbitda']
    
    # Dividend metrics
    if 'dividendYield' in info and info['dividendYield'] is not None:
        ratios['Dividend Yield'] = info['dividendYield'] * 100  # Convert to percentage
    
    # Profitability ratios
    if 'returnOnEquity' in info:
        ratios['ROE'] = info['returnOnEquity'] * 100  # Convert to percentage
    
    if 'returnOnAssets' in info:
        ratios['ROA'] = info['returnOnAssets'] * 100  # Convert to percentage
    
    if 'profitMargins' in info:
        ratios['Profit Margin'] = info['profitMargins'] * 100  # Convert to percentage
    
    if 'operatingMargins' in info:
        ratios['Operating Margin'] = info['operatingMargins'] * 100  # Convert to percentage
    
    # Debt ratios
    if 'debtToEquity' in info:
        ratios['Debt/Equity'] = info['debtToEquity']
    
    # Growth ratios
    if 'earningsQuarterlyGrowth' in info and info['earningsQuarterlyGrowth'] is not None:
        ratios['Quarterly Earnings Growth'] = info['earningsQuarterlyGrowth'] * 100  # Convert to percentage
    
    if 'revenueGrowth' in info and info['revenueGrowth'] is not None:
        ratios['Revenue Growth'] = info['revenueGrowth'] * 100  # Convert to percentage
    
    # Efficiency ratios
    # These would require more complex calculations using financial statements
    
    return ratios

def calculate_intrinsic_value(stock_data):
    """
    Calculate intrinsic value using multiple methods
    
    1. Discounted Cash Flow (DCF)
    2. Graham Number
    3. PEG-based valuation
    """
    valuation_data = {}
    info = stock_data.info
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    valuation_data['Current Price'] = current_price
    
    # Method 1: Simple DCF for growth stocks
    try:
        if 'freeCashflow' in info and info['freeCashflow'] > 0:
            # Simplified DCF
            fcf = info['freeCashflow']
            growth_rate = info.get('earningsGrowth', 0.10)  # Default to 10% if not available
            if growth_rate <= 0:
                growth_rate = 0.10  # Fallback
            
            # Cap growth rate at reasonable levels
            growth_rate = min(growth_rate, 0.25)
            
            # Terminal growth rate
            terminal_growth = 0.03  # 3% terminal growth
            
            # Discount rate (WACC or required return)
            discount_rate = 0.12  # 12% discount rate
            
            # Calculate 5-year DCF
            dcf_value = 0
            for year in range(1, 6):
                projected_fcf = fcf * ((1 + growth_rate) ** year)
                dcf_value += projected_fcf / ((1 + discount_rate) ** year)
            
            # Terminal value
            terminal_value = (fcf * (1 + growth_rate) ** 5 * (1 + terminal_growth)) / (discount_rate - terminal_growth)
            terminal_value_discounted = terminal_value / ((1 + discount_rate) ** 5)
            
            # Total value
            total_value = dcf_value + terminal_value_discounted
            
            # Shares outstanding
            shares_outstanding = info.get('sharesOutstanding', 1)
            
            # Value per share
            dcf_value_per_share = total_value / shares_outstanding
            
            valuation_data['DCF Value'] = dcf_value_per_share
            valuation_data['DCF Upside/Downside'] = (dcf_value_per_share / current_price - 1) * 100  # % difference
    except Exception as e:
        print(f"DCF calculation error: {e}")
    
    # Method 2: Graham Number (for value stocks)
    try:
        if 'bookValue' in info and 'trailingEps' in info and info['trailingEps'] > 0:
            book_value = info['bookValue']
            eps = info['trailingEps']
            
            # Original Graham formula: √(22.5 * EPS * BVPS)
            graham_number = np.sqrt(22.5 * eps * book_value)
            
            valuation_data['Graham Value'] = graham_number
            valuation_data['Graham Upside/Downside'] = (graham_number / current_price - 1) * 100  # % difference
    except Exception as e:
        print(f"Graham calculation error: {e}")
    
    # Method 3: PEG-based valuation
    try:
        if 'trailingPE' in info and 'earningsGrowth' in info and info['earningsGrowth'] > 0:
            pe = info['trailingPE']
            growth = info['earningsGrowth'] * 100  # Convert to percentage
            
            peg = pe / growth
            
            # PEG interpretation
            if peg < 1:
                valuation_data['PEG Ratio'] = peg
                valuation_data['PEG Valuation'] = 'Potentially Undervalued'
            elif peg <= 1.5:
                valuation_data['PEG Ratio'] = peg
                valuation_data['PEG Valuation'] = 'Fairly Valued'
            else:
                valuation_data['PEG Ratio'] = peg
                valuation_data['PEG Valuation'] = 'Potentially Overvalued'
    except Exception as e:
        print(f"PEG calculation error: {e}")
    
    # Final intrinsic value estimate (weighted average of different methods)
    try:
        values = []
        weights = []
        
        if 'DCF Value' in valuation_data:
            values.append(valuation_data['DCF Value'])
            weights.append(0.5)  # Higher weight to DCF
            
        if 'Graham Value' in valuation_data:
            values.append(valuation_data['Graham Value'])
            weights.append(0.3)  # Medium weight to Graham
            
        if len(values) > 0:
            # Normalize weights
            weights = [w/sum(weights) for w in weights]
            
            # Weighted average
            intrinsic_value = sum(v * w for v, w in zip(values, weights))
            
            valuation_data['Intrinsic Value Estimate'] = intrinsic_value
            valuation_data['Estimated Upside/Downside'] = (intrinsic_value / current_price - 1) * 100  # % difference
    except Exception as e:
        print(f"Error calculating final intrinsic value: {e}")
    
    return valuation_data

def generate_fundamental_signal(ratios, valuation, sector_data):
    """Generate fundamental analysis signal based on valuation and ratios"""
    signal_data = {}
    confidence_factors = []
    
    # Check if we have valuation data
    if 'Intrinsic Value Estimate' in valuation:
        intrinsic_value = valuation['Intrinsic Value Estimate']
        current_price = valuation['Current Price']
        upside = valuation['Estimated Upside/Downside']
        
        # Determine signal based on estimated upside/downside
        if upside > 20:
            signal_data['Valuation'] = 'BULLISH'
            # Higher confidence with higher upside
            confidence_score = min(0.9, 0.5 + (upside - 20) / 100)
            confidence_factors.append(confidence_score)
        elif upside < -20:
            signal_data['Valuation'] = 'BEARISH'
            # Higher confidence with higher downside
            confidence_score = min(0.9, 0.5 + (abs(upside) - 20) / 100)
            confidence_factors.append(-confidence_score)
        else:
            signal_data['Valuation'] = 'NEUTRAL'
            confidence_factors.append(0.1)
        
        signal_data['Intrinsic Value'] = f"Rs.{intrinsic_value:.2f}"
        signal_data['Current Price'] = f"Rs.{current_price:.2f}"
        signal_data['Upside/Downside'] = f"{upside:.1f}%"
    else:
        signal_data['Valuation'] = 'NEUTRAL'
        confidence_factors.append(0.1)
    
    # Analyze P/E ratio
    if 'P/E' in ratios:
        pe = ratios['P/E']
        # Low P/E could be bullish, high P/E could be bearish
        # But needs industry context
        if pe < 10:
            signal_data['P/E Analysis'] = 'BULLISH'
            confidence_factors.append(0.7)
        elif pe > 30:
            signal_data['P/E Analysis'] = 'BEARISH'
            confidence_factors.append(-0.7)
        else:
            signal_data['P/E Analysis'] = 'NEUTRAL'
            confidence_factors.append(0.1)
    
    # Analyze profitability
    profitability_score = 0
    profitability_factors = 0
    
    if 'ROE' in ratios:
        roe = ratios['ROE']
        if roe > 15:
            profitability_score += 1
        elif roe < 5:
            profitability_score -= 1
        profitability_factors += 1
    
    if 'Profit Margin' in ratios:
        margin = ratios['Profit Margin']
        if margin > 10:
            profitability_score += 1
        elif margin < 3:
            profitability_score -= 1
        profitability_factors += 1
    
    if profitability_factors > 0:
        if profitability_score > 0:
            signal_data['Profitability'] = 'BULLISH'
            confidence_factors.append(0.6 * (profitability_score / profitability_factors))
        elif profitability_score < 0:
            signal_data['Profitability'] = 'BEARISH'
            confidence_factors.append(-0.6 * (abs(profitability_score) / profitability_factors))
        else:
            signal_data['Profitability'] = 'NEUTRAL'
            confidence_factors.append(0.1)
    
    # Analyze growth
    growth_score = 0
    growth_factors = 0
    
    if 'Quarterly Earnings Growth' in ratios:
        earnings_growth = ratios['Quarterly Earnings Growth']
        if earnings_growth > 15:
            growth_score += 1
        elif earnings_growth < 0:
            growth_score -= 1
        growth_factors += 1
    
    if 'Revenue Growth' in ratios:
        rev_growth = ratios['Revenue Growth']
        if rev_growth > 15:
            growth_score += 1
        elif rev_growth < 0:
            growth_score -= 1
        growth_factors += 1
    
    if growth_factors > 0:
        if growth_score > 0:
            signal_data['Growth'] = 'BULLISH'
            confidence_factors.append(0.8 * (growth_score / growth_factors))
        elif growth_score < 0:
            signal_data['Growth'] = 'BEARISH'
            confidence_factors.append(-0.8 * (abs(growth_score) / growth_factors))
        else:
            signal_data['Growth'] = 'NEUTRAL'
            confidence_factors.append(0.1)
    
    # Analyze debt
    if 'Debt/Equity' in ratios:
        debt_equity = ratios['Debt/Equity']
        if debt_equity < 0.5:
            signal_data['Debt'] = 'BULLISH'
            confidence_factors.append(0.5)
        elif debt_equity > 1.5:
            signal_data['Debt'] = 'BEARISH'
            confidence_factors.append(-0.5)
        else:
            signal_data['Debt'] = 'NEUTRAL'
            confidence_factors.append(0.1)
    
    # Calculate overall signal and confidence
    bullish_factors = sum(factor for factor in confidence_factors if factor > 0)
    bearish_factors = sum(factor for factor in confidence_factors if factor < 0)
    neutral_factors = sum(factor for factor in confidence_factors if factor == 0.1)
    
    # Total of absolute values of all confidence factors for normalization
    total_confidence = sum(abs(factor) for factor in confidence_factors) if confidence_factors else 1
    
    if total_confidence > 0:
        bullish_weight = bullish_factors / total_confidence
        bearish_weight = abs(bearish_factors) / total_confidence
        
        if bullish_weight > bearish_weight + 0.2:
            signal_data['Overall Signal'] = 'BULLISH'
            confidence = bullish_weight
        elif bearish_weight > bullish_weight + 0.2:
            signal_data['Overall Signal'] = 'BEARISH'
            confidence = bearish_weight
        else:
            signal_data['Overall Signal'] = 'NEUTRAL'
            confidence = max(0.3, (neutral_factors / total_confidence))
    else:
        signal_data['Overall Signal'] = 'NEUTRAL'
        confidence = 0.3
    
    signal_data['Confidence'] = round(confidence, 2)
    signal_data['Sector'] = sector_data.get('sector', 'Unknown')
    signal_data['Industry'] = sector_data.get('industry', 'Unknown')
    
    # Add key financial metrics for reference
    signal_data['Key Financials'] = {k: v for k, v in ratios.items() if k in [
        'Market Cap (in Crores)', 'P/E', 'P/B', 'ROE', 'Profit Margin', 'Revenue Growth'
    ]}
    
    return signal_data

def fundamental_analysis_agent(state: State):
    """
    Fetches fundamental data for a stock and evaluates it to generate trade signals.

    Args:
    - state (): The state shared between the graph.

    Returns:
    - dict: A dictionary containing fundamental metrics, trade signal, and confidence score.
    """
    tickers = state["data"]["tickers"]
    # Fetch company info and financials
    fundamental_analysis_signals = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            # Check if we can fetch basic info
            if not hasattr(stock, 'info') or not stock.info:
                fundamental_analysis_signals[ticker] = {
                    "Error": "Unable to fetch fundamental data",
                    "Overall Signal": "NEUTRAL",
                    "Confidence": 0.3
                }
                continue
            
            # Get sector and industry info
            sector_data = get_sector_performance(ticker)
            
            # Calculate financial ratios
            ratios = calculate_financial_ratios(stock)
            
            # Calculate intrinsic value
            valuation = calculate_intrinsic_value(stock)
            
            # Generate signal based on fundamentals
            signal_data = generate_fundamental_signal(ratios, valuation, sector_data)
            
            # Add to results
            fundamental_analysis_signals[ticker] = signal_data
            
        except Exception as e:
            print(f"Error analyzing fundamentals for {ticker}: {e}")
            fundamental_analysis_signals[ticker] = {
                "Error": str(e),
                "Overall Signal": "NEUTRAL",
                "Confidence": 0.3
            }

    final_result_fa = HumanMessage(
    content=json.dumps(fundamental_analysis_signals),
    name="fundamental_analyst_agent",
    )

    state["data"]["analyst_signals"]["fundamental_analyst_agent"] = fundamental_analysis_signals

    return {
        "agent_actions": state["agent_actions"] + [final_result_fa],
        "data": state["data"]
    }

