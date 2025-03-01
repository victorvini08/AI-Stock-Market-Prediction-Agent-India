import yfinance as yf
import json
from langchain_core.messages import HumanMessage
from state import State

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
        stock = yf.Ticker(ticker)

        # Initialize results dictionary
        fundamentals = {
            "P/E Ratio": None,
            "P/B Ratio": None,
            "Debt/Equity Ratio": None,
            "EPS Growth (YoY)": None,
            "Revenue Growth (YoY)": None,
            "Profit Growth (YoY)": None,
            "Free Cash Flow Growth (YoY)": None,
        }

        # Evaluate key fundamental metrics
        bullish_count = 0
        total_count = 0

        try:
            # Price-to-Earnings (P/E) Ratio
            pe_ratio = stock.info.get("trailingPE")
            if pe_ratio and pe_ratio < 25:  # P/E < 25 is generally considered good
                bullish_count += 1
                fundamentals["P/E Ratio"] = "Bullish"
            else:
                fundamentals["P/E Ratio"] = "Bearish/Neutral"
            total_count += 1

            # Price-to-Book (P/B) Ratio
            pb_ratio = stock.info.get("priceToBook")
            if pb_ratio and pb_ratio < 3:  # P/B < 3 is generally considered good
                bullish_count += 1
                fundamentals["P/B Ratio"] = "Bullish"
            else:
                fundamentals["P/B Ratio"] = "Bearish/Neutral"
            total_count += 1

            # Debt-to-Equity Ratio
            debt_equity_ratio = stock.info.get("debtToEquity")
            if debt_equity_ratio and debt_equity_ratio < 1.5:  # D/E < 1 is generally considered good
                bullish_count += 1
                fundamentals["Debt/Equity Ratio"] = "Bullish"
            else:
                fundamentals["Debt/Equity Ratio"] = "Bearish/Neutral"
            total_count += 1

            # EPS Growth (YoY)
            try:
                earnings = stock.financials.loc['Net Income', :].pct_change()
                eps_growth = earnings[-1] * 100  # Percentage growth
                if eps_growth and eps_growth > 10:  # EPS growth > 10% is a bullish sign
                    bullish_count += 1
                    fundamentals["EPS Growth (YoY)"] = "Bullish"
                else:
                    fundamentals["EPS Growth (YoY)"] = "Bearish/Neutral"
                total_count += 1
            except Exception:
                pass

            # Revenue Growth (YoY)
            try:
                revenue = stock.financials.loc['Total Revenue', :].pct_change()
                revenue_growth = revenue[-1] * 100
                if revenue_growth and revenue_growth > 10:
                    bullish_count += 1
                    fundamentals["Revenue Growth (YoY)"] = "Bullish"
                else:
                    fundamentals["Revenue Growth (YoY)"] = "Bearish/Neutral"
                total_count += 1
            except Exception:
                pass

            # Profit Growth (YoY)
            try:
                profit = stock.financials.loc['Gross Profit', :].pct_change()
                profit_growth = profit[-1] * 100
                if profit_growth and profit_growth > 10:
                    bullish_count += 1
                    fundamentals["Profit Growth (YoY)"] = "Bullish"
                else:
                    fundamentals["Profit Growth (YoY)"] = "Bearish/Neutral"
                total_count += 1
            except Exception:
                pass

            # Free Cash Flow Growth (YoY)
            try:
                cash_flow = stock.cashflow.loc['Total Cash From Operating Activities', :].pct_change()
                fcf_growth = cash_flow[-1] * 100
                if fcf_growth and fcf_growth > 10:
                    bullish_count += 1
                    fundamentals["Free Cash Flow Growth (YoY)"] = "Bullish"
                else:
                    fundamentals["Free Cash Flow Growth (YoY)"] = "Bearish/Neutral"
                total_count += 1
            except Exception:
                pass


            # Generate trade signal and confidence score
            confidence_score = bullish_count / total_count if total_count > 0 else 0
            signal = "Bullish" if confidence_score > 0.5 else "Bearish"

            fundamental_analysis_signals[ticker] = {
            "Confidence": confidence_score,
            "Overall Signal": signal,
            "Fundamental Signals": fundamentals
        }


        except Exception as e:
            print(f"Error processing fundamentals for {ticker}: {e}")

    final_result_fa = HumanMessage(
    content=json.dumps(fundamental_analysis_signals),
    name="fundamental_analyst_agent",
    )

    state["data"]["analyst_signals"]["fundamental_analyst_agent"] = fundamental_analysis_signals

    return {
        "agent_actions": state["agent_actions"] + [final_result_fa],
        "data": state["data"]
    }

