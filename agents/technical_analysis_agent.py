import yfinance as yf
import pandas_ta as ta
import json
from langchain_core.messages import HumanMessage
from state import State
from datetime import datetime

def technical_analysis_agent(state:State):
    tickers = state["data"]["tickers"]
    start_date_str = state["data"]["start_date"]
    end_date_str = state["data"]["end_date"]

    start_date = datetime.strptime(start_date_str,"%Y-%m-%d")
    end_date = datetime.strptime(end_date_str,"%Y-%m-%d")

    technical_analysis_signals = {}
    data = yf.download(tickers,start=start_date,end=end_date)
    for ticker in tickers:
        stock_data = data.xs(ticker,level=1,axis=1)
        if stock_data.empty:
            print(f"No stock data found for given ticker: {ticker}")
            continue

        stock_data["EMA_5"] = ta.ema(stock_data["Close"], length=5)  # 5-day Exponential Moving Average
        stock_data["EMA_26"] = ta.ema(stock_data["Close"], length=26)  # 26-day Exponential Moving Average

        macd = ta.macd(stock_data["Close"])
        stock_data['MACD'] = macd['MACD_12_26_9']
        stock_data['MACD_signal'] = macd['MACDs_12_26_9']
        stock_data["RSI"] = ta.rsi(stock_data["Close"], length=14)  # Relative Strength Index

        stock_data["ADX"] = ta.adx(stock_data["High"], stock_data["Low"], stock_data["Close"])["ADX_14"]  # ADX
        stock_data["VWAP"] = ta.vwap(stock_data["High"], stock_data["Low"], stock_data["Close"], stock_data["Volume"])  # VWAP

        stoch = ta.stoch(stock_data["High"], stock_data["Low"], stock_data["Close"])  # Stochastic Oscillator
        stock_data["Stochastic_%K"], stock_data["Stochastic_%D"] = stoch["STOCHk_14_3_3"], stoch["STOCHd_14_3_3"]

        stock_data['Aroon_Up'] = ta.aroon(stock_data['High'],stock_data["Low"])['AROONU_14']
        stock_data['Aroon_Down'] = ta.aroon(stock_data['High'],stock_data["Low"])['AROOND_14']

        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        strategy_signals = {}
        # Generate trade signals from indicators
        if stock_data["RSI"].iloc[-1] < 30:
            bullish_signals += 1  # Oversold
            strategy_signals["RSI"] = "Bullish";
        elif stock_data["RSI"].iloc[-1] > 70:
            bearish_signals += 1  # Overbought
            strategy_signals["RSI"] = "Bearish";
        else:
            strategy_signals["RSI"] = "Neutral"

        total_signals += 1

        if stock_data["MACD"].iloc[-1] > stock_data["MACD_signal"].iloc[-1]:
            bullish_signals += 1  # MACD bullish crossover
            strategy_signals["MACD"] = "Bullish";
        else:
            bearish_signals += 1  # MACD bearish crossover
            strategy_signals["MACD"] = "Bearish";
        total_signals += 1

        if stock_data["EMA_5"].iloc[-1] > stock_data["EMA_26"].iloc[-1]:
            bullish_signals += 1
            strategy_signals["EMA"] = "Bullish";
        else:
            bearish_signals += 1
            strategy_signals["EMA"] = "Bearish";
        total_signals += 1

        if stock_data["ADX"].iloc[-1] > 25:
            if stock_data["Close"].iloc[-1] > stock_data["VWAP"].iloc[-1]:
                bullish_signals += 1  # Strong upward trend
                strategy_signals["ADX"] = "Bullish";
            else:
                bearish_signals += 1  # Strong downward trend
                strategy_signals["ADX"] = "Bearish";
        total_signals += 1


        if stock_data["Stochastic_%K"].iloc[-1] < 20:
            bullish_signals += 1  # Oversold condition
            strategy_signals["Stochastic"] = "Bullish";
        elif stock_data["Stochastic_%K"].iloc[-1] > 80:
            bearish_signals += 1  # Overbought condition
            strategy_signals["Stochastic"] = "Bearish";
        total_signals += 1

        if stock_data["Aroon_Up"].iloc[-1] > stock_data["Aroon_Down"].iloc[-1]:
            bullish_signals += 1
            strategy_signals["Aroon"] = "Bullish";
        else:
            bearish_signals += 1
            strategy_signals["Aroon"] = "Bearish";
        total_signals += 1

        #Denominator is taken as total signals because you need to count neutral signals from the indicators as well.
        confidence = bullish_signals / total_signals if bullish_signals > bearish_signals else bearish_signals / total_signals

        overall_signal = "Bullish"
        if bullish_signals > bearish_signals:
            overall_signal = "Bullish"
        elif bullish_signals < bearish_signals:
            overall_signal = "Bearish"
        else:
            overall_signal = "Neutral"

        technical_analysis_signals[ticker] = {
            "Confidence": confidence,
            "Overall Signal": overall_signal,
            "Technical Signals": strategy_signals
        }

    final_result_ta = HumanMessage(
    content=json.dumps(technical_analysis_signals),
    name="technical_analyst_agent",
    )

    state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis_signals

    return {
        "agent_actions": state["agent_actions"] + [final_result_ta],
        "data": state["data"]
    }

