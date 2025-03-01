import json
from langchain_core.messages import HumanMessage
from state import State
from utils import llm_prediction
from langchain_groq import ChatGroq

def trade_prediction_agent(state: State):
    tickers = state["data"]["tickers"]
    analyst_signals = state["data"]["analyst_signals"]
    portfolio = state["data"]["portfolio"]
    groq_api_key = "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";
    llm = ChatGroq(
    api_key=groq_api_key,
    model="mixtral-8x7b-32768",
    temperature=0.3,
    )
    trade_preds = llm_prediction(tickers,analyst_signals,portfolio,llm)
    final_result_pred = HumanMessage(
    content=json.dumps(trade_preds.content),
    name="trade_prediction_agent",
    )

    return {
        "agent_actions": state["agent_actions"] + [final_result_pred],
        "data": state["data"]
    }