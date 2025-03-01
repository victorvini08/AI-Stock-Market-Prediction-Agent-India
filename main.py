import argparse
import json
import ast
import warnings
from langgraph.graph import END, StateGraph
from datetime import datetime
from dateutil.relativedelta import relativedelta
from langchain_core.messages import HumanMessage
from state import State
from agents.technical_analysis_agent import technical_analysis_agent
from agents.fundamental_analysis_agent import fundamental_analysis_agent
from agents.news_sentiment_analysis_agent import news_sentiment_analysis_agent
from agents.trade_prediction_agent import trade_prediction_agent

def start(state: State):
    """Initialize the workflow with the input message."""
    return state

def create_graph():

    graph = StateGraph(State)

    graph.add_node("start_node",start)

    nodes = {
        "technical_analyst": ("technical_analyst_agent", technical_analysis_agent),
        "fundamentals_analyst": ("fundamental_analysis_agent", fundamental_analysis_agent),
        "news_analyst": ("news_sentiment_analysis_agent", news_sentiment_analysis_agent),
    }

    graph.add_node("trade_prediction_agent", trade_prediction_agent)
    for key in nodes:
        node_name,node_func = nodes[key]
        graph.add_node(node_name,node_func)
        graph.add_edge("start_node",node_name)
        graph.add_edge(node_name,"trade_prediction_agent")

    graph.add_edge("trade_prediction_agent", END)
    graph.set_entry_point("start_node")

    return graph

def main():

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Stock Market AI Agent with input arguments")
    parser.add_argument("--tickers",nargs='+', type=str, help="List of stock tickers")
    parser.add_argument("--funds", nargs='+', type=int, help="List of maximum funds for each stock")
    parser.add_argument("--startdate",type=int, default=9, help="Number of months before today from which you want to start analysis of the stocks")
    # Input arguments
    args = parser.parse_args()
    tickers = args.tickers
    portfolio = args.funds
    months_neg = (args.startdate)*-1
    start_datetime = datetime.today() + relativedelta(months=months_neg)
    start_date = start_datetime.strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')

    graph = create_graph()
    agent = graph.compile()

    final_state = agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Make trading decisions based on the provided data.",
                        )
                    ],
                    "data": {
                        "tickers": tickers,
                        "portfolio": portfolio,
                        "start_date": start_date,
                        "end_date": end_date,
                        "analyst_signals": {},
                    },
                },
            )
    content_str = final_state['agent_actions'][-1].content
    print(json.loads(content_str))
    #print(json.loads())
if __name__=="__main__":
    main()