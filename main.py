# Set environment variable to disable tokenizers parallelism warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from utils import format_trade_prediction

def start(state: State):
    """Initialize the workflow with the input message."""
    return state

def create_graph():
    graph = StateGraph(State)

    graph.add_node("start_node", start)

    nodes = {
        "technical_analyst": ("technical_analyst_agent", technical_analysis_agent),
        "fundamentals_analyst": ("fundamental_analysis_agent", fundamental_analysis_agent),
        "news_analyst": ("news_sentiment_analysis_agent", news_sentiment_analysis_agent),
    }

    graph.add_node("trade_prediction_agent", trade_prediction_agent)
    for key in nodes:
        node_name, node_func = nodes[key]
        graph.add_node(node_name, node_func)
        graph.add_edge("start_node", node_name)
        graph.add_edge(node_name, "trade_prediction_agent")

    graph.add_edge("trade_prediction_agent", END)
    graph.set_entry_point("start_node")

    # Compile the graph to a runnable
    return graph.compile()

def main():
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Suppress specific warnings
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("langchain").setLevel(logging.ERROR)
    logging.getLogger("langchain_core").setLevel(logging.ERROR)
    logging.getLogger("langchain_community").setLevel(logging.ERROR)
    
    parser = argparse.ArgumentParser(description="Stock Market AI Agent with input arguments")
    parser.add_argument("--tickers", nargs='+', type=str, help="List of stock tickers")
    parser.add_argument("--funds", nargs='+', type=int, help="List of maximum funds for each stock")
    parser.add_argument("--startdate", type=int, default=9, help="Number of months before today from which you want to start analysis of the stocks")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON output instead of formatted display")
    parser.add_argument("--quiet", action="store_true", help="Suppress all warning messages")
    
    # Input arguments
    args = parser.parse_args()
    
    # Enable quiet mode if requested
    if args.quiet:
        import sys
        sys.stderr = open(os.devnull, 'w')

    # Validate inputs
    if not args.tickers or not args.funds:
        print("Error: Both tickers and funds must be provided")
        print("Example: python main.py --tickers RELIANCE.NS ZOMATO.NS --funds 10000 5000")
        return
    
    if len(args.tickers) != len(args.funds):
        print("Error: Number of tickers must match number of fund allocations")
        print(f"You provided {len(args.tickers)} tickers but {len(args.funds)} fund allocations")
        return
    
    tickers = args.tickers
    portfolio = args.funds
    months_neg = (args.startdate)*-1
    start_datetime = datetime.today() + relativedelta(months=months_neg)
    start_date = start_datetime.strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Create initial state
    state = State(
        agent_actions=[],
        data={
            "tickers": tickers,
            "portfolio": portfolio,
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {},
            "trade_predictions": {}
        },
    )
    
    try:
        print(f"Analyzing {len(tickers)} stocks: {', '.join(tickers)}")
        print(f"Analysis period: {start_date} to {end_date}")
        print("Running analysis, please wait...\n")
        
        # Create the graph
        app = create_graph()
        
        # Run the graph with invoke method
        final_state = app.invoke(state)
        
        # Print the final prediction in formatted or raw JSON format
        if "trade_predictions" in final_state["data"]:
            if args.raw:
                print(json.dumps(final_state["data"]["trade_predictions"], indent=2))
            else:
                formatted_output = format_trade_prediction(final_state["data"]["trade_predictions"])
                print(formatted_output)
        else:
            print("Error: No trade predictions found in the final state")
            
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__=="__main__":
    main()