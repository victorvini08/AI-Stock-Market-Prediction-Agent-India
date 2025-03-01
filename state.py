import operator
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

def merge_dicts(a: dict[str, any], b: dict[str, any]) -> dict[str, any]:
    return {**a, **b}

class State(TypedDict):
    agent_actions: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, any], merge_dicts]

