from typing import Dict, List, Any, TypedDict, Optional, Union
from enum import Enum

class RouteChoices(str, Enum):
    TOOLS = "tools"
    MULTIPLY = "multiply"
    RETRIEVER = "retriever"
    GENERATE = "generate"
    REWRITE = "rewrite"
    END = "end"

class State(TypedDict):
    # Input and intermediate state
    tools_condition: Optional[bool]
    route_after_tools: Optional[str]
    route_after_grade: Optional[str]
    
    # Input/output data
    documents: Optional[List[Dict[str, Any]]]
    query: Optional[str]
    results: Optional[Dict[str, Any]]
    
    # Tool execution results
    multiply_result: Optional[Any]
    retriever_result: Optional[Any]
    generate_result: Optional[str]
    rewrite_result: Optional[str]
    
    # Final output
    final_output: Optional[str]