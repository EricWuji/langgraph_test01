from typing import Dict, List, Any, Optional, Union
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.callbacks.manager import CallbackManagerForToolRun

# 这里要导入我们修改过的工具类
from src.tools.retriever import RetrieverTool
from src.tools.multiply import MultiplyTool

# 定义可用工具
AVAILABLE_TOOLS = {
    "retriever": RetrieverTool,
    "multiply": MultiplyTool
}

class ToolManager(Runnable):
    """Manages the execution of various tools based on the query"""
    
    def __init__(self, tools: Optional[List[str]] = None):
        """Initialize the tool manager with specified tools"""
        self.tools = tools or list(AVAILABLE_TOOLS.keys())
        self._tool_instances = {}
        
        # Initialize tool instances
        for tool_name in self.tools:
            if tool_name in AVAILABLE_TOOLS:
                self._tool_instances[tool_name] = AVAILABLE_TOOLS[tool_name]()
    
    def invoke(self, query: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Execute all tools and combine their results"""
        results = {}
        
        run_manager = None
        if config and config.get("callbacks"):
            run_manager = config["callbacks"]
            
        if run_manager:
            run_manager.on_text(f"Executing {len(self.tools)} tools for query: {query}")
        
        # Run each tool and collect results
        for tool_name, tool_instance in self._tool_instances.items():
            if run_manager:
                run_manager.on_text(f"Executing {tool_name} tool")
            
            try:
                tool_result = tool_instance.invoke(query, config=config)
                # Merge tool results with the main results
                results.update(tool_result)
                
                if run_manager:
                    run_manager.on_text(f"{tool_name} tool execution completed")
            except Exception as e:
                if run_manager:
                    run_manager.on_text(f"Error executing {tool_name} tool: {str(e)}")
        
        # Determine which tools returned useful results
        active_tools = []
        if results.get("multiply_result"):
            active_tools.append("multiply")
        if results.get("retriever_result") and results.get("retriever_result") != "No relevant health records found.":
            active_tools.append("retriever")
            
        results["active_tools"] = active_tools
        results["route_after_tools"] = "@".join(active_tools) if active_tools else None
        
        return results


def call_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    """Function to call all tools based on the query (for use with LangGraph)"""
    query = state.get("query", "")
    
    tool_manager = ToolManager()
    return tool_manager.invoke(query)