from typing import Dict, List, Any, Optional
import re
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.callbacks.manager import CallbackManagerForToolRun


class MultiplyTool(Runnable):
    """A tool for multiplying numbers found in a query."""
    
    def extract_numbers(self, query: str) -> List[float]:
        """Extract numbers from the query string"""
        # Find all numbers (integers or decimals)
        number_pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(number_pattern, query)
        
        # Convert string matches to float
        numbers = [float(match) for match in matches]
        return numbers
    
    def invoke(self, query: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Execute the multiplication tool with the given query"""
        # 获取回调管理器（如果有）
        run_manager = None
        if config and config.get("callbacks"):
            run_manager = config["callbacks"]
        
        # 记录操作
        if run_manager:
            run_manager.on_text("Extracting numbers for multiplication: " + query)
        
        # 提取数字
        numbers = self.extract_numbers(query)
        
        # 检查是否至少有2个数字可以相乘
        if len(numbers) < 2:
            if run_manager:
                run_manager.on_text("Not enough numbers found for multiplication")
            return {"multiply_result": None}
        
        # 将所有数字相乘
        result = 1
        for num in numbers:
            result *= num
            
        # 为了更好的可读性而格式化结果
        operation = " × ".join([str(num) for num in numbers])
        formatted_result = f"Multiplication: {operation} = {result}"
        
        # 记录完成
        if run_manager:
            run_manager.on_text(f"Multiplication result: {result}")
        
        return {
            "multiply_result": formatted_result,
            "raw_result": result,
            "numbers": numbers
        }


# 兼容旧代码的函数
def multiply_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """A simple tool that multiplies numbers from the query (legacy API)"""
    query = state.get("query", "")
    
    tool = MultiplyTool()
    result = tool.invoke(query)
    
    return {"multiply_result": result["multiply_result"]}