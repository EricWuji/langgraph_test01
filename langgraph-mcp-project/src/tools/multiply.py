def multiply_tool(state):
    """A simple tool that multiplies two numbers from the query"""
    query = state.get("query", "")
    
    # Extract numbers from the query (very basic implementation)
    # In a real application, this would be more sophisticated
    import re
    numbers = re.findall(r'\d+', query)
    
    if len(numbers) >= 2:
        result = int(numbers[0]) * int(numbers[1])
        return {"multiply_result": result}
    else:
        return {"multiply_result": None}