import re
import logging
from langgraph.graph.message import MessageGraph
from typing import List
def format_response(response):
    paragraphs = re.split(r'\n{2,}', response)
    formatted_paragraphs = []
    for para in paragraphs:
        if '```' in para:
            parts = para.split('```')
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = ''.join(parts)
        else:
            para = para.replace('. ', '.\n')
        formatted_paragraphs.append(para.strip())
    return '\n\n'.join(formatted_paragraphs)

def save_graph_visualization(graph, filename: str = "graph.png"):
    try:
        with open(filename, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        logging.info(f"Graph visualization saved as {filename}")
    except IOError as e:
        logging.info(f"Warning: Failed to save graph visualization: {str(e)}")

def filter_messages(state: List):
    if len(state) <= 3:
        return state
    return state[-3:]