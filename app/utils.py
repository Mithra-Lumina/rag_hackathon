import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Literal
from langchain_core.tools import tool
from matplotlib import figure

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

sys_message = """you have access to tools, ONLY RETURN A TOOL IF A USER ASKES YOU ABOUT 
TEMPERATURE AND CO2 LEVELS BETWEEN TWO DATES, Otherwise answer the question normally"""

def markdown_cleanup(md_text):
    md_text = re.sub('\n+', '\n', md_text)
    md_text = re.sub(r'^\n+', '', md_text)
    md_text = re.sub('\n', 'PARAGRAPH_BREAK', md_text)
    md_text = re.sub(r'\s+', ' ', md_text)
    md_text = md_text.replace('PARAGRAPH_BREAK', '\n')
    return md_text

def boolean_filter(documents):
    return {
    "bool": {
        "should": [
        {
            "terms": {
            "title": documents
            }
        },
        {
            "terms": {
            "title.keyword": documents
            }
        }
        ]
    }
    }

def opensearch_format(results):
    return {
        "hits": {
            "hits": results
        }
    }

def os_temp(thing):
    return {
        "_source": thing
    }

def plot_temps(start_time, end_time, out_path=None):
    df = pd.read_json('app/data/climate_edu/global_warming.jsonl', lines=True, dtype={'time': str})
    df = df.round(2)
    filtered_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)] 
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(x='time', y='station', data=filtered_df, ax=ax)

    ax.set_xlabel('Year')
    ax.set_ylabel('Average Temperature')
    ax.set_title('Average Global Temperature Over Time')

    ticks = ax.get_xticks()
    tick_positions = np.linspace(ticks[0], ticks[-1], 10, dtype=int) if len(ticks) > 10 else ticks
    ax.set_xticks(tick_positions)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300) 
    return fig, ax


def plot_co2(start_time, end_time, out_path=None):
    df = pd.read_json('app/data/climate_edu/co2.jsonl', lines=True, dtype={'time': str})
    df = df.round(2)
    filtered_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)] 
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(x='time', y='cycle', data=filtered_df, ax=ax)

    ax.set_xlabel('Year')
    ax.set_ylabel('CO2 Emissions (ppm)')
    ax.set_title('CO2 Emissions Over Time')

    ticks = ax.get_xticks()
    tick_positions = np.linspace(ticks[0], ticks[-1], 10, dtype=int) if len(ticks) > 10 else ticks
    ax.set_xticks(tick_positions)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300)
    return fig, ax

class TempGraphInputSchema(BaseModel):
    """Start and end dates for temperature graph generation."""
    start_date: str = Field(description="Start date in the format YYYY.MM")
    end_date: str = Field(description="End date in the format YYYY.MM")

@tool("create_temperature_graph", args_schema=TempGraphInputSchema)
def create_temperature_graph(start_date: str, end_date: str) -> figure:
    return plot_temps(start_date, end_date, out_path="app/plot_output/temperature_graph.png")

class Co2GraphInputSchema(BaseModel):
    """Start and end dates for temperature graph generation."""
    start_date: str = Field(description="Start date in the format YYYY.MM.DD")
    end_date: str = Field(description="End date in the format YYYY.MM.DD")

@tool("create_co2_graph", args_schema=Co2GraphInputSchema)
def create_co2_graph(start_date: str, end_date: str) -> figure:
    return plot_co2(start_date, end_date, out_path="app/plot_output/co2_graph.png")


def router(
    state: list[BaseMessage],
) -> Literal["create_temperature_graph","create_co2_graph", "__end__"]:
    """Creates tempreture graph."""
    tool_calls = state[-1].tool_calls

    if tool_calls:
        function_name = tool_calls[0].get("name")
        if function_name == "create_temperature_graph":
            return "create_temperature_graph"
        elif function_name == "create_co2_graph":
            return "create_co2_graph"
    else:
        return "__end__"
    
class LangGraphApp:
    def set_up(self, ibm_chat) -> None:
        model = ibm_chat
        builder = MessageGraph()

        model_with_tools = model.bind_tools([create_temperature_graph, create_co2_graph])
        builder.add_node("tools", model_with_tools)

        tool_node = ToolNode([create_temperature_graph, create_co2_graph])
        builder.add_node("create_temperature_graph", tool_node)
        builder.add_node("create_co2_graph", tool_node)
        builder.add_edge("create_temperature_graph", END)
        builder.add_edge("create_co2_graph", END)

        builder.set_entry_point("tools")
        builder.add_conditional_edges("tools", router)
        self.app = builder.compile()

    def query(self, input: str):
        """Query the application."""
        chat_history = (self.app.invoke([SystemMessage(sys_message),HumanMessage(input)]))
        return chat_history