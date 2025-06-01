import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def plot_temps(start_time, end_time):
    df = pd.read_json('data/climate_edu/global_warming.jsonl', lines=True, dtype={'time': str})
    df = df.round(2)
    filtered_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)] 
    fig, ax = plt.subplots(figsize=(12, 6))

    #  Create the line plot
    sns.lineplot(x='time', y='station', data=filtered_df, ax=ax)

    # Set labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Temperature')
    ax.set_title('Average Global Temperature Over Time')

    # Handle x-axis ticks
    ticks = ax.get_xticks()
    tick_positions = np.linspace(ticks[0], ticks[-1], 10, dtype=int) if len(ticks) > 10 else ticks
    ax.set_xticks(tick_positions)
    ax.tick_params(axis='x', rotation=45)

    # Adjust layout2000.01'
    plt.tight_layout()
    return fig, ax

def plot_co2(start_time, end_time):
    df = pd.read_json('data/climate_edu/co2.jsonl', lines=True, dtype={'time': str})
    df = df.round(2)
    filtered_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)] 
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the line plot
    sns.lineplot(x='time', y='cycle', data=filtered_df, ax=ax)

    # Set labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('CO2 Emissions (ppm)')
    ax.set_title('CO2 Emissions Over Time')

    # Handle x-axis ticks
    ticks = ax.get_xticks()
    tick_positions = np.linspace(ticks[0], ticks[-1], 10, dtype=int) if len(ticks) > 10 else ticks
    ax.set_xticks(tick_positions)
    ax.tick_params(axis='x', rotation=45)

    # Adjust layout
    plt.tight_layout()
    return fig, ax