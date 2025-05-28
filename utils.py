import re

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