import config
import langsmith
import os
api_key = os.getenv("OPENAI_API_KEY")

@config
class configuration:
    LANGCHAIN_TRACING_V2=True
    LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    LANGCHAIN_API_KEY=api_key # Key
    LANGCHAIN_PROJECT="langchain_monitoring"