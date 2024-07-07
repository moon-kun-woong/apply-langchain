from langchain.prompts import ChatPromptTemplate
from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
import os
api_key = os.getenv("OPENAI_API_KEY")

class configuration:
    LANGCHAIN_TRACING_V2=True
    LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    LANGCHAIN_API_KEY=api_key
    LANGCHAIN_PROJECT="langchain_monitoring"
    
configuration

llm = ChatOpenAI(
    openai_api_key=api_key, # API 찾아서 넣을것
    model="gpt-3.5-turbo",
   temperature=0.1,
    streaming=True,
    verbose=True,
)

pg_uri = f"postgresql+psycopg2://postgres:postgres@localhost:5432/discord_bot_task"

db = SQLDatabase.from_uri(database_uri=pg_uri)

db_chain = SQLDatabaseChain.from_llm(llm, db , verbose=True)

store = LocalFileStore("cache/")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

custom_suffix = """
    먼저 내가 알고 있는 비슷한 예제를 가져와야 합니다.
    예제가 쿼리를 구성하기에 충분하다 판단된다면 쿼리를 작성해 줘.
    그렇지 않다면 데이터베이스의 테이블을 살펴보고 쿼리할 수 있는 항목을 확인해.
    그런 다음 가장 관련성이 높은 테이블의 스키마를 쿼리해 와줘.
    priority 는 중요도를 나타내며, 숫자가 낮을수록 중요하다.
"""


few_shots = { #지정된 '단어'나 '문장' 에 관해 특정 sql 쿼리를 검색하는 틀을 만드는 것.
    "내가 할일을 리스트로 만들어줘.":"SELECT * FROM task",
    "내 업무 중에서 가장 중요한 일들만 리스트로 보여줘.":"SELECT * FROM task WHERE priority = (SELECT MIN(priority) FROM task);",
    "누군가와 연락해야하는 업무가 있다면 알려주겠어?":"SELECT * FROM task WHERE content LIKE '%연락%';",
    "최근에 만들어진 태스크를 리스트로 보여줘":"SELECT * FROM task ORDER BY created_at DESC LIMIT 10;",
    "해야 할 일을 추가할게 '나는 준무에게 상황 보고를 해야해":"INSERT INTO task (channel_name, content, priority, created_at, server_name, updated_at, user_id) VALUES ('task_manager', '나는 준무에게 상황 보고를 해야해', 1, NOW(), 'my_server', NOW(), 'your_user_id');",
    "삭제할 업무가 있어 '준무에게 상황 보고 하는 일'을 삭제해줘.":"DELETE FROM task WHERE content = '준무에게 상황 보고 하는 일'",
    "다 끝난 업무가 있어 '준무에게 상황 보고 하는 일'이 완료되었으니 삭제해줘.":"DELETE FROM task WHERE content = '준무에게 상황 보고 하는 일' AND completed = true;",
}

embeddings = OpenAIEmbeddings(openai_api_key=api_key)  # API 찾아서 넣을것


cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store , namespace=embeddings.model
)

few_shots_docs = [
    Document(page_content = question, metadata = {"sql_query": few_shots[question]})
    for question in few_shots.keys()
]

vector_db = FAISS.from_documents(few_shots_docs, cached_embedder)

retriever = vector_db.as_retriever()

tool_description = """
    이 도구는 유사한 예시를 이해하여 사용자 질문에 적용하는 데 도움이 된다.
    이 도구에 입력하는 내용은 사용자 질문이어야 한다.
"""

retriever_tool = create_retriever_tool(
    retriever, name= "sql_get_similar", description=tool_description
)

custom_tool_list = [retriever_tool]

agent = create_sql_agent(
    llm=llm, 
    toolkit=toolkit,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    extra_tools=custom_tool_list,
    suffix=custom_suffix,
    verbose=True,
)
def outResponce():
    return agent.invoke("업무 리스트를 보여주겠어?")