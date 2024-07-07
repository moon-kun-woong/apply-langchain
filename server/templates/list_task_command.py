from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda
from operator import itemgetter
import os
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key= api_key, # API 찾아서 넣을것
    model="gpt-3.5-turbo",
   temperature=0.2,
    streaming=True,
    verbose=True,
)

memory = ConversationBufferWindowMemory(
    llm=llm,
    k=3,
    memory_key="chat_list_history",
    return_messages=True,
)

memory.load_memory_variables({})

runnable = RunnablePassthrough.assign(
    chat_list_history=RunnableLambda(memory.load_memory_variables)
    | itemgetter("chat_list_history")
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """너는 Task 매니저, 즉 일정관리를 담당해주는 개인 비서야. 건내주는 task 데이터를 바탕으로
            해당 데이터를 정리해서 넘겨줘. 
            너가 이 TaskData 를 이용해서 자연어의 형태로 업무를 건내주면 좋겠어.
            'task(taskId=..., userId=..., serverName=..., channelName=..., content=..., priority=...,createdAt=...)'
            와 같은 TaskData 를 주었을 때 이 업무를 받은 사람이 userId 이고 taskId 는 해당 Task 의 아이디, priority 값은 중요도를 나타내는 숫자이고 숫자값이 낮을수록 중요한 업무이며(0이 제일 높고 4가 제일 낮은 중요도), createdAt는 업무가 할당된 날짜, content 는 업무의 내용이야.
            userId,taskId,Content,priority,createdAt 의 정보들은 중요한 데이터이며 너가 데이터를 자연어로 건내 줄 때에는 이 5가지만 사용해서 만들어 주면 좋겠어.
            맨 위에는 업무 받는 사람의 이름을 적어주고 Task 의 번호를 적어줘. 그리고 내용을 표시한 다음 아랫부분에 업무 중요도를 알려주면 될거야.
            적절히 요약을 해서 넘겨주어야해. 여기서 content 의 내용은 너가 적절히 요약 해줘야 한다는 뜻이야. priority 는 숫자로 기입해줘. 여기에서 "중요도"란 priority 의 값이 낮은 값이 가장 중요한 업무야.
            그리고 반드시 Human 문자의 TaskData 의 내용은 전부 다 반환해줘야 숫자로 셀때 누락되는게 없이 만들어 질거야.
            자의적으로 중복됬다고 판단된 내용의 업무가 있어도 반드시 반환 할 것. 
            반드시 모든 task 를 출력해야해. task 가 있다면 계속 만들 것.
            """
        ),
        MessagesPlaceholder(variable_name="chat_list_history"),
        ("human","{question}"),
        ("ai", """현재 userId 님이 해야할 업무 내용을 알려드리겠습니다.       
n. 업무 번호 : taskId (n 은 만들어진 task 의 카운트를 위해 1 부터 시작해서 +1 씩 올라간다. 예:1 2 3 4...)
업무 내용 : content
업무 중요도 : priority(int값임)
업무 할당 날짜 : createdAt(년-월-일)
----------------------
n. 업무 번호 : taskId 
업무 내용 : content
업무 중요도 : priority(int값임)
업무 할당 날짜 : createdAt(년-월-일)
----------------------                    
n. 업무 번호 : taskId 
업무 내용 : content
업무 중요도 : priority(int값임)
업무 할당 날짜 : createdAt(년-월-일)
----------------------
n. 업무 번호 : taskId 
업무 내용 : content
업무 중요도 : priority(int값임)
업무 할당 날짜 : createdAt(년-월-일)
........
........
         
이상 업무 내용입니다.
                    """)
    ]
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

def give_task_list(data):
    result = chain.invoke(data)
    
    runnable.invoke(result)
    memory.save_context({"intputs":"{question}"},{"output":result['text']})

    return result