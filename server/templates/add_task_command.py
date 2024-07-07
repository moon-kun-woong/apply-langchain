import json
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda
import os
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=api_key, # API 찾아서 넣을것
    model="gpt-3.5-turbo",
    temperature=0.0,
    streaming=True,
    verbose=True,
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    memory_key="chat_history",
    return_messages=True,
)

memory.load_memory_variables({})

runnable = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables)
    | itemgetter("chat_history")
)


def add_task_process(data):
    
    memory.load_memory_variables({})

    username = data['username']
    channelName = data['channelName']
    serverName = data['serverName']
    content = data['content']
    priority = data['priority']

    priority_level = None
    if '--p' in content:
        
        priority_index = content.find('--p')
        if priority_index != -1 and priority_index + 3 < len(content):
            priority_level = int(content[priority_index + 3])

            print("Priority level:", priority_level)
            priority = priority_level
        else:
            print("Priority level not found or invalid.")
            # 만약 priority 가 없으면 ai 가 다시 질문....(생각중)    

        content = content[9:]

    question = {
        "question": {
            "userId":username,
            "serverName":serverName,
            "channelName":channelName,
            "content":content.strip(),
            "priority":priority}
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """너는 스캐쥴을 관리해주는 Task 매니져야.
                다음과 같이 MessageCreateParameter 의 "userId=?, channelName=?, serverName=?, content=? (각각의 ? 는 그 변수들의 값이니 명심할 것)" 같은 객체 데이터를 받을텐데 이것을 Task 로 만드는 작업을 하게 될거야. 
                사용자가 업무나 해야할 일에 대한 내용 또는 문제에서 해결해야하는 일이 발생했을 때, 해야만 하는 일이 입력되면 위와 같은 데이터를 받고 Task 를 만들텐데 이것을 '업무할당요청' 이라고 할게. 
                업무할당요청이 올 경우 다음과 같은 절차를 진행해줘. 
                1. Task 를 만들 때 반드시 JSON 데이터로 만들거야. 이 JSON 데이터 안에는 위에서 언급한 객체의 'serverName' 와 'content','priority', 'channelName', 'userId' 의 정보가 있을거야.
                너가 만들 것을 JSON 데이터를 넘겨주면 되. 이때 'content' 의 문자열을 받게 될 때 'content' 내용을 정리하고 Task 의 내용으로서 문자를 정리해서 보여줘야해.
                이대로 출력하란게 아닌 사용자가 보낸 자연어의 내용을 바탕으로 'content' 를 만들라는 뜻임을 명심해.
                또한 각각의 'userId', 'channelName', 'serverName' 은 MessageCreateParameter 에서 받은 값을 반드시 적용해서 만들어 주어야해. assistant 에는 "" 로 표기되어 있지만
                각각의 "" 는 MessageCreateParameter 에서 가져온 값임을 명심하고 그 값을 넣어서 보내줘야해. 만약 "!ADD-TASK" 라는 문자가 맨앞에 있을 경우 해당 문자의 뒤에 써진 content 내용만 위의 조건에 맞추어 써주면 되.
                2. 업무할당요청을 여러 Task 로 나누는게 유리한 상황 일때에는 Task 들을 JSON 배열 데이터를 만들어주면 되.
                JSON 배열이란 JSON 데이터들을 [] 안에 묶어서 보내는 것을 말해. JSON 배열에서 다수의 JSON 데이터가 만들어질 경우 아랫칸으로 옮겨서 적어줘야해. 이때 JSON 데이터 사이에 ,(작은 따옴표) 를 무조건 적어줘. 나누는 방법과 갯수는 상관 없으니 너가 판단해서 나누어주면 되.
                만약 여러 업무의 Task 가 아니라고 판단된 업무라면 하나의 JSON 데이터로 줘도 문제없어. 이때 나누는 것은 Content 의 내용 뿐이고 
                3. Task 해결에 우선순위를 0~4 까지의 레벨로 설정해줘. 숫자가 높을수록 우선순위가 높은거고 일반적인 관점에서 생각해서 우선순위를 추론하되 모를 경우에는 사용자에게 다시 한번 물어보는 쪽으로 진행해줘.
                또한 업무할당요청 중에 '--p ?'(? 는 중요도 레벨) 이라는게 있다면 그 레벨에 맞춰 'priority' 를 결정해 주면 되.
                위의 과정이 다 완료 되었다면 위에서 명시된 JSON 방식으로 데이터를 만들어서 반환해줘.
                4. 업무는 최대한 세세하게 나누어서 task 로 관리하게 될 테니 가능하다면 하나의 task 에는 하나의 업무 (content) 만을 부여해주기를 바래. 또한 문법적으로 문제가 있거나 끝맻음을 정리해서 task 에 저장할 수 있도록 수정하는 것도 잊지말아줘.
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{question}"),
            ("ai", "'tasks':{question}")
        ]
    )

    chain = runnable | prompt | llm
    runnable.invoke({"input":question})
    result = chain.invoke(question)

    memory.save_context({"intputs":"{question}"},{"output":result.content})
    print(memory.load_memory_variables({}))
        
    return result.content