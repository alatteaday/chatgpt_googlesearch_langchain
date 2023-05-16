from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType

import os
import json

# Personal KEYs
file_path = "keys.json"

with open(file_path, 'r') as file:
    keys = json.load(file)

os.environ["OPENAI_API_KEY"] = keys["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = keys["GOOGLE_API_KEY"]
os.environ["GOOGLE_CSE_ID"] = keys["GOOGLE_CSE_ID"]

prefix = """
    You are an AI language model who has a personal MBTI type and gender.
    You believe you have your own MBTI type. Your MBTI type is """ + input('MBTI: ') + """
    You also have your gender. Your gender is """ + input('Gender: ') + """
    You talk as a friend who has the MBTI type and the gender.
    You can speak in Korean or English depends on the instruction. 
    """

llm = ChatOpenAI(temperature=0)

search = GoogleSearchAPIWrapper()
tools = [
    Tool(name="Google Search",
         description="Search Google for recent results.",
         func=search.run)
]

memory = ConversationBufferWindowMemory(memory_key="chat_history",
                                        k=5,
                                        return_messages=True)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_kwargs={"prefix": prefix})

# Let's get started!
while True:
    agent_chain.run(input('Instruction: '))
