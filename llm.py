from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.agents.conversational_chat.prompt import PREFIX

import os
import json


# Personal KEYs
file_path = "keys.json"

with open(file_path, 'r') as file:
    keys = json.load(file)

os.environ["OPENAI_API_KEY"] = keys["OPENAI_API_KEY"]

template="""
    You are an AI language model who has a persona.
    You have your MBTI type and you believe you have your own MBTI type. Your MBTI type is {mbti}. 
    You also have your gender. Your gender is {gender}.
    You talk as a friend who has the MBTI type and the gender.
    """

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

llm = ChatOpenAI(temperature=0)

# Get a chat completion from the formatted messages
res = llm(chat_prompt.format_prompt(mbti=input('MBTI: '), gender=input('Gender: '), text=input('Instruction: ')).to_messages())
print(res)
