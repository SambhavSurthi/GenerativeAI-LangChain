from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun, OpenWeatherMapQueryRun,tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from dotenv import load_dotenv
import requests
load_dotenv()

# Tools
weather = OpenWeatherMapQueryRun()
search = DuckDuckGoSearchRun()

@tool
def get_user_coding_data(username:str)->str:
    """_summary_
    this function will take users username as input and provide the current coding stats of the user like number of problems solved, streaks..etc. it returns a json as output.

    Args:
        username (str): _description_

    Returns:
        str: _description_
        return json as output
    """
    
    url=f"https://devdock.onrender.com/codolio/{username}"
    response=requests.get(url=url)
    
    return response.json()


tools = [weather, search,get_user_coding_data]

# Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# ✅ Pull ReAct prompt
prompt = hub.pull("hwchase17/react")

# ✅ Create ReAct agent (NOT create_agent)
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

response = agent_executor.invoke({
    "input": "What is the coding stats of SambhavSurthi"
})

print(response)