import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

search_tool = SerperDevTool()
llm = ChatOpenAI(model = "gpt-3.5-turbo")

def create_research_agent():
    return Agent(
        role = "Research Specialist",
        goal = "Conduct through research on given topics",
        backstory = "You are an experienced researcher with expertise in finding and synthesizing information from various sources",
        verbose = True,
        allow_delegation = False,
        llm = llm,
        tools = [search_tool]
    )

def create_research_task(agent,topic):
    return Task(
        description = f"Research the following topic and provide a comprehensive summary: {topic}",
        agent = agent,
        expected_output = "A detailed summary of the research findings, including key points and insights releated to the topic"
    )

def run_research(topic):
    agent = create_research_agent()
    task = create_research_task(agent,topic)
    crew = Crew(agent = [agent], tasks = [task])
    results = crew.kickoff()
    return results

if __name__ == "__main__":
    print("Welcome to the Research Agents")
    topic = input("Enter the Research topic ")
    result = run_research(topic)
    print("Research Results: ")
    print(result)