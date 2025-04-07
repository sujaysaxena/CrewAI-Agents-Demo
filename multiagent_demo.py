import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai import Process
from crewai_tools import SerperDevTool

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

search_tool = SerperDevTool()
llm = ChatOpenAI(model = "gpt-3.5-turbo")

# Creating a senior researcher agent with memory and verbose mode
def create_research_agent(topic):
    return Agent(
    role='Senior Researcher',
    goal=f'Uncover groundbreaking technologies in {topic}',
    verbose=True,
    memory=True,
    backstory=(
       """ Driven by curiosity, you're at the forefront of
        innovation, eager to explore and share knowledge that could change
        the world."""
    ),
    tools=[search_tool],
    allow_delegation=True,
    llm=llm
)


# Creating a writer agent with custom tools and delegation capability
def create_writer_agent(topic):
    return Agent(
    role='Writer',
    goal=f'Narrate compelling tech stories about {topic}',
    verbose=True,
    memory=True,
    backstory=(
        """With a flair for simplifying complex topics, you craft
        engaging narratives that captivate and educate, bringing new
        discoveries to light in an accessible manner."""
    ),
    tools=[search_tool],
    allow_delegation=False,
    llm=llm
)

def create_research_task(agent1,topic):
    return Task(
    description=(
            f"Identify the next big trend in {topic}. "
            "Focus on identifying pros and cons and the overall narrative. "
            "Your final report should clearly articulate the key points, "
            "its market opportunities, and potential risks."
        ),
        expected_output=f"A comprehensive 3-paragraph report on the latest {topic} trends.",
        agent=agent1
    )

def create_write_task(agent2,topic):
    return Task(
     description=(
            f"Compose an insightful article on {topic}. "
            "Focus on the latest trends and how it's impacting the industry. "
            "This article should be easy to understand, engaging, and positive."
        ),
        expected_output=f"A 4-paragraph article on {topic} advancements formatted as markdown.",
        agent=agent2,
        async_execution=False,
        output_file="new-blog-post.md"
    )

# Forming the tech-focused crew with enhanced configurations
def run_research(topic):
    agent1 = create_research_agent(topic)
    task1 = create_research_task(agent1,topic)
    agent2 = create_writer_agent(topic)
    task2 = create_write_task(agent2,topic)
    crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    process=Process.sequential
    )
    results = crew.kickoff()
    return results

if __name__ == "__main__":
    print("Welcome to the Research Agents")
    topic = input("Enter the Research topic ")
    result = run_research(topic)
    print("Research Results: ")
    print(result)
