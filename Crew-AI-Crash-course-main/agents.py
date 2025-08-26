from crewai import Agent
from tools import yt_tool
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

# LLM using Groq
llm = LLM(
    model="groq/llama-3.2-90b-text-preview",
    temperature=0.7
)

# Blog Researcher Agent
blog_researcher = Agent(
    role='Blog Researcher from Youtube Videos',
    goal='Get the relevant video transcription for the topic {topic} from the provided YT channel',
    verbose=True,
    memory=True,
    backstory="Expert in understanding videos in AI, Data Science, Machine Learning, and GenAI, providing suggestions",
    llm=llm,
    tools=[yt_tool],
    allow_delegation=True
)

# Blog Writer Agent
blog_writer = Agent(
    role='Blog Writer',
    goal='Narrate compelling tech stories about the video {topic} from YT video',
    verbose=True,
    memory=True,
    backstory="With a flair for simplifying complex topics, you craft engaging narratives that captivate and educate.",
    llm=llm,
    tools=[yt_tool],
    allow_delegation=False
)
