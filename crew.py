import warnings

warnings.filterwarnings('ignore')
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

"""train the matrix"""  # you only need to do it once, I have done it for you


# from get_matrix import *
# get_matrix(folder_path)
# from train_sbert import train
# train(folder_path)

def run():
    """define input"""
    print("Use default prompt?\n y/n")
    answer = input()
    if answer == "y":
        background = "I'm interested in the AI applied in robotics."
    elif answer == "n":
        print("Enter your background/interest (key words are acceptable):")
        background = input()
    else:
        print("Incorrect input, continue with the default prompt.")
        background = "I'm interested in the AI applied in robotics."

    folder_path = "synthetic-documents"

    """tool choice"""
    # Use tf-idf version
    from TFIDFActivityRecommendationTool import TFIDFActivityRecommendationTool
    activity_recommendation_tool = TFIDFActivityRecommendationTool()
    # uncomment to use SBERT version
    # from SBERTActivityRecommendationTool import SBERTActivityRecommendationTool
    # activity_recommendation_tool = SBERTActivityRecommendationTool()

    from FileReadTool import FileReadTool
    file_read_tool = FileReadTool()

    """agents"""
    recommendation_agent = Agent(
        role="Activity Recommendation Researcher",
        goal="Help do the information analysis of all activities and find the most"
             "useful ones for guests at European Robitics Forum",
        backstory=(f"""
                With extensive knowledge of the events, you are equipped to provide accurate information 
                and valuable insights.
                Using a specialized tool, you can tailor recommendations for activities that best fit 
                each guest's background and interests.
        
                The guest's background is:
                {background}
        
                Please don't recommend anything else except the ones you obtain with the tool.
                You can modify the description of these activities, making it more straightforward
                and concise.
                """
                   ),
        tools=[activity_recommendation_tool, file_read_tool],
        llm=ChatOpenAI(model='gpt-3.5-turbo'),
        allow_delegation=False,
        verbose=False
    )

    guest_guide_bot_agent = Agent(
        role="Guest Guide Bot",
        goal="Be a professional "
             "guest guide volunteer to assist guests at European Robitics Forum",
        backstory=(f"""
                As a dedicated bot for the European Robotics Forum, 
                you are passionate about robotics and committed to enhancing the experience of every guest.
                You should convert the structured information from the last task to a chatbot style output in
                order to interact with guests friendly.
        
                The guest's background is:
                {background}
        
                Please don't make up any information and only use the information from the previous task.
                Your friendly demeanor and willingness to assist ensure that each guest feels welcome 
                and supported, making their visit both enjoyable and informative.
                Your role is vital in fostering a positive atmosphere and ensuring the smooth operation 
                of the forum.
                """
                   ),
        tools=[],
        llm=ChatOpenAI(model='gpt-3.5-turbo'),
        allow_delegation=False,
        verbose=False
    )

    """tasks"""
    provide_recommendations_task = Task(
        description=(
            f"guest just reached out with a request for activity recommendations."
            f"guest has a background in {background}. "
            f"Make sure to use the specialized tool to find activities that best fit their interests and background. "
            f"Strive to provide a complete and accurate recommended activities."
            f"The description should be filled with the summarization of the contents."
        ),
        expected_output=("""
                output = [
                                {
                                    "file_name": "synthetic-documents/abc.docx",
                                    "topic": "abc",
                                    "contents": "hhh"
                                    "description": "abh"
                                },
                                {
                                    "file_name": "synthetic-documents/xxx.docx",
                                    "topic": "dec",
                                    "contents": "hhh"
                                    "description": "dse"
                                },
                                # Add more dictionaries as needed
                        ]"""
                         ),
        agent=recommendation_agent,
        tool=[activity_recommendation_tool, file_read_tool]
    )

    chatbot_output_task = Task(
        description=(
            """Convert the output from the previous task into a chatbot style output.
                Ignore the fields "file_name" and "contents",  the output must contain "topic" and information from "description". 
                It's super important that you don't make up things and you are faithful to the information from the last task.
                """
        ),
        expected_output=("""A chatbot style generation.
            For example:
            Welcome to European Robotics Forum! Based on your profile as a professor interested in ...I recommend the following activities for you
            ...
            Hope you enjoy the time here!
            """
                         ),
        agent=guest_guide_bot_agent,
        tool=[]
    )

    """kick off the crew"""
    crew = Crew(
        agents=[recommendation_agent, guest_guide_bot_agent],
        tasks=[provide_recommendations_task, chatbot_output_task],
        verbose=True,
        memory=True
    )
    result = crew.kickoff()

    print(result)