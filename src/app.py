#!/usr/bin/env python
import sys
import warnings
from dotenv import load_dotenv
import os
import json

from course_recommendation_multi_agent.crew import CourseRecommendationMultiAgent

# Load environment variables
load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the Course Recommendation System crew.
    The system will interact with the user through conversation to gather preferences
    and provide course recommendations.
    """
    try:
        # Initialize and run the crew
        crew = CourseRecommendationMultiAgent().crew()
        
        # Start the conversation with an empty context
        # The orchestrator agent will handle the initial greeting and information gathering
        result = crew.kickoff()
        print(f"result: {result}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run()