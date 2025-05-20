from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

# Import tools directly from tools.py
from .tools.tools import (
    CourseSearchTool,
    CareerInsightTool,
    StudentProfileTool,
    VectorDBTool
)

@CrewBase
class CourseRecommendationMultiAgent():
    """CourseRecommendationMultiAgent crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def orchestrator_agent(self) -> Agent:
        """Creates the orchestrator agent that manages the overall conversation flow."""
        return Agent(
            config=self.agents_config['orchestrator_agent'],
            verbose=True,
            allow_delegation=True
        )

    @agent
    def profile_agent(self) -> Agent:
        """Creates the profile agent that handles student information collection."""
        return Agent(
            config=self.agents_config['profile_agent'],
            verbose=True,
            allow_delegation=True,
            tools=[StudentProfileTool()]
        )
    
    @agent
    def discovery_agent(self) -> Agent:
        """Creates the discovery agent that searches for relevant courses."""
        return Agent(
            config=self.agents_config['discovery_agent'],
            verbose=True,
            allow_delegation=True,
            tools=[
                CourseSearchTool(),
                VectorDBTool()
            ]
        )
    
    @agent
    def recommendation_agent(self) -> Agent:
        """Creates the recommendation agent that filters and ranks courses."""
        return Agent(
            config=self.agents_config['recommendation_agent'],
            verbose=True,
            allow_delegation=True,
            tools=[VectorDBTool()]
        )

    @agent
    def career_agent(self) -> Agent:
        """Creates the career agent that provides career guidance."""
        return Agent(
            config=self.agents_config['career_agent'],
            verbose=True,
            allow_delegation=True,
            tools=[CareerInsightTool()]
        )

    @task
    def course_discovery_task(self) -> Task:
        """Task for discovering relevant courses."""
        return Task(
            config=self.tasks_config['course_discovery_task']
        )
    
    @task
    def profile_collection_task(self) -> Task:
        """Task for collecting student profile"""
        return Task(
            config=self.tasks_config['profile_collection_task'],
            human_input=True
        )

    @task
    def recommendation_task(self) -> Task:
        """Task for generating course recommendations."""
        return Task(
            config=self.tasks_config['recommendation_task']
        )

    @task
    def career_guidance_task(self) -> Task:
        """Task for providing career guidance."""
        return Task(
            config=self.tasks_config['career_guidance_task']
        )

    @task
    def orchestration_task(self) -> Task:
        """Task for managing the overall conversation flow."""
        return Task(
            config=self.tasks_config['orchestration_task']
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Course Recommendation System crew."""
        return Crew(
            agents=self.agents,
            tasks=[
                self.profile_collection_task(),
                self.course_discovery_task(),
                self.recommendation_task(),
                self.career_guidance_task(),
                self.orchestration_task()
            ],
            process=Process.sequential,
            verbose=True,
            memory=True
        )