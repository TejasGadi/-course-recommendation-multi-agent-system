from crewai.tools import BaseTool
from pydantic import Field
from typing import Optional, Dict, List
import json
import os
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from .models.course import Course
from .models.student import Student
from datetime import datetime
import chromadb
from chromadb.config import Settings

class CourseVectorDB:
    """Vector database for course storage and retrieval"""
    def __init__(self, persist_directory: str = "knowledge/vector_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        self.collection = self.client.get_or_create_collection(
            name="courses",
            metadata={"hnsw:space": "cosine"}
        )

    def add_course(self, course: Course) -> None:
        course_dict = course.to_dict()
        document = f"{course_dict.get('title', '')}\n{course_dict.get('description', '')}"
        self.collection.add(
            documents=[document],
            metadatas=[course_dict],
            ids=[course_dict.get('id')]
        )

    def search_courses(self, query: str, n_results: int = 10, filters: Optional[Dict] = None) -> List[Course]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filters
        )
        return [Course.from_dict(metadata) for metadata in results["metadatas"][0]]

    def update_course(self, course: Course) -> None:
        course_dict = course.to_dict()
        self.collection.update(
            ids=[course_dict.get('id')],
            metadatas=[course_dict],
            documents=[f"{course_dict.get('title', '')}\n{course_dict.get('description', '')}"]
        )

    def delete_course(self, course_id: str) -> bool:
        success = self.collection.delete(
            ids=[course_id]
        )
        return success

class CourseSearchTool(BaseTool):
    """Tool for searching courses using web search."""
    name: str = "Course Search"
    description: str = """
    Use this tool to search for courses online.
    Provide a search query related to courses, and it will return relevant results.
    The search can include course titles, descriptions, providers, and requirements.
    """
    tavily_search: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)

    def _run(self, query: str) -> str:
        try:
            # Enhance query for course-specific search
            enhanced_query = f"educational course {query}"
            results = self.tavily_search.results(
                enhanced_query,
                max_results=5,
                search_depth="advanced"
            )
            
            # Format results for better readability
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "description": result.get("snippet", ""),
                    "url": result.get("url", ""),
                    "source": result.get("source", "")
                })
            
            return json.dumps(formatted_results, indent=2)
        except Exception as e:
            return f"Error in course search: {str(e)}"

class CareerInsightTool(BaseTool):
    """Tool for gathering career insights and job market data."""
    name: str = "Career Insight"
    description: str = """
    Use this tool to get career insights, job market data, and educational pathways.
    Provide a career field or job title to get relevant information.
    """
    tavily_search: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)

    def _run(self, query: str) -> str:
        try:
            # Enhance query for career-specific search
            enhanced_query = f"career path job market {query} requirements skills salary"
            results = self.tavily_search.results(
                enhanced_query,
                max_results=3,
                search_depth="advanced"
            )
            
            # Format career insights
            career_insights = {
                "career_path": query,
                "insights": [],
                "skills_required": [],
                "education_requirements": [],
                "job_market_outlook": "",
                "sources": []
            }
            
            for result in results:
                career_insights["insights"].append({
                    "title": result.get("title", ""),
                    "summary": result.get("snippet", "")
                })
                career_insights["sources"].append(result.get("url", ""))
            
            return json.dumps(career_insights, indent=2)
        except Exception as e:
            return f"Error in career insight search: {str(e)}"

class StudentProfileTool(BaseTool):
    """Tool for managing student profiles and preferences."""
    name: str = "Student Profile Manager"
    description: str = """
    Use this tool to manage student profile information.
    Actions:
    - create: Create new student profile
    - update: Update existing profile
    - get: Retrieve profile information
    """
    profiles: Dict = Field(default_factory=dict)

    def _run(self, action: str, data: Optional[Dict] = None) -> str:
        try:
            if action == "create":
                if not data:
                    return "Error: Profile data required for creation"
                profile_id = data.get("id", str(datetime.now().timestamp()))
                self.profiles[profile_id] = {
                    **data,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
                return f"Profile created successfully with ID: {profile_id}"
            
            elif action == "update":
                if not data or "id" not in data:
                    return "Error: Profile ID required for update"
                profile_id = data["id"]
                if profile_id not in self.profiles:
                    return f"Error: Profile {profile_id} not found"
                self.profiles[profile_id].update(data)
                self.profiles[profile_id]["last_updated"] = datetime.now().isoformat()
                return f"Profile {profile_id} updated successfully"
            
            elif action == "get":
                if not data or "id" not in data:
                    return "Error: Profile ID required"
                profile_id = data["id"]
                if profile_id not in self.profiles:
                    return f"Error: Profile {profile_id} not found"
                return json.dumps(self.profiles[profile_id], indent=2)
            
            else:
                return "Invalid action specified"
        except Exception as e:
            return f"Error in profile management: {str(e)}"

class VectorDBTool(BaseTool):
    name: str = "Vector Database"
    description: str = "Manages course storage and retrieval using vector database"
    db: CourseVectorDB = Field(default_factory=lambda: CourseVectorDB())

    def _run(self, action: str, data: Dict) -> str:
        """
        Perform vector database operations
        action: search, add, update, delete
        data: course data or search parameters
        """
        try:
            if action == "search":
                courses = self.db.search_courses(
                    query=data.get("query", ""),
                    n_results=data.get("n_results", 10),
                    filters=data.get("filters")
                )
                return json.dumps([course.to_dict() for course in courses])
            elif action == "add":
                course = Course.from_dict(data)
                self.db.add_course(course)
                return "Course added successfully"
            elif action == "update":
                course = Course.from_dict(data)
                self.db.update_course(course)
                return "Course updated successfully"
            elif action == "delete":
                success = self.db.delete_course(data["course_id"])
                return "Course deleted successfully" if success else "Course not found"
            else:
                return "Invalid action"
        except Exception as e:
            return f"Error in vector database tool: {str(e)}"

class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Searches the web for course information and career insights"
    tavily_search: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)

    def _run(self, query: str) -> str:
        """
        Search the web for information
        query: search query string
        """
        try:
            results = self.tavily_search.results(
                query,
                max_results=5,
                search_depth="advanced"
            )
            return json.dumps(results)
        except Exception as e:
            return f"Error in web search tool: {str(e)}"

class Course:
    """Temporary Course class until proper model is implemented"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

    def to_dict(self) -> Dict:
        return self.__dict__

class Student:
    """Temporary Student class until proper model is implemented"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

    def to_dict(self) -> Dict:
        return self.__dict__ 