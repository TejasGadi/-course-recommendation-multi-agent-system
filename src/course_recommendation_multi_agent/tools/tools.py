from crewai.tools import BaseTool
from pydantic import Field, BaseModel
from typing import Optional, Dict, List, Literal
import json
import os
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from .models.course import Course
from .models.student import StudentProfile, Availability, Constraints
from datetime import datetime
import chromadb
from chromadb.config import Settings
from crewai import LLM

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


class AskData(BaseModel):
    """Data for the 'ask' action."""
    question: str = Field(..., description="The question to ask the student.")

class StudentProfileToolInput(BaseModel):
    """Input schema for the Student Profile Manager tool."""
    action: Literal["create", "update", "get", "ask", "next_question"] = Field(..., description="The action to perform ('create', 'update', 'get', 'ask', 'next_question').")
    data: Optional[Dict] = Field(None, description="A dictionary containing data relevant to the action.")

class StudentProfileTool(BaseTool):
    """Tool for managing student profiles and preferences."""
    name: str = "Student Profile Collector and Manager"
    description: str = """
    Use this tool to collect and manage student profile information as interact with the student.
    Actions:
    - create: Create new student profile
    - update: Update existing profile
    - get: Retrieve profile information
    - ask: Ask a question to the student and get their response
    - next_question: Get next question based on current profile state
    """
    profiles: Dict = Field(default_factory=dict)
    llm: LLM = Field(default_factory=lambda: LLM(model=os.environ["MODEL"]))
    args_schema: type[BaseModel] = StudentProfileToolInput

    def _get_next_empty_field(self, profile_data: Dict) -> Optional[str]:
        """Identify next empty required field in the profile."""
        # Define the order of fields and their base questions
        required_fields = [
            ("name", "name"),
            ("educational_level", "education"),
            ("age", "age"),
            ("interests", "interests"),
            ("course_mode", "learning mode"),
            ("max_duration_months", "time commitment"),
            ("availability.daily_hours", "daily hours"),
            ("availability.preferred_timing", "preferred timing"),
            ("availability.days_per_week", "weekly commitment"),
            ("constraints.max_cost", "budget"),
            ("constraints.language", "languages"),
            ("constraints.certification_needed", "certification"),
            ("constraints.location_preference", "location"),
            ("career_goals", "career goals"),
            ("previous_courses", "previous courses"),
            ("skills", "current skills")
        ]

        # Check nested fields in availability and constraints
        if "availability" not in profile_data or not isinstance(profile_data["availability"], dict):
            return "availability.daily_hours"
        if "constraints" not in profile_data or not isinstance(profile_data["constraints"], dict):
            return "constraints.language"

        # Find the next empty field
        for field, field_type in required_fields:
            if "." in field:
                parent, child = field.split(".")
                if parent in profile_data and isinstance(profile_data[parent], dict):
                    if child not in profile_data[parent] or not profile_data[parent][child]:
                        return field
            elif field not in profile_data or not profile_data[field]:
                return field
        
        return None

    def _generate_contextual_question(self, field: str, profile_data: Dict) -> str:
        """Generate a contextual question based on the field and previous answers."""
        # Create a prompt for the LLM to generate a contextual question
        context = "Previous answers:\n"
        for key, value in profile_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    context += f"{key}.{sub_key}: {sub_value}\n"
            else:
                context += f"{key}: {value}\n"

        prompt = f"""Based on the following context of previous answers:
                    {context}

                    Generate a natural, conversational question to ask about the user's {field.replace('.', ' ')}. 
                    The question should be contextual and reference previous answers where relevant.
                    If this is the first question (about name), just ask "What is your name?"

                    Question:
                """

        # Use LLM to generate the question
        class QuestionResponse(BaseModel):
            question: str

        llm = LLM(model=os.environ["MODEL"], response_format=QuestionResponse)
        response = llm.call(prompt)
        return response.question

    def _parse_response(self, response: str, field: str) -> any:
        """Parse user response based on field type."""
        if field == "age":
            return int(response)
        elif field in ["interests", "career_goals", "previous_courses", "skills"]:
            # Use LLM to extract list from free text
            # list_schema = type("ListSchema", (BaseModel,), {"items": List[str]})
            class ListSchema(BaseModel):
                items: List[str]
            
            llm = LLM(model=os.environ["MODEL"], response_format=ListSchema)
            parsed = llm.call(f"Extract a list of items from this text: {response}")
            return parsed.items
        elif field == "constraints.certification_needed":
            return response.lower() in ["yes", "true", "1", "y"]
        elif field in ["availability.daily_hours", "availability.days_per_week"]:
            return int(response)
        return response

    def _run(self, action: str, data: Optional[Dict] = None) -> str:
        try:
            if action == "ask":
                if not data or "question" not in data:
                    return "Error: Question required for asking"
                # if not data or "description" not in data:
                #     data["question"] = data["description"]
                #     return "Error: Question required for asking"
                print("\n" + "-"*80)
                print("👤 Question for you:")
                print(data["question"])
                print("-"*80)
                response = input("Your answer: ").strip()
                return response

            elif action == "create":
                # Ask for name first
                question = self._generate_contextual_question("name", {})
                name = self._run("ask", {"question": question})
                if not name:
                    return "Error: Name is required"
                
                self.profiles[name] = {
                    # "name": "Anonymous",
                    "name": name,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
                return name

            elif action == "update":
                if not data or "name" not in data:
                    return "Error: Name required for update"
                name = data["name"]
                if name not in self.profiles:
                    return f"Error: Profile for {name} not found"
                
                field = self._get_next_empty_field(self.profiles[name])
                if not field:
                    return "Profile is complete"

                # Generate contextual question based on previous answers
                question = self._generate_contextual_question(field, self.profiles[name])

                # For nested fields
                if "." in field:
                    parent, child = field.split(".")
                    if parent not in self.profiles[name]:
                        self.profiles[name][parent] = {}
                    
                    response = self._run("ask", {"question": question})
                    parsed_value = self._parse_response(response, field)
                    self.profiles[name][parent][child] = parsed_value
                else:
                    response = self._run("ask", {"question": question})
                    parsed_value = self._parse_response(response, field)
                    self.profiles[name][field] = parsed_value

                self.profiles[name]["last_updated"] = datetime.now().isoformat()
                
                # Try to create a StudentProfile object to validate
                try:
                    profile_data = self.profiles[name]
                    if "availability" in profile_data:
                        profile_data["availability"] = Availability(**profile_data["availability"])
                    if "constraints" in profile_data:
                        profile_data["constraints"] = Constraints(**profile_data["constraints"])
                    StudentProfile(**profile_data)
                except Exception as e:
                    return f"Profile updated but validation failed: {str(e)}"
                
                return f"Profile for {name} updated successfully"

            elif action == "get":
                if not data or "name" not in data:
                    return "Error: Name required"
                name = data["name"]
                if name not in self.profiles:
                    return f"Error: Profile for {name} not found"
                return json.dumps(self.profiles[name], indent=2)

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