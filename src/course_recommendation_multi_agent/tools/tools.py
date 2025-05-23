from crewai.tools import BaseTool
from pydantic import Field, BaseModel, validator
from typing import Optional, Dict, List, Literal, Union, Any, ClassVar, Type
import json
import os
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from .models.course import Course
from datetime import datetime
import chromadb
from chromadb.config import Settings
from crewai import LLM

# Field-specific Pydantic models for validation
class NameField(BaseModel):
    name: str

class AgeField(BaseModel):
    age: int

class EducationalLevelField(BaseModel):
    educational_level: str

class CourseInterestsField(BaseModel):
    course_interests: List[str]

class CourseModeField(BaseModel):
    course_mode: Literal["online", "in-person", "hybrid"]

class DurationField(BaseModel):
    max_duration_months: int

class DailyHoursField(BaseModel):
    daily_hours: int

class PreferredTimingField(BaseModel):
    preferred_timing: Literal["morning", "afternoon", "evening", "flexible"]

class DaysPerWeekField(BaseModel):
    days_per_week: int

class MaxCostField(BaseModel):
    max_cost: float

class LanguageField(BaseModel):
    language: str

class CertificationField(BaseModel):
    certification_needed: bool

class LocationField(BaseModel):
    location_preference: str

class CareerGoalsField(BaseModel):
    career_goals: List[str]

class PreviousCoursesField(BaseModel):
    previous_courses: List[str]

class SkillsField(BaseModel):
    skills: List[str]

class ProfileCompletion(BaseModel):
    completion_percentage: float
    next_field: Optional[str]
    is_complete: bool

class AskData(BaseModel):
    """Data for the 'ask' action."""
    question: str = Field(..., description="The question to ask the student.")

class CreateData(BaseModel):
    """Data for the 'create' action."""
    pass  # No additional data needed for create action

class UpdateData(BaseModel):
    """Data for the 'update' action."""
    name: str = Field(..., description="Name of the student whose profile to update.")

class GetData(BaseModel):
    """Data for the 'get' action."""
    name: str = Field(..., description="Name of the student whose profile to retrieve.")

# Vector DB functionality class
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

#Tool Input schema
class CourseWebSearchInput(BaseModel):
    """Input schema for the Course Web Search tool."""
    query: str = Field(
        ...,
        description="The search query related to courses. Can include course titles, descriptions, providers, and requirements."
    )

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Search query cannot be empty")
        if len(v) < 3:
            raise ValueError("Search query must be at least 3 characters long")
        return v

    class Config:
        arbitrary_types_allowed = True

#Tool Input schema
class CareerInsightInput(BaseModel):
    """Input schema for the Career Insight tool."""
    query: str = Field(
        ...,
        description="The career field or job title to get insights about. Used to fetch career insights, job market data, and educational pathways."
    )

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Career query cannot be empty")
        if len(v) < 3:
            raise ValueError("Career query must be at least 3 characters long")
        return v

    class Config:
        arbitrary_types_allowed = True
    
#Tool Input schema
class CoursesVectorDBInput(BaseModel):
    """Input schema for the Courses Vector Database tool."""
    action: Literal["search", "add", "update", "delete"] = Field(
        ..., 
        description="The action to perform on the courses database"
    )
    data: Dict[str, Any] = Field(
        ...,
        description="""
        Data for the action:
        - search: {"query": str, "n_results": Optional[int], "filters": Optional[Dict]}
        - add: Course data dictionary
        - update: Course data dictionary
        - delete: {"course_id": str}
        """
    )

    @validator('data')
    def validate_data(cls, v, values):
        action = values.get('action')
        if not action:
            raise ValueError("Action is required")
            
        if action == "search":
            if "query" not in v:
                raise ValueError("Search requires 'query' field")
        elif action in ["add", "update"]:
            required_fields = ["id", "title", "description", "provider"]
            missing = [f for f in required_fields if f not in v]
            if missing:
                raise ValueError(f"Missing required course fields: {', '.join(missing)}")
        elif action == "delete":
            if "course_id" not in v:
                raise ValueError("Delete requires 'course_id' field")
        return v

    class Config:
        arbitrary_types_allowed = True




#Tool
class CourseWebSearchTool(BaseTool):
    """Tool for searching courses using web search."""
    name: str = "Course Web Search"
    description: str = """
    Use this tool to search for courses online.
    Provide a search query related to courses, and it will return relevant results.
    The search can include course titles, descriptions, providers, and requirements.
    """
    tavily_search: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)
    args_schema: Type[BaseModel] = CourseWebSearchInput

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

#Tool
class CareerInsightTool(BaseTool):
    """Tool for gathering career insights and job market data."""
    name: str = "Career Insight"
    description: str = """
    Use this tool to get career insights, job market data, and educational pathways.
    Provide a career field or job title to get relevant information.
    """
    tavily_search: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)
    args_schema: Type[BaseModel] = CareerInsightInput

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



class StudentProfileToolInput(BaseModel):
    """Input schema for the Student Profile Manager tool."""
    action: Literal["create", "update", "get", "ask"] = Field(
        ..., 
        description="The action to perform on the student profile."
    )
    name: Optional[str] = Field(
        default=None,
        description="Name of the student for profile operations. Required for 'update' and 'get' actions."
    )
    question: Optional[str] = Field(
        default=None,
        description="Question to ask the student. Required for 'ask' action."
    )

    @validator('name')
    def validate_name(cls, v, values):
        action = values.get('action')
        if action in ['update', 'get'] and not v:
            raise ValueError(f"'{action}' action requires 'name' field")
        return v

    @validator('question')
    def validate_question(cls, v, values):
        action = values.get('action')
        if action == 'ask' and not v:
            raise ValueError("'ask' action requires 'question' field")
        return v

    class Config:
        arbitrary_types_allowed = True

#Tool
class StudentProfileTool(BaseTool):
    """Tool for managing student profiles and preferences."""
    name: str = "Student Profile Collector and Manager"
    description: str = """
    Use this tool to collect and manage student profile information as interact with the student.
    Actions and required fields:
    - create: Create new student profile
        Required fields: none
    - update: Update existing profile
        Required fields: name
    - get: Retrieve profile information
        Required fields: name
    - ask: Ask a question to the student
        Required fields: question
    """
    profiles: Dict = Field(default_factory=dict)
    llm: LLM = Field(default_factory=lambda: LLM(model=os.environ["MODEL"]))
    args_schema: type[BaseModel] = StudentProfileToolInput

    FIELD_MODELS: ClassVar[Dict[str, Type[BaseModel]]] = {
        "name": NameField,
        "age": AgeField,
        "educational_level": EducationalLevelField,
        "course_interests": CourseInterestsField,
        "course_mode": CourseModeField,
        "max_duration_months": DurationField,
        "availability.daily_hours": DailyHoursField,
        "availability.preferred_timing": PreferredTimingField,
        "availability.days_per_week": DaysPerWeekField,
        "constraints.max_cost": MaxCostField,
        "constraints.language": LanguageField,
        "constraints.certification_needed": CertificationField,
        "constraints.location_preference": LocationField,
        "career_goals": CareerGoalsField,
        "previous_courses": PreviousCoursesField,
        "skills": SkillsField
    }

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

    def _calculate_completion(self, profile_data: Dict) -> ProfileCompletion:
        """Calculate profile completion percentage and next empty field."""
        total_fields = len(self.FIELD_MODELS)
        filled_fields = 0
        next_empty = None

        for field in self.FIELD_MODELS.keys():
            if "." in field:
                parent, child = field.split(".")
                if (parent in profile_data and 
                    isinstance(profile_data[parent], dict) and 
                    child in profile_data[parent] and 
                    profile_data[parent][child]):
                    filled_fields += 1
                elif not next_empty:
                    next_empty = field
            elif field in profile_data and profile_data[field]:
                filled_fields += 1
            elif not next_empty:
                next_empty = field

        completion_percentage = (filled_fields / total_fields) * 100

        return ProfileCompletion(
            completion_percentage=round(completion_percentage, 2),
            next_field=next_empty,
            is_complete=completion_percentage == 100
        )

    def _validate_field_response(self, field: str, response: str) -> any:
        """Validate and parse response using field-specific Pydantic model."""
        try:
            model = self.FIELD_MODELS[field]
            
            # For fields that expect lists, pre-process the response
            if field in ["course_interests", "career_goals", "previous_courses", "skills"]:
                llm = LLM(model=os.environ["MODEL"], response_format=model)
                parsed = llm.call(f"Extract a list from this text: {response}")
                return getattr(parsed, field.split(".")[-1])
            
            # For boolean fields
            if field == "constraints.certification_needed":
                return model(certification_needed=response.lower() in ["yes", "true", "1", "y"]).certification_needed
            
            # For other fields
            field_name = field.split(".")[-1]
            return getattr(model(**{field_name: response}), field_name)
            
        except Exception as e:
            raise ValueError(f"Invalid response for {field}: {str(e)}")

    def _update_profile(self, name: str, field: str, value: any) -> ProfileCompletion:
        """Update profile with validated field value and return completion status."""
        if "." in field:
            parent, child = field.split(".")
            if parent not in self.profiles[name]:
                self.profiles[name][parent] = {}
            self.profiles[name][parent][child] = value
        else:
            self.profiles[name][field] = value

        self.profiles[name]["last_updated"] = datetime.now().isoformat()
        return self._calculate_completion(self.profiles[name])

    def _run(self, action: str, name: Optional[str] = None, question: Optional[str] = None) -> str:
        try:
            if action == "ask":
                if not question:
                    return "Error: Question required for asking"
                print("\n" + "-"*80)
                print("👤 Question for you:")
                print(question)
                print("-"*80)
                response = input("Your answer: ").strip()
                return response

            elif action == "create":
                question = self._generate_contextual_question("name", {})
                name = self._run("ask", question=question)
                if not name:
                    return "Error: Name is required"
                
                self.profiles[name] = {
                    "name": name,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
                completion = self._calculate_completion(self.profiles[name])
                return json.dumps({
                    "name": name,
                    "completion": completion.dict()
                })

            elif action == "update":
                if not name:
                    return "Error: Name required for update"
                if name not in self.profiles:
                    return f"Error: Profile for {name} not found"
                
                completion = self._calculate_completion(self.profiles[name])
                if completion.is_complete:
                    return json.dumps({
                        "message": "Profile is complete",
                        "completion": completion.dict()
                    })

                field = completion.next_field
                question = self._generate_contextual_question(field, self.profiles[name])
                response = self._run("ask", question=question)
                
                try:
                    validated_value = self._validate_field_response(field, response)
                    new_completion = self._update_profile(name, field, validated_value)
                    
                    return json.dumps({
                        "message": f"Updated {field}",
                        "completion": new_completion.dict()
                    })
                except ValueError as e:
                    return json.dumps({
                        "error": str(e),
                        "completion": completion.dict()
                    })

            elif action == "get":
                if not name:
                    return "Error: Name required"
                if name not in self.profiles:
                    return f"Error: Profile for {name} not found"
                
                completion = self._calculate_completion(self.profiles[name])
                return json.dumps({
                    "profile": self.profiles[name],
                    "completion": completion.dict()
                })

            else:
                return "Invalid action specified"
        except Exception as e:
            return f"Error in profile management: {str(e)}"

class CoursesVectorDB(BaseTool):
    name: str = "Courses Database"
    description: str = "Manages course storage and retrieval using vector database, perform vector database operations actions: search, add, update, delete , data is courses database"
    db: CourseVectorDB = Field(default_factory=lambda: CourseVectorDB())
    args_schema: Type[BaseModel] = CoursesVectorDBInput

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