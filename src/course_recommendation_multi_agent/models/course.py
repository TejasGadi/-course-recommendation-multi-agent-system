from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class Provider(BaseModel):
    name: str
    institution: Optional[str] = None
    platform: str

class Duration(BaseModel):
    total_weeks: int
    daily_hours: float
    total_hours: int

class Schedule(BaseModel):
    start_date: str
    end_date: str
    time_slots: List[str]
    flexible: bool

class Mode(BaseModel):
    type: str
    features: List[str]

class Language(BaseModel):
    primary: str
    subtitles: List[str]

class Prerequisites(BaseModel):
    skills: List[str]
    courses: List[str]
    academic_level: str
    technical_requirements: List[str]

class Level(BaseModel):
    academic: str
    difficulty: str
    target_audience: List[str]

class Cost(BaseModel):
    amount: float
    currency: str
    payment_options: List[str]
    financial_aid: bool

class Certification(BaseModel):
    provided: bool
    type: str
    accredited: bool
    validity: str

class CareerOutcomes(BaseModel):
    roles: List[str]
    skills_gained: List[str]
    job_guarantee: bool
    career_support: bool

class Content(BaseModel):
    topics: List[str]
    projects: List[str]
    assessments: List[str]

class Metadata(BaseModel):
    rating: float
    reviews_count: int
    last_updated: str
    url: str
    tags: List[str]

class Course(BaseModel):
    """Comprehensive course model with all details."""
    id: str
    title: str
    description: str
    provider: Provider
    duration: Duration
    schedule: Schedule
    mode: Mode
    language: Language
    prerequisites: Prerequisites
    level: Level
    cost: Cost
    certification: Certification
    career_outcomes: CareerOutcomes
    content: Content
    metadata: Metadata

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Course':
        """Create a Course instance from a dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Course instance to dictionary."""
        return self.model_dump()

    def get_search_text(self) -> str:
        """Get text representation for vector search."""
        return f"""
        {self.title}
        {self.description}
        Provider: {self.provider.name} ({self.provider.platform})
        Level: {self.level.difficulty} ({self.level.academic})
        Topics: {', '.join(self.content.topics)}
        Skills: {', '.join(self.career_outcomes.skills_gained)}
        Tags: {', '.join(self.metadata.tags)}
        """ 