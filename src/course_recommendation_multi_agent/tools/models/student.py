from typing import List, Optional, Dict
from pydantic import BaseModel

class Availability(BaseModel):
    daily_hours: int
    preferred_timing: str  # morning, afternoon, evening
    days_per_week: int

class Constraints(BaseModel):
    max_cost: Optional[float] = None
    language: List[str] = ["English"]
    certification_needed: bool = False
    location_preference: str = "any"  # online, offline, hybrid, any

class Student(BaseModel):
    id: str
    educational_level: str
    age: int
    interests: List[str]
    course_mode: str  # online/offline/hybrid/any
    availability: Availability
    max_duration_months: int
    career_goals: Optional[List[str]] = None
    constraints: Constraints
    previous_courses: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "educational_level": self.educational_level,
            "age": self.age,
            "interests": self.interests,
            "course_mode": self.course_mode,
            "availability": self.availability.dict(),
            "max_duration_months": self.max_duration_months,
            "career_goals": self.career_goals,
            "constraints": self.constraints.dict(),
            "previous_courses": self.previous_courses,
            "skills": self.skills
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Student":
        if isinstance(data.get("availability"), dict):
            data["availability"] = Availability(**data["availability"])
        if isinstance(data.get("constraints"), dict):
            data["constraints"] = Constraints(**data["constraints"])
        return cls(**data) 