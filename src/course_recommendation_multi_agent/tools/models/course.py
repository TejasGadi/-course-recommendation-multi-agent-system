from typing import List, Optional, Dict
from pydantic import BaseModel
from datetime import datetime

class CourseSchedule(BaseModel):
    daily_hours: float
    timing: str  # morning, afternoon, evening, flexible
    days_per_week: int

class CourseMetadata(BaseModel):
    provider: str
    duration_months: int
    schedule: CourseSchedule
    start_date: Optional[datetime] = None
    mode: str  # online, offline, hybrid
    language: str
    prerequisites: List[str] = []
    suitable_for: List[str]  # educational levels
    cost: Optional[float] = None
    certification: bool = False
    certification_type: Optional[str] = None

class Course(BaseModel):
    id: str
    title: str
    description: str
    metadata: CourseMetadata
    career_outcomes: List[str] = []
    url: str
    source_credibility: float = 1.0
    last_validated: datetime
    validation_score: float = 1.0
    student_feedback: List[Dict] = []

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "metadata": self.metadata.dict(),
            "career_outcomes": self.career_outcomes,
            "url": self.url,
            "source_credibility": self.source_credibility,
            "last_validated": self.last_validated.isoformat(),
            "validation_score": self.validation_score,
            "student_feedback": self.student_feedback
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Course":
        if isinstance(data.get("last_validated"), str):
            data["last_validated"] = datetime.fromisoformat(data["last_validated"])
        if isinstance(data.get("metadata"), dict):
            data["metadata"] = CourseMetadata(**data["metadata"])
        return cls(**data) 