#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from course_recommendation_multi_agent.tools.tools import CourseVectorDB
from course_recommendation_multi_agent.models.course import Course

def load_courses_from_jsonl(file_path: str) -> list:
    """Load courses from a JSONL file."""
    courses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                courses.append(json.loads(line))
    return courses

def main():
    # Initialize vector database
    db = CourseVectorDB()
    
    # Get the absolute path to the data directory
    # data_dir = Path(project_root) / 'data'
    courses_file = "../../data/courses.jsonl"
    
    print(f"Loading courses from {courses_file}...")
    
    try:
        # Load courses from JSONL file
        courses_data = load_courses_from_jsonl(courses_file)
        print(f"Found {len(courses_data)} courses")
        
        # Add each course to the vector database
        for course_data in courses_data:
            course = Course.from_dict(course_data)
            db.add_course(course)
            print(f"Added course: {course.title}")
        
        print("\nSuccessfully loaded all courses into the vector database!")
        
        # Verify by performing a test search
        print("\nPerforming test search...")
        test_results = db.search_courses("python programming", n_results=2)
        print("\nTest search results:")
        for course in test_results:
            print(f"- {course.title} ({course.provider['name']})")
            
    except Exception as e:
        print(f"Error loading courses: {str(e)}")
        raise

if __name__ == "__main__":
    main() 