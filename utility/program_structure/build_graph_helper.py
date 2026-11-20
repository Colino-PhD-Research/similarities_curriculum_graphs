import random

"""
This file has common mapings, variables, and functions that are used in both 
build_graph_from_db and guild_graph_from_json
"""

# seed random generator so we get same results each time
# (this affects background color)
random.seed(42)

# Fill colors for requirement groups (picklists). Distinct warm/neutral hues so they
# stand out clearly from the course-type fills (blue #4477AA, teal #009988).
bg_colors = [
    "#E8A838", "#D46B2D", "#B85C8A", "#7B5EA7", "#5B8DD9",
    "#3A9E6E", "#C4773B", "#8C6BAE", "#4A90C4", "#D4734A",
    "#6AAF8B", "#C45A7A", "#7A9EC4", "#B87A3A", "#5A8FAA",
    "#D4A044", "#8B5E9A", "#4A7DC4", "#C46A3A", "#6A9E5A",
    "#B84A6A", "#5A7FAA", "#D48A34", "#7A6EAE", "#4A9E8A",
]

# key is the course type encoded in the json file
# value is a human readable, nice version, of each course type
course_type_map = {
    "cs_course_core": "Core CS Course",
    "ai_program_required_courses": "AI Program Required Course",
    "ai_program_unrestricted_electives": "AI Program Unrestricted Elective",
    "unrestricted_elective": "Unrestricted Elective"
    }

# key is the course type encoded in the json file
# value is the color to be used for nodes of that type
course_color_map = {
    "cs_course_core": "blue",
    "ai_program_required_courses": "orange",
    "ai_program_unrestricted_electives": "red",
    "unrestricted_elective": "green"
    }

# key is the university and program related attributes as encoded in json file
# value is nice, human readable string for each attribute
# there is also a key not in here for semester_schedule
graph_attribute_map = {"university": "University", 
                       "location": "Location", 
                       "school": "School", 
                       "department": "Department", 
                       "degree": "Degree", 
                       "ai_program": "AI Program", 
                       "curriculum_link": "Degree Requirements",
                       "max_classes_per_term": "max_classes_per_term",
                       "program_name": "Program Name"}

def make_readable_course_type(course_type):
    return course_type_map[course_type]

# return flattened list of prequisites with no double requirement satisfying courses
# we always take the first option of a picklist to satisfy requirement
# ex if prereqs are [["A"], ["B", "C"]]
# then we return ["A", "B"]
def filter_requisites(requisites):
    flattened_requisites = []
    
    for grouping in requisites:
        flattened_requisites.append(grouping[0])

    return flattened_requisites

