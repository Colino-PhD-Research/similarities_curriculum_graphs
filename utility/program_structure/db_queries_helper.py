from utility.universal_helpers import parse_course_code, get_db_connection, extract_courses_from_string
from copy import copy

"""
This file contains a series of functions that wrap database calls for getting some
of the most common data we may want about a univeristy or particular program/courses at
a university

Currently, these functions are primarily used to create the curriculum graph in
build_graph_from_db.py
"""

# This is a global variable that keeps track of any requirements we don't need to include
# because it is satisfied by another class
# It is used in the get course data function and get duplicated requirement courses function
duplicated_requirement_courses = set()

def get_duplicated_requirement_courses():
    return duplicated_requirement_courses

# get the id of a university
# name must exactly match either institution name or alternate name in db
# otherwise, None is returned if not found
def get_university_id(university_name):
    conn = get_db_connection()

    conn.execute("SELECT id FROM institutions \
                 WHERE institution_name = ? or alternate_name = ? \
                 ORDER BY created_at DESC LIMIT 1", \
                    [university_name, university_name])
    query_result = conn.fetchone()

    conn.close()

    return query_result[0] if query_result else None

# get the program id for a given university name and program name
# program name is defined in model as: 
#   program["program_name"] = ",".join([self.data_json["degree"], self.data_json["ai_program"] if self.data_json["ai_program"] else "none"])
def get_program_id(university_name, program_name):
    conn = get_db_connection()

    # program id is determined by institution id and program name
    conn.execute("SELECT id FROM programs WHERE institution_id = \
                    (SELECT id FROM institutions WHERE institution_name = ? OR alternate_name = ? ORDER BY created_at DESC LIMIT 1) \
                    AND program_name = ?", [university_name, university_name, program_name])
    query_result = conn.fetchone()
    conn.close()

    return query_result[0] if query_result else None

def get_number_scraped_prereqs(university, program):
    program_id = get_program_id(university, program)

    conn = get_db_connection()
    conn.execute( """
                SELECT COUNT(cr.id) AS num_relations
                FROM programs p
                LEFT JOIN program_to_courses ptc ON ptc.program_id = p.id
                LEFT JOIN course_relationships cr ON cr.primary_course_id = ptc.course_id
                WHERE p.id = ?
                 """, [program_id])
    query_result = conn.fetchone()
    conn.close()

    return int(query_result[0]) 

def has_scraped_prereqs(university, program):
    return get_number_scraped_prereqs(university, program) > 0

# returns True if at least one course linked to the program has both a real title
# (not just the "PREFIX NUMBER" placeholder used when no title was scraped) and a
# description; prints and returns None (distinct from False) if the university/program
# isn't found in eaai27.db, so callers can tell "not found" apart from "found but undocumented"
def has_title_and_descr(university, program):
    program_id = get_program_id(university, program)

    if program_id is None:
        print(f"{university} - {program} not found in database")
        return None

    conn = get_db_connection()
    conn.execute("""
                SELECT COUNT(c.id) AS num_valid
                FROM program_to_courses ptc
                JOIN courses c ON c.id = ptc.course_id
                WHERE ptc.program_id = ?
                  AND c.description IS NOT NULL AND c.description != ''
                  AND c.course_name IS NOT NULL AND c.course_name != ''
                  AND c.course_name != (c.common_prefix || ' ' || c.course_number)
                """, [program_id])
    query_result = conn.fetchone()
    conn.close()

    return int(query_result[0]) > 0

# get all course id's associated with a particular program
def get_list_of_program_course_ids(program_id):
    conn = get_db_connection()

    # make query
    # must be a course that has the correct program id
    conn.execute("SELECT DISTINCT(course_id) FROM program_to_courses WHERE program_id = ? ORDER BY course_id", [program_id])
                
    query_result = conn.fetchall()

    # turn list of tuples into a list of just ids
    course_ids = []
    for course in query_result:
        course_ids.append(course[0])

    # close connection
    conn.close()

    return course_ids

# count of distinct courses linked to a program -- cheaper than fetching every course
# row just to call len() on it, useful for programs with an unusually large course count
def get_program_course_count(program_id):
    conn = get_db_connection()

    conn.execute("SELECT COUNT(DISTINCT course_id) FROM program_to_courses WHERE program_id = ?", [program_id])
    query_result = conn.fetchone()
    conn.close()

    return query_result[0] if query_result else 0

# get (course_code, classification) for every course linked to a program in one query,
# instead of resolving course ids and then fetching each course's data separately.
# DISTINCT ON course_id guards against duplicate (program_id, course_id) rows in
# program_to_courses, same as the DISTINCT in get_list_of_program_course_ids above
def get_program_course_codes_and_types(program_id):
    conn = get_db_connection()

    conn.execute("""
                SELECT DISTINCT ON (ptc.course_id)
                       c.common_prefix || ' ' || c.course_number AS course_code, ptc.classification
                FROM program_to_courses ptc
                JOIN courses c ON c.id = ptc.course_id
                WHERE ptc.program_id = ?
                ORDER BY ptc.course_id
                """, [program_id])
    query_result = conn.fetchall()
    conn.close()

    return query_result

# like get_program_course_codes_and_types, but also returns each course's title, so
# callers can pre-filter (e.g. an honors-section check on the title) without building
# the full graph
def get_program_course_codes_types_and_titles(program_id):
    conn = get_db_connection()

    conn.execute("""
                SELECT DISTINCT ON (ptc.course_id)
                       c.common_prefix || ' ' || c.course_number AS course_code,
                       ptc.classification, c.course_name
                FROM program_to_courses ptc
                JOIN courses c ON c.id = ptc.course_id
                WHERE ptc.program_id = ?
                ORDER BY ptc.course_id
                """, [program_id])
    query_result = conn.fetchall()
    conn.close()

    return query_result

# do we at least have a course description or a unique course name that isn't
# just the default copy of the prefix and code?
def have_course_data(course_code, course_id):
    conn = get_db_connection()

    conn.execute("SELECT course_name, description\
                 FROM courses \
                 WHERE id = ? " \
                 "ORDER BY created_at DESC LIMIT 1", [course_id])
    query_result = conn.fetchone()

    if not query_result:
        conn.close()
        raise Exception(f"Something went wrong, can't find course_id {course_id}")
    
    course_name = query_result[0]
    description = query_result[1]

    conn.close()

    if description or course_name != course_code:
        return True
    
    return False

# given a grouping we want to pick one course out of the grouping to pick (this is used when
# there is a pick one out of long list of prereqs). Any course will do, but if we do have info
# on only one course in the group, we should choose that one to keep in graph
# returns a list of courses (will most likely have 1-2 elements)
def find_courses_to_keep(grouping, course_name_to_id_map):
    # If our grouping is not flattened (there's an and/or relationship)
    # pick the grouping that has more classes as that will be more typical
    # ex grouping = [[MAT 100, MAT 200], MAT 300] then we would keep MAT 100 and MAT 200
    for group in grouping:
        if isinstance(group, list):
            return group

    # find course here 
    # this part assumes flattened grouping (all ORs no ANDs)
    for course in grouping:
        if have_course_data(course, course_name_to_id_map[course]):
            return [course]

    # if no courses above have detailed info, pick first one
    return [grouping[0]]

# grouping ex: [MAT 100, MAT 101, MAT 102]
# grouping ex: [[MAT 100, MAT 200], MAT 300]
# grouping ex: [[MAT 100, MAT 200, "MAT 201 AND MAT 202"], MAT 300]
def is_course_name_in_grouping(course_name, grouping):
    for group in grouping:
        if isinstance(group, list):
            if course_name in group:
                return True
        else:
            #if group == course_name:
            # we can't check for strict equality because we might have an instance like
            # "MAT 201 AND MAT 202" in which case we are just searching for a substring
            if course_name in group:
                return True
    return False

# remove duplicate prereq requirements from list_course_names in place
# This will make sure if one OR another class can satisfy a preqreq, only one will stay in the list
def check_course_in_unofficial_picklist(course_name, course_name_to_id_map, unofficial_picklists, list_course_names, classes_to_remove):
    # see if this class is part of any grouping. If it is, add all classes but current to
    # classes to remove

    for grouping in unofficial_picklists:
        if is_course_name_in_grouping(course_name, grouping):
            # let's determine which course to keep-- which ever course has the most info filled out
            courses_to_keep = find_courses_to_keep(grouping, course_name_to_id_map)
            to_remove_unflattened = copy(grouping)

            # flatten to remove
            to_remove = []
            for sublist in to_remove_unflattened:
                if isinstance(sublist, list):
                    to_remove.extend(sublist)
                else:
                    to_remove.append(sublist)

            for course_to_keep in courses_to_keep:
                to_remove.remove(course_to_keep)

            # add all the classes to remove if they haven't been accounted for already
            for c in to_remove:
                if c not in classes_to_remove:
                    classes_to_remove.append(c)
            break

    # remove all courses in classes_to_remove from list_course_names
    for course in classes_to_remove:
        if course in list_course_names:
            list_course_names.remove(course)
"""
Most of the time we want to use program id. This makes sure we are getting picklists
for a particular program. Ex an official picklist like "CORE CS [4]" may vary across 
programs (DS, CS, etc).

However, when dealing with unofficial picklists (ones specific to prereqs) we don't want
to do it by program because it should be standard across the university. This allows us
to still get prereq info of courses that we scraped in one program and may be a prereq to
courses in another program but was not scraped for some reason.

Concrete ex, in complex_and_or_jsons, Southern Connecticut State University has 3 programs.
Their DS program has an entry for DSC 101. Their other two programs list DSC 101 as a possible
prereq but not a core course and it has no entry in the json for those two programs. However,
we still want to be able to access the prereq-related picklists for DSC 101 in the other programs
when building the graph and simplifying it.
"""
def get_all_picklists(conn, course_id, program_id, use_program_id=True):
    if use_program_id:
        conn.execute("SELECT DISTINCT p.picklist_name FROM courses_to_picklist pc \
        JOIN picklists p ON (pc.picklist_id = p.id) \
        WHERE course_id = ? AND program_id = ?", [course_id, program_id])
    else:
        conn.execute("SELECT DISTINCT p.picklist_name FROM courses_to_picklist pc \
        JOIN picklists p ON (pc.picklist_id = p.id) \
        WHERE course_id = ?", [course_id])

    query_result = conn.fetchall()

    return query_result

# official picklists are picklists explicitly mentioned in catalgoue/website. They have some name
# and the LLM should have given it a name format of "<pick list name> [<num classes>]" in JSON
def get_official_picklists(conn, course_id, program_id):
    query_result = get_all_picklists(conn, course_id, program_id)

    # extract picklist names from list of tuple results
    picklists = []
    unofficial_picklists = []
    for result in query_result:
        # if there is no "[" in the picklist name, it as a user created picklist for
        # pre/coreqs, not a true picklist we want to display to user
        if result[0].find("[") >= 0:
            picklists.append(result[0])
        else:
            unofficial_picklists.append(result[0])
    
    return picklists

def get_course_id(course, institution_id):
    pre, code = parse_course_code(course) 

    conn = get_db_connection()

    conn.execute("SELECT id FROM courses \
                 WHERE common_prefix = ? AND course_number = ? \
                 AND institution_id = ? \
                 ORDER BY created_at \
                 DESC LIMIT 1",
                 [pre, code, institution_id])
    result = conn.fetchone()

    conn.close()

    if result:
        return result[0]
    else:
        raise Exception(f"Course {course} not found for institution id {institution_id}")


# make a list of lists. Each nested list is a list of course names for an unofficial picklist grouping
# an unofficial picklist is a picklist we made to distinguish between multiple requirement paths for prereqs
# an official picklist is one documented by university in catalogue/website
# unofficial picklists are named in db as "course code / course code / course code/ etc"
# don't need program id because we want it across university (see comment for get_all_picklists)
def get_unofficial_picklists_groupings(conn, course_id):
    conn.execute("SELECT DISTINCT picklist_name FROM picklists WHERE id IN \
                 (SELECT picklist_id FROM requisite_picklist_to_course \
                  WHERE course_id = ?)", [course_id])
    query_result = conn.fetchall()
    
    picklists = []
    groupings = []
    
    if query_result:
        for result in query_result:
            picklists.append(result[0])
   
    for name in picklists:
        # group could be something like [MAT 100, MAT 101, MAT 102]
        # or something  like [MAT 100 and MAT 200, MAT 300]
        group = name.split("/")
        
        stripped_group = []
        for course in group:
            if course.upper().find("AND") == -1:
                # there's no and which means we have an individual course
                course = course.strip()
                pre, code = parse_course_code(course)
                # this is same as group but doesn't have extra spaces and stuff and course name
                # is standardized
                stripped_group.append(pre + " " + code)
            else:
                # we have one of those special and/or relationships
                # course_list = course.upper().split("AND")
                # intermediate_group = []

                # for c in course_list:
                #     course = c.strip()
                #     pre, code = parse_course_code(course)
                #     # intermediate group turns MAT 100 and MAT 200 grouping into
                #     # [MAT 100, MAT 200]
                #     intermediate_group.append(pre + " " + code)
            
                # stripped_group.append(intermediate_group)
                stripped_group.append(course.strip())

        # each list within groupings corresponds to one picklist
        groupings.append(stripped_group)
    
    # we do set conversions before putting pack as a list to make sure no classes repeated
    return groupings

# get all data we could care about for a particular course in a particular program
# return a dictionary with the data
# classifier is whatever classifier model has been trained in utility/models
def get_course_data(course_id, institution_id, program_id, allow_messy=False):
    course_data = {}

    conn = get_db_connection()

    # get course basic data
    conn.execute("SELECT common_prefix, course_number, course_name, course_url, course_hours, \
                  description, id FROM courses WHERE id = ? AND institution_id = ? \
                  ORDER BY created_at \
                 DESC LIMIT 1", [course_id, institution_id])
    course_basics = conn.fetchone()

    if not course_basics:
        raise Exception(f"Course not found! Course id: {course_id}, Institution id: {institution_id}")

    # fill in course dictionary object with data
    course_data["course_code"] = course_basics[0] + " " + course_basics[1]
    course_data["title"] = course_basics[2]
    course_data["course_url"] = course_basics[3]
    course_data["course_hours"] = course_basics[4]
    course_data["course_description"] = course_basics[5]

    conn.execute("SELECT classification FROM program_to_courses " \
            "WHERE program_id = ? AND course_id = ?", [program_id, course_basics[6]])
    classification_result = conn.fetchone()

    if classification_result:
        course_data["course_type"] = classification_result[0]
    else:
        # no classification means unscraped and was a prereq so put as core
        course_data["course_type"] = "cs_course_core"

    # get picklists
    picklists = get_official_picklists(conn, course_id, program_id)
    
    course_data["picklist"] = picklists

    if not allow_messy:
        course_data["prerequisites"], course_data["classes_removed_this_iteration"] = \
            requisites_helper_per_course(conn, course_id, institution_id, program_id, None, allow_messy)

        for course in course_data["classes_removed_this_iteration"]:
            duplicated_requirement_courses.add(course)
    else:
        course_data["prerequisites, solid"], course_data["prerequisites, dotted"] = \
            requisites_helper_per_course(conn, course_id, institution_id, program_id, None, allow_messy)

    # close database connection when done
    conn.close()

    return course_data

"""
Given an unofficial picklist grouping (list of lists where each sublist is an unofficial picklist)
and a course of interest, return what the unofficial picklist grouping is (first one found)
if it doesn't exist, return None
"""
def get_grouping_for_class(course, unofficial_picklist_groupings):
    # unofficial picklist grouping has a list item for each picklist 
    # each list that represents a picklist is comprised of strings as items which may have AND or ccharacters in it
    # ex unofficial picklist grouping: [['CS 2020 AND CS 2170', 'CS 2190']]
    for picklist in unofficial_picklist_groupings:
        for group in picklist:
            courses_in_item = extract_courses_from_string(group)
            
            if course in courses_in_item:
                return picklist
    return None

# relationship can be preqrequisites, corequisites, or strict_corequisites
# helper for getting co and strict coreqquisites
# could technically get preqrequisites at all but would be exclusive of co/strict requisites
# will also return duplicate requisite courses so we don't need to include them in graph if allow_messy True
# if allow_messy True returns all prereqs and what courses were removed bc of OR relationships
# if allow_messy False return prerequisites that should be solid lines (not OR relationship) and dotted 
# prereq (OR relationsihp) with the groupoing that makes the OR (Ex [B, C, D])
def requisites_helper_per_course(conn, course_id, institution_id, program_id, relationship=None, allow_messy=False):
    if relationship: 
        # we care about a particular relationship
        conn.execute("SELECT common_prefix, course_number \
                     FROM courses \
                     WHERE id IN (" \
                     "SELECT secondary_course_id FROM course_relationships \
                        WHERE institution_id = ? AND primary_course_id = ? \
                            AND relationship = ?)", [institution_id, course_id, relationship])
    else:
        # we want anything in table that has any type of relationship
        # get prerequisites (includes co and strict co)
        conn.execute("SELECT common_prefix, course_number, id \
                     FROM courses \
                     WHERE id IN (" \
                     "SELECT secondary_course_id FROM course_relationships \
                        WHERE institution_id = ? and primary_course_id = ?)", [institution_id, course_id])

    query_result = conn.fetchall()

    # get a list of prereq names (this has AND relationships separate)
    # ex. ["MATH A AND MATH B", "MATH C"] -> ["MATH A", "MATH B", "MATH C"]
    requisites = []
    course_name_to_id_map = {}
    for result in query_result:
        course_name = result[0] + " " + result[1]
        requisites.append(course_name)
        course_name_to_id_map[course_name] = result[2]

    # Need to make a copy because in below loop we want to iterate over the copy
    # while we mutate the original
    req_copy = copy(requisites)
    
    # all the "duplicate" requirement classes we will remove from list_course_names
    classes_removed = []

    # The way this is setup in db course id is NOT in the picklist, the picklist are the prerequiste
    # picklist to that course id
    unofficial_picklist_groupings = get_unofficial_picklists_groupings(conn, course_id)
    
    if allow_messy:
        # we want a list of prerequisites that are NOT part of an unofficial picklist
        prerequisite_solid = []
        # we want a list of prerequisites that ARE part of an unofficial picklist 
        prerequisite_dotted = []
        
        for prereq in requisites:
            grouping_found = get_grouping_for_class(prereq, unofficial_picklist_groupings)
            if grouping_found:
                prerequisite_dotted.append((prereq, grouping_found))
            else:
                prerequisite_solid.append(prereq)
        return prerequisite_solid, prerequisite_dotted        
    else:
        # remove duplicate requirements. For example if a preq to this course is one of [A, B, C],
        # make sure either A or B or C is in prereqs list but not more than one
        for i in range(len(req_copy)):
            course_name = req_copy[i]
            
            if course_name not in classes_removed:
                check_course_in_unofficial_picklist(course_name, course_name_to_id_map, unofficial_picklist_groupings, requisites, classes_removed)

        return requisites, classes_removed
    

# fn that gets corequisites for a given course and program
def get_corequisites_per_course(course_id, institution_id, program_id):
    conn = get_db_connection()
    corequisites, _ = requisites_helper_per_course(conn, course_id, institution_id, program_id, "corequisites")
    conn.close()

    return corequisites

# fn that gets strict strict_corequisites for a given course and program
def get_strict_corequisites_per_course(course_id, institution_id, program_id):
    conn = get_db_connection()
    corequisites, _ = requisites_helper_per_course(conn, course_id, institution_id, program_id, "strict_corequisites")
    conn.close()

    return corequisites

# returns list of tuples where each tuple is (main course, coreq or strict coreq)
# tuple entries are combo of course prefix and course number separate by space
def requisites_helper(institution_id, relationship):
    conn = get_db_connection()
    
    conn.execute("SELECT pc.common_prefix, pc.course_number, sc.common_prefix, sc.course_number \
                 FROM course_relationships cr \
                 JOIN courses pc ON (cr.primary_course_id = pc.id) \
                 JOIN courses sc ON (cr.secondary_course_id = sc.id) \
                 WHERE cr.relationship = ? AND cr.institution_id = ?",
                 [relationship, institution_id])
    
    query_result = conn.fetchall()
    
    requisites = set()
    
    for result in query_result:
        requisites.add((result[0] + " " + result[1], result[2] + " " + result[3]))
    
    conn.close()
    
    return sorted(list(requisites))
                 
                 
# get all corequisites for a program
# returns list of tuples of class prefix and numbers
def get_corequisites(institution_id):
    return requisites_helper(institution_id, "corequisites")

# get all strict coreqs for a program
# returns list of tuples of class prefix and numbers
def get_strict_corequisites(institution_id):
    return requisites_helper(institution_id, "strict_corequisites")

# fn that gets overall university data
def get_university_data(university_name, program_name):
    university_data = {}

    university_data["university"] = university_name
    university_data["program_name"] = program_name

    institution_id = get_university_id(university_name)

    if not institution_id:
        raise Exception(f"Institution id not found for {university_name}")

    program_id = get_program_id(university_name, program_name)

    if not program_id:
        raise Exception(f"Program id not found for {university_name}, with program name {program_name}")


    # open db connection
    conn = get_db_connection()

    # get location
    conn.execute("SELECT city, state FROM location_information \
                 WHERE id = (SELECT location_id FROM institutions \
                    WHERE id = ? ORDER BY created_at DESC LIMIT 1)", [institution_id])
    query_result = conn.fetchone()
    location = query_result[0] + ", " + query_result[1] if query_result else None

    university_data["location"] = location

    # get school
    conn.execute("SELECT school_name FROM schools \
                 WHERE id = \
                 (SELECT school_id FROM programs \
                 WHERE id = ?)", [program_id])
    query_result = conn.fetchone()

    school = query_result[0] if query_result else None
    university_data["school"] = school

    # get remaining info from program table
    conn.execute("SELECT department, degree, ai_program, curriculum_link, max_classes_per_term \
                 FROM programs WHERE id = ? \
                 ORDER BY created_at DESC LIMIT 1", \
                    [program_id])
    query_result = conn.fetchone()

    university_data["department"] = query_result[0]
    university_data["degree"] = query_result[1]
    university_data["ai_program"] = query_result[2]
    university_data["curriculum_link"] = query_result[3]
    university_data["max_classes_per_term"] =query_result[4]

    # get semester schedule (column may not exist in older DBs)
    try:
        conn.execute("SELECT semester_schedule FROM institutions \
                     WHERE id = ? ORDER BY created_at DESC LIMIT 1",
                     [institution_id])
        query_result = conn.fetchone()
        university_data["semester_schedule"] = query_result[0] if query_result else None
    except Exception:
        university_data["semester_schedule"] = None

    # close db connection
    conn.close()

    return university_data
