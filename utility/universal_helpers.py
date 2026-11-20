import re
import duckdb
import sys
import os

import database.config.config as config

def get_db_connection():
    # open db connection
    conn = duckdb.connect(config.PROJ_ROOT / config.DB_NAME)

    # make queries case insensitive
    conn.execute("SET default_collation = 'nocase'")
    
    return conn

def parse_course_code(course_code):
        """
        Parse course code into prefix and number components.
        Handles different formats in the original JSON like:
        - "CS 101"
        - "CSCI-141"
        - "CSCI141"
        - "CSCI electives"
        - "CAS CS 120" 

        - TODO: can't handle STA 301 L

        For the edge case Rutgers which has format AA:BBB:CCC we do the prefix
        as AA:BBB and number component as CCC
        """
        course_code = course_code.upper()

        if not course_code:
            return "UNKNOWN", "000"
            
        # Handle formats with spaces (e.g., "CS 101")
        parts = course_code.split()

        def has_digit(string):
            for character in string:
                if character.isdigit():
                    return True
            return False

        if len(parts) > 2:
            l = len(parts)
            if has_digit(parts[l-1]):
                # very last part is number, preceding parts are all prefix
                return " ".join(parts[:l-1]), parts[l-1] 
            else:
                # handle case like CAS CS electives
                return " ".join(parts[:l-1]), "000"         

        if len(parts) == 2:
            # Check if second part has a number
            if has_digit(parts[1]):
                return parts[0], parts[1]
            else:
                # Handle case like "CSCI electives"
                return parts[0], "000"
                
        # Handle hyphenated formats (e.g., "CSCI-141")
        if '-' in course_code:
            parts = course_code.split('-')
            l = len(parts)
            if has_digit(parts[l-1]):
                # very last part is number, preceding parts are all prefix
                return " ".join(parts[:l-1]), parts[l-1] 
            else:
                # handle case like CAS CS electives
                return " ".join(parts[:l-1]), "000"   
        
        # Handle dotted formats like Rutgers (e.g., "01:198:111")
        if ':' in course_code:
            parts = course_code.split(':')
            l = len(parts)
            if has_digit(parts[l-1]):
                # very last part is number, preceding parts are all prefix
                return " ".join(parts[:l-1]), parts[l-1] 
            else:
                # handle case like CAS CS electives
                return " ".join(parts[:l-1]), "000"   
                
        # Handle formats without spaces (e.g., "CSCI141")
        # Find the boundary between letters and numbers
        match = re.search(r'([A-Za-z]+)(\d.*)', course_code)
        if match:
            return match.group(1), match.group(2)
            
        # Fallback for other formats
        return course_code, "000"

# given a complex prereq relationship string (nested AND/OR)
# return list of courses that make it up
def extract_courses_from_string(s):
    allowed_characters = [":", "-", ".", " "]
    # this will remove quotes and brackets and commas
    simplified_s = "".join([c for c in s if (c.isalnum() or c in allowed_characters)])

    # remove AND keyword
    simplified_s = simplified_s.replace("AND", "")

    # we should now have a string of just course prefixes and numbers. There's two scenarios:
    #      1) course prefix + course number have space in between them (most common)
    #      2) course prefix + course number don't haven't space in between them
    split_s = simplified_s.split(" ")

    # remove extra spaces
    split_s = [w for w in split_s if (w != " " and w!= "")]

    if len(split_s) == 1:
        return [parse_course_code(split_s[0])]
    else:
        result = []
        # determine if case 1 or 2

        # if case 2
        course_code, number = parse_course_code(split_s[0])
        if number == "000":
            # it's case 1
            for i in range(0, len(split_s) - 1, 2):
                result.append(f"{split_s[i]} {split_s[i+1]}")
        else:
            result.append(f"{course_code} {number}")
            for i in range(1, len(split_s)):
                course_code, number = parse_course_code(split_s[i])

                if number == "000":
                    raise Exception(f"trouble parsing {split_s} without spaces")

                result.append(f"{course_code} {number}")

    return result

# Category can be ("Major", "Minor", "Concentration", "Elective", "Certificate", "None")
def get_all_uni_from_category(category, semester_only):
    uni_prog = get_all_universities_and_programs_by_category(category, semester_only)
    majors_universities = [u[0] for u in uni_prog]
    return set(majors_universities)

# Category can be ("Major", "Minor", "Minor with major", "Minor without major", "Concentration", "Concentration with major", "Concentration without major", "Elective", "Certificate", "None")
def get_all_universities_and_programs_by_expanded_category(category, semester_only=False):
    regular_categories = ["Major", "Minor", "Concentration", "Elective", "Certificate", "None"]

    if category in regular_categories:
        return get_all_universities_and_programs_by_category(category, semester_only)
    else:
        options = ["Minor with major", "Minor without major", "Concentration with major", "Concentration without major"]
        if category not in options:
            raise Exception(f"{category} not a valid option. Valid options are {options}")
        
        majors_unis = get_all_uni_from_category("Major", semester_only)

        other_category = "Minor" if "Minor" in category else "Concentration"
        
        other_category_unis = get_all_uni_from_category(other_category, semester_only)

        if "with major" in category:
            final_unis = other_category_unis & majors_unis
        else:
            final_unis = other_category_unis - majors_unis
        
        # final unis now has a list of unique universities that meet our criteria. However, we want (university, program)
        # pairs in our return list that we can operate on so now we have to take the output of get all univeristies and
        # programs by category for the category in question and filter out the pairs we want to keep which are the pairs
        # where the university is in final_unis

        unfiltered = get_all_universities_and_programs_by_category(other_category, semester_only)
        filtered = [p for p in unfiltered if p[0] in final_unis]

        return filtered

# Category can be ("Major", "Minor", "Concentration", "Elective", "Certificate", "None")
def get_all_universities_and_programs_by_category(category, semester_only=False):
     # list of tuples. First element in tuple is university name
    # second element in tuple is program name
    universities_programs = []

    conn = get_db_connection()

    if semester_only:
        conn.execute(f"""
            select distinct p.id, i.institution_name, p.program_name
            from programs p join institutions i on (p.institution_id = i.id)
            where ai_program = '{category}' and i.semester_schedule is distinct from false
            order by i.institution_name, p.program_name
            """)
    else:
        conn.execute(f"""
            select distinct p.id, i.institution_name, p.program_name
            from programs p join institutions i on (p.institution_id = i.id)
            where ai_program = '{category}' 
            order by i.institution_name
            """)
    
    program_ids = conn.fetchall()

    for data in program_ids:
        universities_programs.append((data[1], data[2]))
    
    conn.close()

    return universities_programs


# AI majors with no courses linked to them in eaai27.db, so build_graph returns an
# empty graph (0 nodes) that the similarity measures can't use. Found by building
# every major with allow_messy=True; remove a pair once its courses are in the db
EMPTY_AI_MAJORS = {
    ("Boise State University", "Bachelor of Science in Artificial Intelligence Science"),
    ("Texas Tech University", "Online Bachelor of Science in Human-Centered Artificial Intelligence"),
    ("University of Arkansas at Little Rock", "Bachelor of Science in Artificial Intelligence"),
    ("University of Kentucky", "Bachelor of Science in Artificial Intelligence"),
    ("University of Wisconsin-Stevens Point", "Bachelor of Science in Artificial Intelligence"),
    ("University of Wisconsin-Whitewater", "Bachelor of Science Artificial Intelligence"),
    ("Worcester Polytechnic Institute", "Bachelor of Science in Artificial Intelligence"),
}

# AI majors whose build_graph(..., allow_messy=True) graph has fewer than 20 nodes or
# fewer than 10 edges, i.e. too little prereq data captured to compare structure.
# Hardcoded because building all the graphs to check takes ~40 seconds
SMALL_AI_MAJORS = {
    ("Adelphi University", "Bachelor of Science in Artificial Intelligence"),  # 25 nodes, 0 edges
    ("Angelo State University", "Bachelor of Science in Artificial Intelligence"),  # 46 nodes, 0 edges
    ("College of Our Lady of the Elms", "Data Science and Artificial Intelligence"),  # 30 nodes, 0 edges
    ("Florida International University", "Bachelor of Science in Data Science and Artificial Intelligence"),  # 47 nodes, 0 edges
    ("Indiana University-Indianapolis", "Bachelor of Science in Artificial Intelligence"),  # 54 nodes, 0 edges
    ("Indiana University-Indianapolis", "Bachelor of Arts in Artificial Intelligence"),  # 53 nodes, 0 edges
    ("Keiser University-Ft Lauderdale", "Bachelor of Science in Artificial Intelligence"),  # 1 nodes, 0 edges
    ("Massachusetts Institute of Technology", "Bachelor of Science in Artificial Intelligence and Decision Making"),  # 44 nodes, 0 edges
    ("Mercyhurst University", "Artificial Intelligence and Data Science Bachelor of Science"),  # 18 nodes, 0 edges
    ("Miami Dade College", "Applied Artificial Intelligence Bachelor of Science"),  # 31 nodes, 0 edges
    ("Purdue University-Main Campus", "Bachelor of Science in Artificial Intelligence"),  # 44 nodes, 2 edges
    ("Purdue University-Main Campus", "Bachelor of Arts in Artificial Intelligence"),  # 27 nodes, 0 edges
    ("Saint Johns University", "Bachelor of Science in Computing and Machine Learning"),  # 18 nodes, 0 edges
    ("Saint Peter's University", "Bachelor of Science in Data Science and Artificial Intelligence"),  # 27 nodes, 0 edges
    ("Temple University", "Bachelor of Science in Artificial Intelligence"),  # 1 nodes, 0 edges
    ("University of California-Santa Barbara", "Bachelor of Science in Artificial Intelligence"),  # 23 nodes, 0 edges
    ("University of Maryland Global Campus", "Bachelor of Science in Artificial Intelligence"),  # 20 nodes, 0 edges
    ("University of the Cumberlands", "Bachelor of Science in Artificial Intelligence"),  # 17 nodes, 4 edges
}

# same as get_all_universities_and_programs_by_category("Major") but without the
# majors in EMPTY_AI_MAJORS or SMALL_AI_MAJORS. Every pair returned builds a graph
# with at least 20 nodes and 10 edges with build_graph(..., allow_messy=True);
# with allow_messy=False some raise a KeyError
def get_verified_ai_majors(semester_only=False):
    majors = get_all_universities_and_programs_by_category("Major", semester_only)
    excluded = EMPTY_AI_MAJORS | SMALL_AI_MAJORS
    return [p for p in majors if p not in excluded]

def get_all_universities_and_programs(semester_only=False):
    # list of tuples. First element in tuple is university name
    # second element in tuple is program name
    universities_programs = []

    conn = get_db_connection()

    # get list of unique universities in alphabetical order
    if semester_only:
        conn.execute("""
                     SELECT distinct id, institution_name 
                     FROM institutions 
                     WHERE semester_schedule is distinct from false 
                     ORDER BY institution_name
                     """)
    else:
        conn.execute("SELECT distinct id, institution_name FROM institutions ORDER BY institution_name")
    university_ids_names = conn.fetchall()

    # get programs for ea university
    for university_id, university_name in university_ids_names:
        conn.execute("SELECT distinct program_name FROM programs \
                     WHERE institution_id = ?", [university_id])
        query_result = conn.fetchall()

        for result in query_result:
            universities_programs.append((university_name, result[0]))

    # get list of program

    conn.close()

    return universities_programs

def get_universities_and_programs_not_on_semester_schedule(include_null=True):
    # list of tuples. First element in tuple is university name
    # second element in tuple is program name
    universities_programs = []

    conn = get_db_connection()

    # get list of unique universities in alphabetical order
    if include_null:
        conn.execute("""
                    SELECT i.institution_name, p.program_name
                    FROM institutions i
                    JOIN programs p ON p.institution_id = i.id
                    WHERE i.semester_schedule IS NOT TRUE
                    ORDER BY i.institution_name, p.program_name;
                    """)
    else:
        conn.execute("""
                    SELECT i.institution_name, p.program_name
                    FROM institutions i
                    JOIN programs p ON p.institution_id = i.id
                    WHERE i.semester_schedule IS FALSE
                    ORDER BY i.institution_name, p.program_name;
                    """)

    query_result = conn.fetchall()

    conn.close()

    return query_result
    

def get_percentages(counts, num_schools, ndigits=2):
    percentages = {}

    for key, value in counts.items():
        percentages[key] = round(value / num_schools, ndigits)
    
    return percentages