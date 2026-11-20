#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 13:34:07 2025

@author: carterithier

Build a networkx G object based on data in the database for a particular
university and program name
"""

import networkx as nx
import re as _re
from utility.program_structure.build_graph_helper import *
from utility.program_structure import db_queries_helper as db
from copy import copy

# make a deep copy of the graph so any manipulations we do doesn't affect
# our original graph structure
def deep_copy_graph(orig_G):
    G = orig_G.__class__()
    G.add_nodes_from(orig_G)
    G.add_edges_from(orig_G.edges)
    
    return G

def _find_or_group(prereq, grouping):
    """Return which AND-sub-group index 'prereq' belongs to in a flat grouping list.

    Example: grouping ["A", "B AND C"] → sub-groups [["A"], ["B", "C"]]
    "B" → 1,  "A" → 0
    """
    for i, item in enumerate(grouping or []):
        if ' AND ' in item.upper():
            parts = [p.strip() for p in _re.split(r'\s+AND\s+', item, flags=_re.IGNORECASE)]
            if prereq in parts:
                return i
        else:
            if prereq == item.strip():
                return i
    return 0  # fallback: assign to first group

# taken from chatgpt
# course descriptions can be very long and get cutoff in title visualization
# this will include line breaks to make sure all of it is visible in tooltip
# specifically for course description
def wrap_tooltip(text, max_line_length=95):
    """Insert <\n> tags every `max_line_length` characters without breaking words."""
    if not text:
        return ""
    
    # add course description part because that will be part of line
    text = "Course description: " + text
    
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line + " " + word) <= max_line_length:
            line += (" " if line else "") + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    result = "\n".join(lines)
    return result[len("Course description: "):]

def add_data_to_graph(G, course_id, edges_list, university_id, program_id,
                      picklist_to_bg_color_map, 
                      bg_colors_copy, university, allow_messy = False):
    bg_color = None

    course_data = db.get_course_data(course_id, university_id, program_id, allow_messy)
    
    # course attributes is what we pass to the node to make sure it has all needed metadata
    # in graph we are building
    if allow_messy:
        course_attrs = {attr: course_data.get(attr, None) for attr in ["course_description", \
                                                                    "course_url", "picklist", \
                                                                    "course_hours", "prerequisites, solid", \
                                                                    "prerequisites, dotted", \
                                                                    "classes_removed_this_iteration", \
                                                                    "course_type"]}
    else:
        course_attrs = {attr: course_data.get(attr, None) for attr in ["course_description", \
                                                                    "course_url", "picklist", \
                                                                    "course_hours", "prerequisites", \
                                                                    "classes_removed_this_iteration", \
                                                                     "course_type"]}
    course_attrs["class_name"] = course_data.get("title", "")
    course_attrs["is_AI_class"] = None

    if allow_messy and course_data["prerequisites, dotted"]:
        course_attrs["unofficial_picklist_reqs"] = []
        for _, picklist in course_data["prerequisites, dotted"]:
            if picklist not in course_attrs["unofficial_picklist_reqs"]:
                course_attrs["unofficial_picklist_reqs"].append(picklist)


    # add picklist description and color to map if it doesn't already exist
    picklists = course_data["picklist"]
    if picklists:
        # TODO: for now for visualization, we only label one of possible picklists color wise
        # it is modeled correclty in db though and when hovering over, will see other picklist
        picklist = picklists[0]
        if picklist not in picklist_to_bg_color_map.keys():
            picklist_to_bg_color_map[picklist] = bg_colors_copy[0]
            bg_colors_copy.remove(picklist_to_bg_color_map[picklist])
            
        bg_color = picklist_to_bg_color_map[picklist]
    
    # Preserve full picklist membership names for downstream visualizations (interactive_graphs clustering, tooltips, etc.)
    # `picklist` below is later overwritten to a 0/1 flag for layout logic, so we keep names separately.
    course_attrs["picklist_names"] = picklists or []
    
    # keep track of this so we know if we need to definitely keep this course in our map
    course_attrs["picklist"] = 1 if picklists else 0
    
    # build portion that will show up when hovered over (called the node "title")
    title_str = f"""
        Course name: {course_data.get("title", "Unknown")}\n
        Course code: {course_data["course_code"]}\n
        Course description: {wrap_tooltip(course_data["course_description"])}\n
        Course URL: {course_data["course_url"]}\n
        Course type: {make_readable_course_type(course_data["course_type"])}\n
        Picklist: {course_data["picklist"]}\n
        Course hours: {course_data["course_hours"]}
    """
    
    course_attrs["title"] = title_str
    color_name = course_color_map[course_attrs["course_type"]]
    
    # set background color if it exists
    color_bg = {
        "background": color_name,
        "border": bg_color,
        # 'highlight': {
        #     'background': color_name,
        #     'border': bg_color
        # }
        
    }
    
    color_no_bg = {
        "background": color_name
    }
    
    color = color_bg if bg_color else color_no_bg
    
    # add a border width if there's a background color
    if bg_color:
        course_attrs["borderWidth"] = 7
    
    # add the node to the graph
    # this color attribute determines the color the node will be displayed as
    # for some reason to display correctly it must be passed directly to add node instead of within course_attrs
    G.add_node(course_data["course_code"], color=color, **course_attrs)

    if allow_messy:
        # keep track of any prerequisites (edges) we need to add 
        for prereq in course_data["prerequisites, solid"]:
            edge = (prereq, course_data["course_code"])
            G.add_edge(edge[0], edge[1], OR_rel=False, finalized=True)
            edges_list.append(edge)
        
        # Build or_set mapping: each distinct picklist (independent OR condition)
        # gets its own sequential index. Edges sharing a picklist share an or_set.
        _or_set_map = {}
        _or_set_idx = 0
        for prereq, grouping in course_data["prerequisites, dotted"]:
            edge = (prereq, course_data["course_code"])
            _key = tuple(grouping) if grouping else ()
            if _key not in _or_set_map:
                _or_set_map[_key] = _or_set_idx
                _or_set_idx += 1
            or_set  = _or_set_map[_key]
            or_group = _find_or_group(prereq, grouping)
            G.add_edge(edge[0], edge[1], OR_rel=True, finalized=False,
                       or_group=or_group, or_set=or_set)
            edges_list.append(edge)

    else:
        for prereq in course_data["prerequisites"]:
            edge = (prereq, course_data["course_code"])
            G.add_edge(edge[0], edge[1])
            edges_list.append(edge)

# this function will create nodes for all courses in a particular program
# attached to each node it will add attribute data such as descripiont, url, node color, etc
# program name is defined as : program["program_name"] = ",".join([self.data_json["degree"], self.data_json["ai_program"] if self.data_json["ai_program"] else "none"])
# -obsolete classifier is the model trained in utility/models
def add_courses_and_edges(G, edges_list, university_name, program_name,  
                          picklist_to_bg_color_map, bg_colors_copy, allow_messy=False):
    university_id = db.get_university_id(university_name)
    program_id = db.get_program_id(university_name, program_name)
    course_ids = db.get_list_of_program_course_ids(program_id)

    for course_id in course_ids:
        add_data_to_graph(G, course_id, edges_list, university_id, program_id, 
                      picklist_to_bg_color_map, bg_colors_copy, university_name, allow_messy)

    # save final color mapper for picklists in the graph
    G.graph["picklist_to_bg_color_map"] = picklist_to_bg_color_map

    # save whether messy or not
    G.graph["messy"] = allow_messy

def post_process_graph(G, edges_list, university_name, program_name,
                       picklist_to_bg_color_map, bg_colors_copy, allow_messy):
    university_id = db.get_university_id(university_name)
    program_id = db.get_program_id(university_name, program_name)

    all_nodes = list(G.nodes())
    before_count = len(all_nodes)

    for node in all_nodes:
        if not G.nodes[node]:
            course_id = db.get_course_id(node, university_id)
            add_data_to_graph(G, course_id, edges_list, university_id, program_id, 
                      picklist_to_bg_color_map, bg_colors_copy,  university_name, allow_messy)
    
    after_count = len(list(G.nodes))

    # if the nodes count is different it means we added nodes that were not part of the 
    # program in the database but may be prereqs or prereqs of prereqs, etc
    # we should keep running this process until the node count doesn't change
    # meaning we have gotten all data from courses in teh database that were scraped
    if before_count != after_count:
        post_process_graph(G, edges_list, university_name, program_name,  
                       picklist_to_bg_color_map, bg_colors_copy, allow_messy)

    # save final color mapper for picklists in the graph
    G.graph["picklist_to_bg_color_map"] = picklist_to_bg_color_map

def prereq_not_needed(G, course):
    # find classes the course is a prereq for
    courses_this_is_prereq_for = []

    for edge in G.edges():
        if edge[0] == course:
            courses_this_is_prereq_for.append(edge[1])

    # go through each course in the list we just made
    for target_course in courses_this_is_prereq_for:
        # find the prereqs and classes removed for each course in our list
        prereqs = G.nodes[target_course]["prerequisites"]
        classes_removed = G.nodes[target_course]["classes_removed_this_iteration"]

        # find prereqs we care about (they involve classes that were not removed)
        preqreqs_we_care_about = copy(prereqs)
        for course_to_remove in classes_removed:
            if course_to_remove in preqreqs_we_care_about:
                preqreqs_we_care_about.remove(course_to_remove)

        # check if I remove this course that the prereqs we care about are still satisfied
        G_copy = deep_copy_graph(G)
        G_copy.remove_node(course)

        for prereq in preqreqs_we_care_about:
            if (prereq, target_course) not in G_copy.edges():
                # we cannot remove it
                return False
    
    return True


def remove_extranous_courses(G):
    # remove duplicate requisite courses
    duplicate_courses = db.get_duplicated_requirement_courses()

    for course in duplicate_courses:
        if G.in_degree(course) == 0 and not G.nodes[course]["picklist"] and \
            prereq_not_needed(G, course):
            G.remove_node(course)


# program name is defined as : program["program_name"] = ",".join([self.data_json["degree"], self.data_json["ai_program"] if self.data_json["ai_program"] else "none"])
def build_graph(university_name, program_name, allow_messy=True):
    # make a directed graph
    G = nx.DiGraph() 

    university_data = db.get_university_data(university_name, program_name)

    # assign graph attributes related to the institution
    for attr in graph_attribute_map.keys():
        if university_data.get(attr, None):
            G.graph[attr] = university_data[attr]
    
    G.graph["semester_schedule"] = university_data["semester_schedule"]
    
    # build graph
    edges_list = []

    # keep track of picklists and what colors they map to
    # key is picklist name, value is background hex color
    picklist_to_bg_color_map = {}

    # keep track of colors beign used in graph
    bg_colors_copy = copy(bg_colors)
    
    add_courses_and_edges(G, edges_list, university_name, program_name, 
                          picklist_to_bg_color_map, bg_colors_copy, allow_messy)

    # In above we may have some nodes added that are prereqs of a program but not core
    # to the program so they have no info (ex. G.nodes["MATH 100"] = {})
    # Here we find those nodes and add data to them if we find them or at least make sure
    # they node has all necessary fields
    post_process_graph(G, edges_list, university_name, program_name,
                       picklist_to_bg_color_map, bg_colors_copy, allow_messy)

    if not allow_messy:
        remove_extranous_courses(G)

    return G

# same as build_graph but undirected (prereq direction is dropped). Node, edge and
# graph attributes are kept. If both A->B and B->A exist (ex. corequisites) they
# become one edge, keeping the attributes of only one of the two directions
def build_undirected_graph(university_name, program_name, allow_messy=True):
    return build_graph(university_name, program_name, allow_messy).to_undirected()




    
    
    
    
    
