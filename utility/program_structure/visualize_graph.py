#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 13:34:32 2025

@author: carterithier

Given a json file or identifying program information at a university (in the database),
this will create a html file with a visualization of the curriculum graph along with 
labels, tooltips, etc

It will not include any AI related information that requires calculations. Just literally
drawing the graph. This is helpful to have a visual when debugging the main visualization
file (visualize_graph) which does do calculations.
"""

from . import build_graph_from_db as bg_db
from . import build_graph_helper as g
from bs4 import BeautifulSoup
from networkx.drawing.nx_agraph import to_agraph
from pyvis.network import Network as net
from IPython.display import display, HTML
import argparse
import os
from . import db_queries_helper as db
from database.config import config

# this returns a string that is in html markup language with university details
# such as location, school, degree, etc
def add_university_details(G):
    result = ""
    
    for attr in g.graph_attribute_map.keys():
        if attr != "max_classes_per_term":
            if G.graph.get(attr, None):
                if attr == "curriculum_link":
                    result += f"<a href = {G.graph[attr]}> {g.graph_attribute_map[attr]} </a>\n"
                elif attr != "university":
                    result += f"<i>{g.graph_attribute_map[attr]}</i>: {G.graph[attr]}<br>"
    
    return result

# taken from claude and chatgpt
# make a legend that tells you what each color node represents
def  add_color_legend(soup, picklist_to_bg_color_map):
    """Add a color legend to the visualization"""
    # Create a container div for the legend
    legend_div = soup.new_tag("div")
    legend_div["class"] = "legend"
    legend_div["style"] = """
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        font-family: Arial, sans-serif;
        font-size: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        z-index: 1000;
    """

    # Title: AI vs Non-AI
    ai_title = soup.new_tag("div")
    ai_title.string = "AI vs Non-AI"
    ai_title["style"] = "font-weight: bold; margin-top: 12px; margin-bottom: 8px; text-align: center;"
    legend_div.append(ai_title)

    # Non-AI class (circle)
    non_ai_div = soup.new_tag("div")
    non_ai_div["style"] = "display: flex; align-items: center; margin: 5px 0;"

    circle = soup.new_tag("div")
    circle["style"] = """
        width: 15px; height: 15px;
        border-radius: 50%;
        background-color: transparent;
        margin-right: 8px;
        border: 2px solid black;
    """
    non_ai_div.append(circle)

    non_ai_label = soup.new_tag("span")
    non_ai_label.string = "Non-AI class"
    non_ai_div.append(non_ai_label)

    legend_div.append(non_ai_div)

    # AI class (triangle)
    ai_div = soup.new_tag("div")
    ai_div["style"] = "display: flex; align-items: center; margin: 5px 0;"

    # Container for triangle
    triangle_wrapper = soup.new_tag("div")
    triangle_wrapper["style"] = "position: relative; width: 16px; height: 16px; margin-right: 8px;"

    # Outer triangle (black)
    outer_triangle = soup.new_tag("div")
    outer_triangle["style"] = """
        width: 0;
        height: 0;
        border-left: 9px solid transparent;
        border-right: 9px solid transparent;
        border-bottom: 16px solid black;
        position: absolute;
        top: 0;
        left: 0;
    """

    # Inner triangle (white, smaller, centered)
    inner_triangle = soup.new_tag("div")
    inner_triangle["style"] = """
        width: 0;
        height: 0;
        border-left: 7px solid transparent;
        border-right: 7px solid transparent;
        border-bottom: 12px solid white;
        position: absolute;
        top: 2px;
        left: 2px;
    """

    triangle_wrapper.append(outer_triangle)
    triangle_wrapper.append(inner_triangle)
    ai_div.append(triangle_wrapper)

    ai_label = soup.new_tag("span")
    ai_label.string = "AI class"
    ai_div.append(ai_label)

    legend_div.append(ai_div)
    
    # Add a title for the legend
    legend_title = soup.new_tag("div")
    legend_title.string = "Course Types"
    legend_title["style"] = "font-weight: bold; margin-bottom: 8px; text-align: center;"
    legend_div.append(legend_title)
    
    # Add each course type with its color
    for course_type, readable_name in g.course_type_map.items():
        color = g.course_color_map[course_type]
        
        # Create a row for this type
        item_div = soup.new_tag("div")
        item_div["style"] = "display: flex; align-items: center; margin: 5px 0;"
        
        # Add color box
        color_box = soup.new_tag("div")
        color_box["style"] = f"width: 15px; height: 15px; background-color: {color}; margin-right: 8px; border: 1px solid #ccc;"
        item_div.append(color_box)
        
        # Add label
        label = soup.new_tag("span")
        label.string = readable_name
        item_div.append(label)
        
        # Add to legend
        legend_div.append(item_div)
    
    return legend_div
    

# Adds header with university info and legend
def add_to_html(html_name, G):
    # load the file
    with open(html_name) as inf:
        txt = inf.read()
        soup = BeautifulSoup(txt, features="html.parser")
    
    # make new div that has all the content at the university level we want
    new_div = soup.new_tag("div")
    new_div["class"] = "header"
    
    # add univeristy name as header
    new_header = soup.new_tag("h1")
    new_header.string = G.graph["university"]
    new_header["style"] = "text-align:center; font-size:32px; font-weight:bold; margin-top: 20px 0"
    new_div.append(new_header)
   
    # add details like location, school, degree, etc
    p_tag = soup.new_tag("p")
    p_content =  BeautifulSoup(add_university_details(G), "html.parser")
    p_tag.append(p_content)
    p_tag["style"] = "text-align:center; font-size:15px; margin-top: 20px 0"
    new_div.append(p_tag)
    
    # Add color coded legend
    legend_div = add_color_legend(soup, G.graph["picklist_to_bg_color_map"])
    new_div.append(legend_div)

    
    # insert new div into DOM 
    card = soup.find("div", {"class": "card"})
    card.insert(0, new_div)
    
    # center the graph so it lines up with the above added info
    style_tag = soup.find("style")
    if "#mynetwork" in style_tag.text:
        css_content = style_tag.string
        
        css_content = css_content.replace("float: left", "margin: 0 auto")
        
        style_tag.string = css_content

    # save the file again
    with open(html_name, "w") as outf:
        outf.write(str(soup))



def display_graph(G, path=None):
    # get graph that just has node names and edges and not all the extra attributes we 
    # keep track of
    G_bare = G.__class__()
    G_bare.add_nodes_from(G.nodes())
    G_bare.add_edges_from(G.edges())

    # create an agraph with proper layout
    A = to_agraph(G_bare)
    A.graph_attr.update(ranksep="1")
    A.layout(prog="dot")

    # extract position from agraph and add to our original graph
    for node in A.nodes():
        pos = node.attr["pos"] # stored as "x,y"
        if pos:
            x, y = map(float, pos.split(","))
            G.nodes[node.name]["pos"] = (x, y) 

    # display graph
    nt = net("750px", "1000px", notebook=False, directed=True)
    nt.toggle_physics(False)
    
    options = """
    {
      "nodes": {
          "color": {
          "inherit": false
        },
        "borderWidth": null,
        "borderWidthSelected": null,
        "opacity": null,
        "size": null
      },
      "edges": {
          "color": {
          "inherit": false
        },
        "arrows": {
          "to": {
            "enabled": true
          }
        }
      }
    }
    
    """
    
    nt.set_options(options)

    # add nodes with precomputed positions
    scale = 2.4 #1.8
    for node, data in G.nodes(data=True):
        x,y = data["pos"]

        # if we have a class name that is different than the course code (ex course code MATH 324
        # with class name Linear Algebra), add this info to the label
        label = node
        if data["class_name"] != node:
            label += f"\n{data["class_name"]}"

        nt.add_node(node,
                    label=label,
                    title=data["title"],
                    color=data.get("color", "#97C2FC"),
                    x=x*scale,
                    y=-y*scale, # flip y axis for graphviz -> pyvis
                    size=10,
                    physics=False)
    
    # now we need to add edges for each node in our nt
    # now we need to add edges for each node in our nt
    for source, target in G.edges():
        edge_data = G.get_edge_data(source, target)
        dotted = True if G.get_edge_data(source, target) and edge_data.get("OR_rel", False) else False
        nt.add_edge(source, target, dashes=dotted)

    def join_str(name):
        return "_".join(name.split())

    university = join_str(G.graph['university'])
    program_name = join_str(G.graph["program_name"])

    html_name = f"{university}_{program_name}.html"

    if path:
        html_name = path + "/" + html_name
    else:
        # default to program stucture directory
        html_name = str(config.PROJ_ROOT) + f"/html/{html_name}"

    nt.show(html_name, local=True, notebook=False)
    
    # edit html for extra things we want to include
    add_to_html(html_name, G)
    
    
def main(args):
    has_prereq_data = db.has_scraped_prereqs(args.university_name, args.program_name)

    if has_prereq_data:
        G = bg_db.build_graph(args.university_name, args.program_name, allow_messy=True)
    else:
        print("==========================================================")
        print(f"No prereq data for {args.university_name}: {args.program_name}")
        print("\n")
        return
    
    display_graph(G)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Display curriculum graph of degree program. Provide university and program name")
    parser.add_argument("-u", "--university_name", type=str, help="Name of university that matches database entry", required=True)
    parser.add_argument("-p", "--program_name", type=str, help="Name of program that matches database entry", required=True)
    
    args = parser.parse_args()

    main(args)
    