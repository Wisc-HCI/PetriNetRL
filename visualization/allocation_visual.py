import sys
import math
import os
import xml.etree.ElementTree as ET
import re
import textwrap
import csv
import argparse
from xml.etree.ElementTree import Element, SubElement, tostring

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from constants import *
from utils import *

# Function to create an SVG element
def create_svg(width, height):
    svg = Element('svg', width=str(width), height=str(height), xmlns="http://www.w3.org/2000/svg")
    return svg

# Function to embed an SVG by inlining its content
def embed_svg(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root

# Helper function to parse dimension attributes
def parse_dimension(value):
    return int(re.sub(r'[^0-9.]', '', value)) if value else 1

def create_half_circle(svg, x, y, radius, fill_left="gray", fill_right="black"):
    # Left semi-circle (counterclockwise)
    left_half = SubElement(svg, 'path', {
        'd': f"M {x} {y - radius} A {radius} {radius} 0 0 0 {x} {y + radius} Z",
        'fill': fill_left
    })

    # Right semi-circle (clockwise)
    right_half = SubElement(svg, 'path', {
        'd': f"M {x} {y - radius} A {radius} {radius} 0 0 1 {x} {y + radius} Z",
        'fill': fill_right
    })


def add_agent_bubble(svg, x, y, radius, agent_text, fill_color="lightgray", is_split=False):
    if not is_split:
        bubble = SubElement(svg, 'circle', cx=str(x), cy=str(y), r=str(radius), fill=fill_color)
        text = SubElement(svg, 'text', x=str(x), y=str(y + radius / 8), fill='white', 
                            style='font-size: {}px; text-anchor: middle; dominant-baseline: middle;'.format(radius))
        text.text = agent_text
    else:
        create_half_circle(svg, x, y, radius, fill_left=fill_color[0], fill_right=fill_color[1])
        left_text = SubElement(svg, 'text', x=str(x-radius/2.5), y=str(y + radius / 8), fill='white', 
                            style='font-size: {}px; text-anchor: middle; dominant-baseline: middle;'.format(radius))
        left_text.text = agent_text[0]
        right_text = SubElement(svg, 'text', x=str(x+radius/2.5), y=str(y + radius / 8), fill='white', 
                            style='font-size: {}px; text-anchor: middle; dominant-baseline: middle;'.format(radius))
        right_text.text = agent_text[1]


def add_primitive_bubble(svg, x, y, agent_bubble_radius, prim_bubble_radius, assignemnts, is_split=False, fill_color="red", non_human_agent=None):
    if not is_split:
        for idx, a in enumerate(assignemnts):
            theta = math.pi/5 * (idx+3)
            x_i = x + agent_bubble_radius*math.cos(theta)
            y_i = y + agent_bubble_radius*math.sin(theta)
            bubble = SubElement(svg, 'circle', cx=str(x_i), cy=str(y_i), r=str(prim_bubble_radius), fill=fill_color)
            text = SubElement(svg, 'text', x=str(x_i), y=str(y_i + prim_bubble_radius / 8), fill='white', 
                                style='font-size: {}px; text-anchor: middle; dominant-baseline: middle;'.format(prim_bubble_radius))
            text.text = a
    else:
        for idx, a in enumerate(assignemnts["Human"]):
            theta = math.pi/5 * (idx+3)
            x_i = x + agent_bubble_radius*math.cos(theta)
            y_i = y + agent_bubble_radius*math.sin(theta)
            bubble = SubElement(svg, 'circle', cx=str(x_i), cy=str(y_i), r=str(prim_bubble_radius), fill=fill_color[0])
            text = SubElement(svg, 'text', x=str(x_i), y=str(y_i + prim_bubble_radius / 8), fill='white', 
                                style='font-size: {}px; text-anchor: middle; dominant-baseline: middle;'.format(prim_bubble_radius))
            text.text = a
        for idx, a in enumerate(assignemnts[non_human_agent]):
            theta = math.pi + math.pi/5 * (-idx-3)
            x_i = x + agent_bubble_radius*math.cos(theta)
            y_i = y + agent_bubble_radius*math.sin(theta)
            bubble = SubElement(svg, 'circle', cx=str(x_i), cy=str(y_i), r=str(prim_bubble_radius), fill=fill_color[1])
            text = SubElement(svg, 'text', x=str(x_i), y=str(y_i + prim_bubble_radius / 8), fill='white', 
                                style='font-size: {}px; text-anchor: middle; dominant-baseline: middle;'.format(prim_bubble_radius))
            text.text = a

def add_outline(svg, x, y, length, height, radius=5, color="#FF0000"):
    path = SubElement(svg, 'path', d='M {} {} H {} A {} {} 0 0 1 {} {} V {} A {} {} 0 0 1 {} {} H {} A {} {} 0 0 1 {} {} V {} A {} {} 0 0 1 {} {} Z'.format(
        x+radius, y,
        x+length-radius,
        radius, radius, x+length, y+radius,
        y+height-radius,
        radius, radius, x+length-radius, y+height,
        x+radius,
        radius, radius, x, y+height-radius,
        y+radius,
        radius, radius, x+radius, y
        ),
        stroke=color, style='stroke: {}; fill: transparent'.format(color))

# Function to add text label above each allocation with wrapping
def add_allocation_label(svg, x, y, text_content, max_width):
    # Approximate character limit based on max_width
    char_limit = max_width // 7
    lines = textwrap.wrap(text_content, width=char_limit)

    label = SubElement(svg, 'text', x=str(x), y=str(y), fill='black', 
                       style='font-size: 14px; text-anchor: middle; font-weight: bold;')
    for i, line in enumerate(lines):
        tspan = SubElement(label, 'tspan', x=str(x), dy=str(15 if i > 0 else 0))
        tspan.text = line

def run(arguments):
    # Determine which input json file to use 
    f = None
    if arguments.input_file is not None:
        f = arguments.input_file

    # Load petrinet data from json (transitions, places)
    [_json_obj, _weights, json_task, _targets_obj, _primitives_obj, json_agents] = LOAD_JOB_FILE(f)

    task_steps = [str(i) for i in range(len(list(json_task.keys())))]
    tasks_ordered = [None for _ in json_task.keys()]
    task_assignment = [None for _ in json_task.keys()]
    primitive_assignment = [{} for _ in json_task.keys()]
    task_names = [None for _ in json_task.keys()]
    added_primitives_idxs = [-1 for _ in json_task.keys()]
    non_human_agent = None

    for key in json_task.keys():
        if tasks_ordered[json_task[key]["order"]-1] is None:
            tasks_ordered[json_task[key]["order"]-1] = [json_task[key]["name"]]
        else:
            tasks_ordered[json_task[key]["order"]-1].append(json_task[key]["name"])

    with open(arguments.csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if "decide" in row[1]:
                task_name = row[1].split(" decide")[0].strip()
                idx = -1
                for eidx, i in enumerate(tasks_ordered):
                    if i is not None and task_name in i:
                        idx = eidx
                task_assignment[idx] = row[5]
                task_names[idx] = task_name
                if row[5] == 'Human':
                    primitive_assignment[idx]["Human"] = []
                elif "Human" not in row[5]:
                    primitive_assignment[idx][row[5]] = []
                    if non_human_agent is None:
                        non_human_agent = row[5]
                else:
                    primitive_assignment[idx]["Human"] = []
                    other_agent = row[5].replace("Human", "").replace(";", "")
                    primitive_assignment[idx][other_agent] = []
                    if non_human_agent is None:
                        non_human_agent = other_agent
            elif "-" in row[1]:
                splits = row[1].split("-")
                task_name = splits[1].strip()
                try:
                    tidx = task_names.index(task_name)
                except:
                    tidx = -1
                if tidx >= 0 and added_primitives_idxs[tidx] == -1:
                    added_primitives_idxs[tidx] = row[0]
                    primitive_assignment[tidx][row[5]].append(row[2][0])
                elif tidx >= 0 and added_primitives_idxs[tidx] == row[0]:
                    primitive_assignment[tidx][row[5]].append(row[2][0])

    print(task_names)
    print(primitive_assignment)

    # Constants for layout
    initial_x_offset = 30
    padding = 20
    agent_bubble_radius = 20
    prim_bubble_radius = 5
    bubble_spacing = 5
    worker_robot_height = 20 + agent_bubble_radius
    worker_robot_width = agent_bubble_radius
    column_width = 2 * worker_robot_width
    icon_size = worker_robot_height 

    num_columns = len([x for x in task_names if x is not None])

    max_bubbles = num_columns
    svg_width = (num_columns+1) * (column_width + padding)
    svg_height = worker_robot_height + 3 * (icon_size + padding) + 2 * (agent_bubble_radius + bubble_spacing) + max_bubbles * (2 * agent_bubble_radius + bubble_spacing) + 40

    human_color = "gray"
    robot_color = "black"
    human_prim_color = "purple"
    robot_prim_color = "pink"

    svg = create_svg(svg_width, svg_height)

    offset_from_label = 10
    outline_padding = 10
    add_outline(svg, 
                initial_x_offset - column_width/4, #x
                worker_robot_height/2 + offset_from_label - outline_padding/2,  #y
                len([x for x in task_names if x is not None])*(column_width + padding),  #length
                worker_robot_height + outline_padding, #height
                color="#000000")

    for i, row in enumerate(task_assignment):
        x_offset = i * (column_width + padding)  + column_width / 2 + initial_x_offset
        task_name = task_names[i]
        
        # Add allocation label with wrapping
        if task_name is None:
            continue

        add_allocation_label(svg, x_offset, 15, str(i+1), column_width)

        if row is None:
            continue

        # Determine allocation
        if row == 'Human':
            add_agent_bubble(svg, x_offset, worker_robot_height + offset_from_label, agent_bubble_radius, "H", fill_color=human_color)
            add_primitive_bubble(svg, x_offset, worker_robot_height + offset_from_label, agent_bubble_radius, prim_bubble_radius, primitive_assignment[i]["Human"], is_split=False, fill_color=human_prim_color)
        elif "Human" not in row:
            add_agent_bubble(svg, x_offset, worker_robot_height + offset_from_label, agent_bubble_radius, "R", fill_color=robot_color)
            add_primitive_bubble(svg, x_offset, worker_robot_height + offset_from_label, agent_bubble_radius, prim_bubble_radius, primitive_assignment[i][non_human_agent], is_split=False, fill_color=robot_prim_color)
        else:
            add_agent_bubble(svg, x_offset, worker_robot_height + offset_from_label, agent_bubble_radius, ["H", "R"], is_split=True, fill_color=[human_color, robot_color])
            add_primitive_bubble(svg, x_offset, worker_robot_height + offset_from_label, agent_bubble_radius, prim_bubble_radius, primitive_assignment[i], is_split=True, non_human_agent=non_human_agent, fill_color=[human_prim_color, robot_prim_color])

    # Save SVG
    with open('outputs/work_allocation.svg', 'wb') as out_file:
        out_file.write(tostring(svg))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=None, help="")
    parser.add_argument("--csv-file", type=str, default=None, help="")
    args = parser.parse_args()

    run(args)
