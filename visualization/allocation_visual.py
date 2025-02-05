import sys
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

# Function to add an embedded SVG to the main SVG
def add_image(svg, embedded_svg, x, y, width, height, percentage=-1):
    if percentage == -1:
        original_width = parse_dimension(embedded_svg.attrib.get("width", "1"))
        original_height = parse_dimension(embedded_svg.attrib.get("height", "1"))

        group = SubElement(svg, 'g', transform=f'translate({x},{y}) scale({width / original_width},{height / original_height})')
    else:
        group = SubElement(svg, 'g', transform=f'translate({x},{y}) scale({percentage / 100},{percentage / 100})')
    for element in embedded_svg:
        group.append(element)
    return group

# Function to add multiple bubbles with text horizontally
def add_bubbles(svg, x_start, y, radius, texts_colors, spacing=5, is_human=False):
    total_width = len(texts_colors) * (2 * radius + spacing) - spacing
    x_start_adjusted = x_start - total_width / 2
    fill_color = "blue" if is_human else "red"

    for idx, text_content in enumerate(texts_colors):
        x = x_start_adjusted + idx * (2 * radius + spacing / 2) + radius
        bubble = SubElement(svg, 'circle', cx=str(x), cy=str(y), r=str(radius), fill=fill_color)
        text = SubElement(svg, 'text', x=str(x), y=str(y + radius / 8), fill='white', 
                          style='font-size: {}px; text-anchor: middle; dominant-baseline: middle;'.format(radius))
        text.text = text_content

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
    worker_robot_height = 100
    column_width = 2 * worker_robot_height
    icon_size = worker_robot_height 
    padding = 5
    bubble_radius = 10
    bubble_spacing = 1

    num_columns = sum(1 for _ in task_assignment)

    max_bubbles = max([5 if row and "Human" not in row else 2 for row in task_assignment if row])
    svg_width = num_columns * (column_width + padding)
    svg_height = worker_robot_height + 3 * (icon_size + padding) + 2 * (bubble_radius + bubble_spacing) + max_bubbles * (2 * bubble_radius + bubble_spacing) + 40

    svg = create_svg(svg_width, svg_height)

    # Embed worker and robot SVGs
    worker_svg_data = embed_svg('assets/worker.svg')
    robot_svg_data = embed_svg('assets/robot.svg')

    for i, row in enumerate(task_assignment):
        x_offset = i * (column_width + padding)
        task_name = task_names[i]
        
        # Add allocation label with wrapping
        if task_name is None:
            continue

        add_allocation_label(svg, x_offset + column_width / 2, 15, task_name, column_width)

        if row is None:
            continue

        # Determine allocation
        if row == 'Human':
            add_image(svg, worker_svg_data, x_offset + column_width // 3, 40, worker_robot_height, worker_robot_height)
            add_bubbles(svg, x_offset + column_width / 2, worker_robot_height + 30 - bubble_radius, 
                        bubble_radius, primitive_assignment[i]["Human"], is_human=True)
        elif "Human" not in row:
            add_image(svg, robot_svg_data, x_offset + column_width // 4, 40, worker_robot_height, worker_robot_height)
            add_bubbles(svg, x_offset + column_width / 2, worker_robot_height + 30 - bubble_radius, 
                        bubble_radius, primitive_assignment[i][non_human_agent])
        else:
            add_image(svg, worker_svg_data, x_offset, 40, column_width // 2, worker_robot_height)
            add_bubbles(svg, x_offset + column_width // 4, worker_robot_height + 30 - bubble_radius, 
                        bubble_radius, primitive_assignment[i]["Human"], is_human=True)

            add_image(svg, robot_svg_data, x_offset + column_width // 2, 40, column_width // 2, worker_robot_height)
            add_bubbles(svg, x_offset + 3 * column_width // 4, worker_robot_height + 30 - bubble_radius, 
                        bubble_radius, primitive_assignment[i][non_human_agent])

    # Save SVG
    with open('outputs/work_allocation.svg', 'wb') as out_file:
        out_file.write(tostring(svg))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=None, help="")
    parser.add_argument("--csv-file", type=str, default=None, help="")
    args = parser.parse_args()

    run(args)
