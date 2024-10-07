# Copyright 2024-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Tuple


def _create_tree_structure(  # noqa: C901
    span_buffer: List[Dict[str, Any]], event_buffer: List[Dict[str, Any]]
) -> List[Tuple]:
    """
    Create a tree structure from the trace buffer using the parent_id and attach events as subnodes.

    Args:
        span_buffer (List[Dict[str, Any]]): The trace buffer to create the tree structure from.
        event_buffer (List[Dict[str, Any]]): The event buffer containing events related to spans.

    Returns:
        List[Tuple]: The formatted tree structure as a list of tuples.
    """
    nodes = []

    node_dict = {item["id_"]: item.copy() for item in span_buffer}

    for node in node_dict.values():
        node["children"] = []

    for node in node_dict.values():
        parent_id = node["parent_id"]
        if parent_id and parent_id in node_dict:
            node_dict[parent_id]["children"].append(node)

    event_dict = {}
    for event in event_buffer:
        span_id = event.get("span_id")
        if span_id not in event_dict:
            event_dict[span_id] = []
        event_dict[span_id].append(event)

    def build_tree(node, depth=0):
        node_name = node["id_"].split(".")[0]
        node_time = node["duration"]

        row = len(nodes)
        nodes.append((row, depth, node_name, node_time))

        span_id = node["id_"]
        if span_id in event_dict:
            for event in event_dict[span_id]:
                event_name = event.get("event_type", "Unknown Event")
                event_row = len(nodes)
                nodes.append((event_row, depth + 1, event_name, ""))

        for child in node.get("children", []):
            build_tree(child, depth + 1)

    root_nodes = [
        node
        for node in node_dict.values()
        if node["parent_id"] is None or node["parent_id"] not in node_dict
    ]
    for root in root_nodes:
        build_tree(root)

    return nodes


def _create_svg(data: List[Tuple]) -> str:
    """
    Create an SVG file from the data.

    Args:
        data (List[Tuple]): The data to create the SVG file from.

    Returns:
        str: The SVG file.
    """
    svg_template = """
<g transform="translate({x}, {y})">
    <rect x=".5" y=".5" width="{width}" height="40" rx="8.49" ry="8.49" style="fill: #24272e; stroke: #afdfe5; stroke-miterlimit: 10;"/>
    <text transform="translate({node_name_indent} {text_centering})" style="fill: {font_color}; font-size: {font_size_node_name}px;">
        <tspan x="0" y="0">{node_name}</tspan>
    </text>
    <text transform="translate({time_indent} {text_centering})" style="fill: #b7d989; font-size: {font_size_time}px; font-style: italic;">
        <tspan x="0" y="0">{node_time}</tspan>
    </text>
</g>"""

    body = "".join(
        svg_template.format(
            x=indent * 30,
            y=row * 45,
            width=40 * 8,  # 40 is the height of the box
            node_name_indent=40 * 0.35,
            text_centering=40 * 0.6341,
            font_size_node_name=40 * 0.4188,
            node_name=node_name,
            time_indent=40 * 6.5,
            font_size_time=40 * 0.4188 - 4,
            node_time=node_time,
            font_color="#cdf1f9" if "event" in node_name.lower() else "#fff",
        )
        for row, indent, node_name, node_time in data
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 {len(data) * 45}">
{body}
</svg>"""
