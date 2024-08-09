"""
Auxiliary methods for the Argilla Llama Index integration.
"""

from datetime import datetime
from typing import Dict, List

from llama_index.core.callbacks.schema import CBEvent

def _get_time_diff(event_1_time_str: str, event_2_time_str: str) -> float:
    """
    Get the time difference between two events Follows the American format (month, day, year).

    Args:
        event_1_time_str (str): The first event time.
        event_2_time_str (str): The second event time.

    Returns:
        float: The time difference between the two events.
    """
    time_format = "%m/%d/%Y, %H:%M:%S.%f"

    event_1_time = datetime.strptime(event_1_time_str, time_format)
    event_2_time = datetime.strptime(event_2_time_str, time_format)

    return round((event_2_time - event_1_time).total_seconds(), 4)

def _calc_time(events_data: Dict[str, List[CBEvent]], id: str) -> float:
    """
    Calculate the time difference between the start and end of an event using the events_data.

    Args:
        events_data (Dict[str, List[CBEvent]]): The events data, stored in a dictionary.
        id (str): The event id to calculate the time difference between start and finish timestamps.

    Returns:
        float: The time difference between the start and end of the event.
    """

    start_time = events_data[id][0].time
    end_time = events_data[id][1].time
    return _get_time_diff(start_time, end_time)


def _create_svg(data: List) -> str:
    """
    Create an SVG file from the data.

    Args:
        data (List): The data to create the SVG file from.

    Returns:
        str: The SVG file.
    """

    svg_template = """
<g transform="translate({x}, {y})">
    <rect x=".5" y=".5" width="{width}" height="47" rx="8.49" ry="8.49" style="fill: #24272e; stroke: #afdfe5; stroke-miterlimit: 10;"/>
    <text transform="translate({node_name_indent} {text_centering})" style="fill: #fff; font-size: {font_size_node_name}px;">
        <tspan x="0" y="0">{node_name}</tspan>
    </text>
    <text transform="translate({time_indent} {text_centering})" style="fill: #b7d989; font-size: {font_size_time}px; font-style: italic;">
        <tspan x="0" y="0">{node_time}</tspan>
    </text>
</g>"""

    body = "".join(
        svg_template.format(
            x=indent * 40,
            y=row * 54,
            width=47 * 8.65,  # 47 is the height of the box
            node_name_indent=47 * 0.35,
            text_centering=47 * 0.6341,
            font_size_node_name=47 * 0.4188,
            node_name=node_name,
            time_indent=47 * 7.15,
            font_size_time=47 * 0.4188 - 4,
            node_time=node_time,
        )
        for row, indent, node_name, node_time in data
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 {len(data) * 54}">
{body}
</svg>"""
