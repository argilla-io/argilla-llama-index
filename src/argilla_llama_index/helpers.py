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
