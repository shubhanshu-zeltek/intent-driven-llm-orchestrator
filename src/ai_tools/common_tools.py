from langchain_core.tools import tool
import numexpr
import pytz
from datetime import datetime


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Use this tool whenever an exact mathematical calculation is required.
    """
    try:
        result = numexpr.evaluate(expression).item()
        return str(result)

    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def current_datetime(region: str = "Asia/Kolkata") -> str:
    """
    Return the current date and time for an IANA timezone.

    Examples:
        Asia/Kolkata
        America/New_York
        Europe/London
    """
    try:
        tz = pytz.timezone(region)
        return datetime.now(tz).strftime(
            "%m/%d/%Y %I:%M %p"
        )

    except pytz.exceptions.UnknownTimeZoneError:
        return (
            f"Error: '{region}' is not a recognized timezone. "
            "Use an IANA timezone such as 'Asia/Kolkata' "
            "or 'America/New_York'."
        )

    except Exception as e:
        return (
            f"Error retrieving datetime for region "
            f"{region!r}: {e}"
        )

def get_common_tools():
    return [calculator, current_datetime]
