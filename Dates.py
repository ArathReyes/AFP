import datetime as dt
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay, DateOffset

def from_excel_date(excel_date):
    """
    Convert an Excel serial date to a Python datetime object.
    
    Parameters:
    excel_date (float): The Excel serial date.
    
    Returns:
    datetime: The corresponding Python datetime object.
    """
    return dt.datetime(1899, 12, 30) + dt.timedelta(days=excel_date)



def bump_date(start_date, bump_str):
    """
    Bumps a date by US Business Days/Weeks/Months/Years.
    Example inputs: '5d', '-2W', '1M', '10Y'
    """
    # 1. Define the US Business Day logic (skips weekends + federal holidays)
    us_bus = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    
    # 2. Parse the string into value and unit
    # Splitting numeric part from the letter
    import re
    match = re.match(r"([-+]?\d+)([a-zA-Z]+)", bump_str)
    if not match:
        raise ValueError("Invalid bump format. Use '5d', '1M', etc.")
    
    n = int(match.group(1))
    unit = match.group(2).lower()

    # 3. Apply logic based on unit
    dt = pd.to_datetime(start_date)

    if unit == 'd':
        # Add n US Business Days
        result = dt + (n * us_bus)
    elif unit == 'w':
        # Add n Weeks, then snap to nearest business day
        result = dt + pd.DateOffset(weeks=n)
        if not us_bus.is_on_offset(result):
            result = us_bus.rollforward(result)
    elif unit == 'm':
        # Add n Months, then snap to nearest business day
        result = dt + pd.DateOffset(months=n)
        if not us_bus.is_on_offset(result):
            result = us_bus.rollforward(result)
    elif unit == 'y':
        # Add n Years, then snap to nearest business day
        result = dt + pd.DateOffset(years=n)
        if not us_bus.is_on_offset(result):
            result = us_bus.rollforward(result)
    else:
        raise ValueError(f"Unsupported unit: {unit}")

    return result.date()