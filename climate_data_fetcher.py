"""
Climate Data Fetcher
====================
Fetches historical climate data from Open-Meteo API (free, no API key needed).
Data available from 1940 to present.

Perfect for PRISM Engine analysis - just another time series input!
"""

import requests
import pandas as pd
from datetime import datetime, timedelta


def fetch_climate_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: list = None
) -> pd.DataFrame:
    """
    Fetch historical climate data for a specific location.
    
    Parameters:
    -----------
    latitude : float
        Location latitude (e.g., 40.0 for Lafayette, IN)
    longitude : float
        Location longitude (e.g., -86.9 for Lafayette, IN)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    variables : list, optional
        Climate variables to fetch. Defaults to common ones.
        
    Available variables:
        - temperature_2m_mean    (daily mean temp in °C)
        - temperature_2m_max     (daily max temp)
        - temperature_2m_min     (daily min temp)
        - precipitation_sum      (daily precipitation in mm)
        - rain_sum              (daily rain in mm)
        - snowfall_sum          (daily snowfall in cm)
        - wind_speed_10m_max    (max wind speed in km/h)
        - shortwave_radiation_sum (solar radiation in MJ/m²)
        
    Returns:
    --------
    pd.DataFrame with date index and requested variables
    """
    
    # Default variables if none specified
    if variables is None:
        variables = [
            "temperature_2m_mean",
            "temperature_2m_max", 
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max"
        ]
    
    # Build the API URL
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(variables),
        "timezone": "America/New_York"
    }
    
    print(f"Fetching climate data for ({latitude}, {longitude})...")
    print(f"Date range: {start_date} to {end_date}")
    
    # Make the request
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    data = response.json()
    
    # Convert to DataFrame
    daily_data = data.get("daily", {})
    
    if not daily_data:
        raise Exception("No data returned from API")
    
    df = pd.DataFrame(daily_data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    
    print(f"Successfully fetched {len(df)} days of data!")
    
    return df


def fetch_us_cities_climate(start_date: str, end_date: str) -> dict:
    """
    Fetch climate data for major US cities.
    Useful for building a multi-location climate dataset.
    
    Returns dict of {city_name: DataFrame}
    """
    
    cities = {
        "New_York": (40.71, -74.01),
        "Los_Angeles": (34.05, -118.24),
        "Chicago": (41.88, -87.63),
        "Houston": (29.76, -95.37),
        "Phoenix": (33.45, -112.07),
        "Denver": (39.74, -104.99),
        "Miami": (25.76, -80.19),
        "Seattle": (47.61, -122.33),
        "Lafayette_IN": (40.42, -86.87)  # Added for you!
    }
    
    results = {}
    
    for city, (lat, lon) in cities.items():
        print(f"\n--- {city} ---")
        try:
            df = fetch_climate_data(lat, lon, start_date, end_date)
            # Add city prefix to columns
            df.columns = [f"{city}_{col}" for col in df.columns]
            results[city] = df
        except Exception as e:
            print(f"Error fetching {city}: {e}")
    
    return results


def create_climate_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived climate indicators from raw data.
    These could be inputs to PRISM Engine!
    
    Creates:
    - temp_range: Daily temperature swing
    - temp_anomaly: Deviation from rolling mean
    - precip_7d: 7-day precipitation total
    - extreme_heat: Boolean for temps > 35°C
    """
    
    result = df.copy()
    
    # Temperature range (volatility-like measure)
    if "temperature_2m_max" in df.columns and "temperature_2m_min" in df.columns:
        result["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    
    # Temperature anomaly (deviation from 30-day rolling mean)
    if "temperature_2m_mean" in df.columns:
        rolling_mean = df["temperature_2m_mean"].rolling(window=30, min_periods=1).mean()
        result["temp_anomaly"] = df["temperature_2m_mean"] - rolling_mean
    
    # 7-day precipitation accumulation
    if "precipitation_sum" in df.columns:
        result["precip_7d"] = df["precipitation_sum"].rolling(window=7, min_periods=1).sum()
    
    # Extreme heat flag
    if "temperature_2m_max" in df.columns:
        result["extreme_heat"] = (df["temperature_2m_max"] > 35).astype(int)
    
    return result


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    
    # Example 1: Fetch data for a single location
    # Using Lafayette, IN coordinates
    print("=" * 60)
    print("EXAMPLE 1: Single Location Fetch")
    print("=" * 60)
    
    df = fetch_climate_data(
        latitude=40.42,
        longitude=-86.87,
        start_date="2020-01-01",
        end_date="2024-12-31"
    )
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    # Example 2: Create derived indicators
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Derived Climate Indicators")
    print("=" * 60)
    
    df_with_indicators = create_climate_index(df)
    print("\nNew columns added:")
    print([col for col in df_with_indicators.columns if col not in df.columns])
    print(df_with_indicators[["temp_range", "temp_anomaly", "precip_7d"]].tail(10))
    
    # Example 3: Save to CSV for PRISM Engine
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Saving Data")
    print("=" * 60)
    
    output_file = "/home/claude/lafayette_climate_2020_2024.csv"
    df_with_indicators.to_csv(output_file)
    print(f"Data saved to: {output_file}")
    
    # Quick correlation preview (useful for PRISM!)
    print("\n" + "=" * 60)
    print("CORRELATION MATRIX (Preview for PRISM)")
    print("=" * 60)
    numeric_cols = df_with_indicators.select_dtypes(include="number").columns
    print(df_with_indicators[numeric_cols].corr().round(2))
