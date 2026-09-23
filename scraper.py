import requests
import pandas as pd
from datetime import datetime

class EpiasScraper:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://seffaflik.epias.com.tr/transparency/service"
        self.headers = {
            "Accept": "application/json",
        }

    def fetch_ptf_smf(self, start_date: str, end_date: str) -> pd.DataFrame:
        # Örnek/Mock Veri Döndürme (API key bağlayana kadar hata vermemesi için)
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        df = pd.DataFrame({
            'Tarih': dates,
            'PTF': [2000.0] * len(dates),
            'SMF': [1700.0] * len(dates)
        })
        return df

    def fetch_kgup_and_production(self, start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        df = pd.DataFrame({
            'Tarih': dates,
            'KGUP': [100.0] * len(dates),
            'Gerceklesen_Uretim': [95.0] * len(dates)
        })
        return df

class MeteoScraper:
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"

    def fetch_weather(self, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "wind_speed_10m,shortwave_radiation",
            "timezone": "Europe/Istanbul"
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        hourly_data = data.get("hourly", {})
        df = pd.DataFrame({
            'Tarih': pd.to_datetime(hourly_data.get('time')),
            'Ruzgar_Hizi_10m': hourly_data.get('wind_speed_10m'),
            'Gunes_Isinimi': hourly_data.get('shortwave_radiation')
        })
        return df