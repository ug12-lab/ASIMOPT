import pandas as pd

class DataProcessor:
    def __init__(self):
        pass

    def merge_datasets(self, epias_df: pd.DataFrame, meteo_df: pd.DataFrame) -> pd.DataFrame:
        epias_df['Tarih'] = pd.to_datetime(epias_df['Tarih']).dt.tz_localize(None)
        meteo_df['Tarih'] = pd.to_datetime(meteo_df['Tarih']).dt.tz_localize(None)
        
        merged_df = pd.merge(epias_df, meteo_df, on='Tarih', how='outer')
        merged_df = merged_df.sort_values(by='Tarih').reset_index(drop=True)
        return merged_df

    def apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df.copy()

        numeric_cols = processed_df.select_dtypes(include=['number']).columns
        processed_df[numeric_cols] = processed_df[numeric_cols].interpolate(method='linear', limit_direction='both')

        processed_df['Saat'] = processed_df['Tarih'].dt.hour
        processed_df['Haftanin_Gunu'] = processed_df['Tarih'].dt.dayofweek
        processed_df['Yilin_Gunu'] = processed_df['Tarih'].dt.dayofyear
        processed_df['Saat_Gun_Etkilesimi'] = processed_df['Saat'] * processed_df['Haftanin_Gunu']

        return processed_df