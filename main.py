import os
from fastapi import FastAPI, Query
from scraper import EpiasScraper, MeteoScraper
from processor import DataProcessor
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="ASIMOPT API",
    description="Asimetrik Kayıp Fonksiyonu Tabanlı Üretim Tahmini ve Dengesizlik Maliyeti Optimizasyonu (ASIMOPT) Veri Sağlayıcısı",
    version="1.0.0"
)

EPIAS_API_KEY = os.getenv("EPIAS_API_KEY", "")

epias_scraper = EpiasScraper(api_key=EPIAS_API_KEY)
meteo_scraper = MeteoScraper()
processor = DataProcessor()

@app.get("/")
def read_root():
    return {"message": "ASIMOPT API Render Üzerinde Başarıyla Çalışıyor!"}

@app.get("/api/v1/dataset")
def get_training_dataset(
    start_date: str = Query(..., description="Başlangıç tarihi (YYYY-MM-DD)", example="2022-01-01"),
    end_date: str = Query(..., description="Bitiş tarihi (YYYY-MM-DD)", example="2022-01-31"),
    lat: float = Query(39.92, description="Santral Enlemi (Varsayılan Ankara)"),
    lon: float = Query(32.85, description="Santral Boylamı (Varsayılan Ankara)")
):
    try:
        epias_market_df = epias_scraper.fetch_ptf_smf(start_date, end_date)
        epias_prod_df = epias_scraper.fetch_kgup_and_production(start_date, end_date)
        meteo_df = meteo_scraper.fetch_weather(lat, lon, start_date, end_date)
        
        epias_df = processor.merge_datasets(epias_market_df, epias_prod_df)
        merged_df = processor.merge_datasets(epias_df, meteo_df)
        processed_df = processor.apply_preprocessing(merged_df)
        
        result = processed_df.to_dict(orient="records")
        
        return {
            "status": "success",
            "row_count": len(result),
            "data": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)