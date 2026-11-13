# -*- coding: utf-8 -*-
"""
main.py
=======
ASIMOPT - Render.com FastAPI Sunucusu.
Eski sahte veri (scraper.py) yerine doğrudan gerçek birlesik_veri.csv
ve tables.py sonuçlarını sunar.
"""

import os
from fastapi import FastAPI
from features import veriyi_yukle
from tables import tum_analizi_calistir

app = FastAPI(
    title="ASIMOPT Akademik API",
    description="Gerçekleşmiş EPİAŞ Uzlaştırma Verileri ve Asimetrik Kayıp Optimizasyon Sonuçları",
    version="2.0.0",
)


@app.get("/")
def read_root():
    return {
        "status": "success",
        "message": "ASIMOPT API (v2.0) Gerçek EPİAŞ Veri Seti ile Çalışıyor!",
        "endpoints": ["/api/v1/summary", "/api/v1/tables"],
    }


@app.get("/api/v1/summary")
def get_summary():
    df = veriyi_yukle()
    return {
        "status": "success",
        "toplam_satir": len(df),
        "baslangic": str(df["ts"].min()),
        "bitis": str(df["ts"].max()),
        "ortalama_ptf": round(float(df["PTF"].mean()), 2),
        "ortalama_smf": round(float(df["SMF"].mean()), 2),
        "ortalama_tau": round(float(df["tau_gercek"].mean()), 4),
    }


@app.get("/api/v1/tables")
def get_tables(hedef: str = "ruzgar", min_gecikme: int = 48):
    sonuclar = tum_analizi_calistir(hedef_sutun=hedef, min_gecikme=min_gecikme)
    return {
        "status": "success",
        "hedef_sutun": sonuclar["hedef_sutun"],
        "min_gecikme": sonuclar["min_gecikme"],
        "basabas_dogruluk": sonuclar["basabas_dogruluk"],
        "yon_tahmin_dogrulugu": round(sonuclar["ek_bilgiler"]["yon_tahmin_dogrulugu"] * 100.0, 2),
        "cogunluk_sinifi_dogrulugu": round(sonuclar["ek_bilgiler"]["cogunluk_sinifi_dogrulugu"] * 100.0, 2),
        "tablo1_performans": sonuclar["tablo1_performans"].to_dict(orient="records"),
        "tablo2_dm_testi": sonuclar["tablo2_dm_testi"].to_dict(orient="records"),
        "tablo3_sistem_yonu": sonuclar["tablo3_sistem_yonu"].to_dict(orient="records"),
        "tablo4_karisiklik": sonuclar["tablo4_karisiklik"].reset_index().to_dict(orient="records"),
        "tablo5_tasarruf_dogruluk": sonuclar["tablo5_tasarruf_dogruluk"].to_dict(orient="records"),
        "tablo6_duyarlilik": sonuclar["tablo6_duyarlilik"].to_dict(orient="records"),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
