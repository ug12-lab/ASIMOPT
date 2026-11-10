# -*- coding: utf-8 -*-
"""
features.py
===========
ASIMOPT - Öznitelik Mühendisliği (Feature Engineering) ve Veri Hazırlama Modülü.
Teklif anında (gün öncesinde) bilinen gecikmeli (lag) değişkenleri üretir;
gelecek bilgisi sızıntısını (data leakage) kesin olarak önler.
"""

import os
import numpy as np
import pandas as pd

VARSAYILAN_YOLLAR = [
    "birlesik_veri.csv",
    os.path.join("cikti", "birlesik_veri.csv"),
    os.path.join("epias_veri.csv", "cikti", "birlesik_veri.csv"),
]


def veri_yolunu_bul(ozel_yol=None):
    """Veri dosyasının yolunu otomatik tespit eder."""
    if ozel_yol and os.path.exists(ozel_yol):
        return ozel_yol
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for p in VARSAYILAN_YOLLAR:
        aday1 = os.path.join(base_dir, p)
        if os.path.exists(aday1):
            return aday1
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "birlesik_veri.csv bulunamadı! Lütfen dosyayı proje klasörüne ekleyin."
    )


def veriyi_yukle(dosya_yolu=None):
    """CSV dosyasını okur ve temel uzlaştırma değişkenlerini kontrol eder."""
    if isinstance(dosya_yolu, pd.DataFrame):
        df = dosya_yolu.copy()
    else:
        yol = veri_yolunu_bul(dosya_yolu)
        df = pd.read_csv(yol)

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    elif "Tarih" in df.columns:
        df["ts"] = pd.to_datetime(df["Tarih"])
    else:
        raise ValueError("Veri setinde 'ts' veya 'Tarih' sütunu bulunamadı.")

    df = df.sort_values("ts").reset_index(drop=True)

    # Gerçekleşmiş uzlaştırma fiyatlarından birim dengesizlik cezalarının hesabı
    if "pi_eksik" not in df.columns:
        df["pi_eksik"] = (df["NDF_neg"] - df["PTF"]).clip(lower=0.0)
    if "pi_fazla" not in df.columns:
        df["pi_fazla"] = (df["PTF"] - df["PDF_poz"]).clip(lower=0.0)

    toplam_ceza = df["pi_eksik"] + df["pi_fazla"]
    if "tau" not in df.columns:
        df["tau"] = np.where(toplam_ceza > 0, df["pi_fazla"] / toplam_ceza, 0.5)
    else:
        # Eksik (pi_eksik + pi_fazla == 0 olan 60 saat) değerleri 0.5 (simetrik) ile doldur
        df["tau"] = df["tau"].fillna(0.5)

    # Gerçekleşmiş tau* değerini [0.05, 0.95] aralığında sınırla (uç değer kararlılığı)
    df["tau_gercek"] = df["tau"].clip(0.05, 0.95)

    return df


def oznitelikleri_uret(df, hedef_sutun="ruzgar"):
    """
    Gün öncesi teklif anında (en az 24 saat öncesinden) bilinen
    nedensel (causal) öznitelikleri üretir.
    """
    data = df.copy()

    # 1. Takvimsel ve Döngüsel (Cyclical) Öznitelikler
    data["saat"] = data["ts"].dt.hour
    data["haftanin_gunu"] = data["ts"].dt.dayofweek
    data["ay"] = data["ts"].dt.month
    data["yilin_gunu"] = data["ts"].dt.dayofyear
    data["hafta_sonu"] = (data["haftanin_gunu"] >= 5).astype(int)

    data["sin_saat"] = np.sin(2 * np.pi * data["saat"] / 24.0)
    data["cos_saat"] = np.cos(2 * np.pi * data["saat"] / 24.0)
    data["sin_ay"] = np.sin(2 * np.pi * data["ay"] / 12.0)
    data["cos_ay"] = np.cos(2 * np.pi * data["ay"] / 12.0)

    # 2. Üretim Gecikmeleri (Teklif anında bilinen: t-24 ve öncesi)
    for lag in [24, 25, 48, 72, 168]:
        data[f"{hedef_sutun}_lag{lag}"] = data[hedef_sutun].shift(lag)

    # 24 saat gecikmeli hareketli istatistikler
    kaydirilmis_uretim = data[hedef_sutun].shift(24)
    data[f"{hedef_sutun}_roll24_mean"] = kaydirilmis_uretim.rolling(24).mean()
    data[f"{hedef_sutun}_roll24_std"] = kaydirilmis_uretim.rolling(24).std()
    data[f"{hedef_sutun}_roll168_mean"] = kaydirilmis_uretim.rolling(168).mean()

    if "uretim_toplam" in data.columns:
        data["uretim_toplam_lag24"] = data["uretim_toplam"].shift(24)
    if "dogalgaz" in data.columns:
        data["dogalgaz_lag24"] = data["dogalgaz"].shift(24)
    if "barajli" in data.columns:
        data["barajli_lag24"] = data["barajli"].shift(24)

    # 3. Sistem Yönü ve Piyasa Gecikmeleri (Önceden tahmin edilen tau_hat modeli için)
    for lag in [24, 48, 72, 168]:
        data[f"tau_lag{lag}"] = data["tau_gercek"].shift(lag)

    kaydirilmis_tau = data["tau_gercek"].shift(24)
    data["tau_roll24_mean"] = kaydirilmis_tau.rolling(24).mean()
    data["tau_roll168_mean"] = kaydirilmis_tau.rolling(168).mean()

    data["PTF_lag24"] = data["PTF"].shift(24)
    data["SMF_lag24"] = data["SMF"].shift(24)
    data["fiyat_farki_lag24"] = data["PTF_lag24"] - data["SMF_lag24"]
    data["pi_eksik_lag24"] = data["pi_eksik"].shift(24)
    data["pi_fazla_lag24"] = data["pi_fazla"].shift(24)

    # İlk 192 saati (lag/rolling kaynaklı NaN satırları) temizle
    data = data.dropna().reset_index(drop=True)
    return data


def oznitelik_listeleri(hedef_sutun="ruzgar"):
    """Üretim modeli ve Tau tahmin modeli için öznitelik listelerini döndürür."""
    uretim_oznitelikleri = [
        "saat", "haftanin_gunu", "ay", "yilin_gunu", "hafta_sonu",
        "sin_saat", "cos_saat", "sin_ay", "cos_ay",
        f"{hedef_sutun}_lag24", f"{hedef_sutun}_lag25",
        f"{hedef_sutun}_lag48", f"{hedef_sutun}_lag72", f"{hedef_sutun}_lag168",
        f"{hedef_sutun}_roll24_mean", f"{hedef_sutun}_roll24_std",
        f"{hedef_sutun}_roll168_mean",
        "uretim_toplam_lag24", "dogalgaz_lag24", "barajli_lag24",
    ]

    tau_oznitelikleri = [
        "saat", "haftanin_gunu", "ay", "yilin_gunu", "hafta_sonu",
        "sin_saat", "cos_saat", "sin_ay", "cos_ay",
        "tau_lag24", "tau_lag48", "tau_lag72", "tau_lag168",
        "tau_roll24_mean", "tau_roll168_mean",
        "PTF_lag24", "SMF_lag24", "fiyat_farki_lag24",
        "pi_eksik_lag24", "pi_fazla_lag24",
        f"{hedef_sutun}_lag24", "uretim_toplam_lag24",
    ]
    return uretim_oznitelikleri, tau_oznitelikleri


def egitim_test_ayir(df, egitim_orani=0.80):
    """Zaman serisi sırasını bozmadan Eğitim (%80) ve Test (%20) olarak ayırır."""
    kesim = int(len(df) * egitim_orani)
    train_df = df.iloc[:kesim].copy().reset_index(drop=True)
    test_df = df.iloc[kesim:].copy().reset_index(drop=True)
    return train_df, test_df
