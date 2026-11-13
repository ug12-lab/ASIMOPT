# -*- coding: utf-8 -*-
"""
features.py
===========
ASIMOPT - Öznitelik Mühendisliği (Feature Engineering) ve Veri Hazırlama Modülü.

Danışman geri bildirimi doğrultusunda:
- 1 Ocak 2026 tarihinin 24 saati (yeni piyasa mekanizması başlangıcı) çıkarılarak
  veri seti 2022-01-01 00:00 -> 2025-12-31 23:00 (35.064 saat) ile sınırlandırılmıştır.
- tau* kırpması her yerde tutarlı biçimde [TAU_MIN, TAU_MAX] = [0.05, 0.95] olarak uygulanır.
- Varsayılan ana sonuç en az 48 saat gecikme (min_gecikme=48) ile teklif anı veri sızıntısını önler;
  24 saat gecikmeli sürüm (min_gecikme=24) ek karşılaştırma tablosu için desteklenir.
"""

import os
import numpy as np
import pandas as pd

TAU_MIN = 0.05
TAU_MAX = 0.95
BITIS_TARIHI = pd.Timestamp("2025-12-31 23:00:00")

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


def veriyi_yukle(dosya_yolu=None, bitis_tarihi=BITIS_TARIHI):
    """
    CSV dosyasını okur, 1 Ocak 2026 verisini hariç tutar ve
    gerçekleşmiş uzlaştırma fiyatlarından birim dengesizlik maliyetleri ile tau* hesaplar.
    """
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

    # 1 Ocak 2026 (yeni piyasa mekanizması dönemi) saatlerini kapsam dışı bırak
    if bitis_tarihi is not None:
        df = df[df["ts"] <= pd.Timestamp(bitis_tarihi)].copy().reset_index(drop=True)

    # Gerçekleşmiş uzlaştırma fiyatlarından birim dengesizlik cezalarının hesabı
    if "pi_eksik" not in df.columns:
        df["pi_eksik"] = (df["NDF_neg"] - df["PTF"]).clip(lower=0.0)
    if "pi_fazla" not in df.columns:
        df["pi_fazla"] = (df["PTF"] - df["PDF_poz"]).clip(lower=0.0)

    toplam_ceza = df["pi_eksik"] + df["pi_fazla"]
    if "tau" not in df.columns:
        df["tau"] = np.where(toplam_ceza > 0, df["pi_fazla"] / toplam_ceza, 0.5)
    else:
        # Dengede (pi_eksik + pi_fazla == 0) saatlerde simetrik 0.5 değeri atanır
        df["tau"] = df["tau"].fillna(0.5)

    # Tüm modüllerde tutarlı [0.05, 0.95] kırpma aralığı
    df["tau_gercek"] = df["tau"].clip(TAU_MIN, TAU_MAX)

    return df


def _gecikme_setleri(min_gecikme=48):
    """Minimum gecikme süresine (48 saat veya 24 saat) göre lag listelerini döndürür."""
    if min_gecikme not in (24, 48):
        raise ValueError("min_gecikme yalnızca 48 (ana sonuç) veya 24 (ek sonuç) olabilir.")
    if min_gecikme == 48:
        uretim_lags = [48, 49, 72, 96, 168]
        tau_lags = [48, 72, 96, 168]
    else:
        uretim_lags = [24, 25, 48, 72, 168]
        tau_lags = [24, 48, 72, 168]
    return uretim_lags, tau_lags


def oznitelikleri_uret(df, hedef_sutun="ruzgar", min_gecikme=48):
    """
    Gün öncesi teklif anında bilinen nedensel (causal) öznitelikleri üretir.
    min_gecikme=48 (ana sonuç): t-48 ve öncesi (SMF, pi, tau ve üretim yayımlanma gecikmesi güvenli).
    min_gecikme=24 (ek sonuç): t-24 ve öncesi.
    """
    data = df.copy()
    uretim_lags, tau_lags = _gecikme_setleri(min_gecikme)

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

    # 2. Üretim Gecikmeleri
    for lag in uretim_lags:
        data[f"{hedef_sutun}_lag{lag}"] = data[hedef_sutun].shift(lag)

    kaydirilmis_uretim = data[hedef_sutun].shift(min_gecikme)
    data[f"{hedef_sutun}_roll24_mean"] = kaydirilmis_uretim.rolling(24).mean()
    data[f"{hedef_sutun}_roll24_std"] = kaydirilmis_uretim.rolling(24).std()
    data[f"{hedef_sutun}_roll168_mean"] = kaydirilmis_uretim.rolling(168).mean()

    if "uretim_toplam" in data.columns and hedef_sutun != "uretim_toplam":
        data[f"uretim_toplam_lag{min_gecikme}"] = data["uretim_toplam"].shift(min_gecikme)
    if "dogalgaz" in data.columns and hedef_sutun != "dogalgaz":
        data[f"dogalgaz_lag{min_gecikme}"] = data["dogalgaz"].shift(min_gecikme)
    if "barajli" in data.columns and hedef_sutun != "barajli":
        data[f"barajli_lag{min_gecikme}"] = data["barajli"].shift(min_gecikme)

    # 3. Sistem Yönü ve Piyasa Gecikmeleri
    for lag in tau_lags:
        data[f"tau_lag{lag}"] = data["tau_gercek"].shift(lag)

    kaydirilmis_tau = data["tau_gercek"].shift(min_gecikme)
    data["tau_roll24_mean"] = kaydirilmis_tau.rolling(24).mean()
    data["tau_roll168_mean"] = kaydirilmis_tau.rolling(168).mean()

    data[f"PTF_lag{min_gecikme}"] = data["PTF"].shift(min_gecikme)
    data[f"SMF_lag{min_gecikme}"] = data["SMF"].shift(min_gecikme)
    data[f"fiyat_farki_lag{min_gecikme}"] = (
        data[f"PTF_lag{min_gecikme}"] - data[f"SMF_lag{min_gecikme}"]
    )
    data[f"pi_eksik_lag{min_gecikme}"] = data["pi_eksik"].shift(min_gecikme)
    data[f"pi_fazla_lag{min_gecikme}"] = data["pi_fazla"].shift(min_gecikme)

    # Lag ve hareketli pencere kaynaklı başlangıç NaN satırlarını temizle
    data = data.dropna().reset_index(drop=True)
    return data


def oznitelik_listeleri(hedef_sutun="ruzgar", min_gecikme=48):
    """Üretim tahmin modeli ve Tau/Yön tahmin modeli için öznitelik listelerini döndürür."""
    uretim_lags, tau_lags = _gecikme_setleri(min_gecikme)

    uretim_adaylari = (
        ["saat", "haftanin_gunu", "ay", "yilin_gunu", "hafta_sonu",
         "sin_saat", "cos_saat", "sin_ay", "cos_ay"]
        + [f"{hedef_sutun}_lag{lag}" for lag in uretim_lags]
        + [f"{hedef_sutun}_roll24_mean", f"{hedef_sutun}_roll24_std",
           f"{hedef_sutun}_roll168_mean",
           f"uretim_toplam_lag{min_gecikme}",
           f"dogalgaz_lag{min_gecikme}",
           f"barajli_lag{min_gecikme}"]
    )

    tau_adaylari = (
        ["saat", "haftanin_gunu", "ay", "yilin_gunu", "hafta_sonu",
         "sin_saat", "cos_saat", "sin_ay", "cos_ay"]
        + [f"tau_lag{lag}" for lag in tau_lags]
        + ["tau_roll24_mean", "tau_roll168_mean",
           f"PTF_lag{min_gecikme}", f"SMF_lag{min_gecikme}",
           f"fiyat_farki_lag{min_gecikme}",
           f"pi_eksik_lag{min_gecikme}", f"pi_fazla_lag{min_gecikme}",
           f"{hedef_sutun}_lag{min_gecikme}", f"uretim_toplam_lag{min_gecikme}"]
    )

    uretim_oznitelikleri = list(dict.fromkeys(uretim_adaylari))
    tau_oznitelikleri = list(dict.fromkeys(tau_adaylari))
    return uretim_oznitelikleri, tau_oznitelikleri


def egitim_test_ayir(df, egitim_orani=0.80):
    """
    Zaman serisi sırasını bozmadan Eğitim (%80) ve Test (%20) olarak ayırır.
    Hiperparametre ayarlaması (tuning) yapılmadığı ve parametreler sabit tutulduğu için
    ayrı bir doğrulama kümesi (validation set) kullanılmamıştır.
    """
    kesim = int(len(df) * egitim_orani)
    train_df = df.iloc[:kesim].copy().reset_index(drop=True)
    test_df = df.iloc[kesim:].copy().reset_index(drop=True)
    return train_df, test_df
