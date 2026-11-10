# -*- coding: utf-8 -*-
"""
tables.py
=========
ASIMOPT - Rapor Tabloları, Dengesizlik Maliyeti Hesabı ve Diebold-Mariano (DM) Testi.

Bu dosya çalıştırıldığında (`python tables.py`):
1. Veri setini yükler ve nedensel öznitelikleri üretir (features.py).
2. Sabit tau (0.3, 0.5, 0.7), Gerçekçi ASIMOPT (sistem yönünü önceden tahmin ederek
   tau_hat belirleyen model) ve Üst Sınır / Oracle ASIMOPT (gerçekleşmiş uzlaştırma
   fiyatlarından hesaplanan tau*) modellerini eğitir (model.py).
3. Gerçekleşmiş uzlaştırma fiyatlarından hesaplanan dengesizlik maliyetlerini ve
   Diebold-Mariano (DM) istatistiksel anlamlılık testi sonuçlarını tablo olarak üretir.
"""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import veriyi_yukle, oznitelikleri_uret, egitim_test_ayir
from model import modelleri_egit_ve_tahmin_et, pinball_loss


def saatlik_dengesizlik_maliyeti(y_true, y_pred, pi_fazla, pi_eksik):
    """
    Gerçekleşmiş uzlaştırma fiyatlarından hesaplanan saatlik dengesizlik maliyeti (TL).
    - Fazla üretim (y_true > y_pred): (y_true - y_pred) * pi_fazla
    - Eksik üretim (y_true < y_pred): (y_pred - y_true) * pi_eksik
    """
    fazla_uretim = np.maximum(y_true - y_pred, 0.0)
    eksik_uretim = np.maximum(y_pred - y_true, 0.0)
    return fazla_uretim * pi_fazla + eksik_uretim * pi_eksik


def diebold_mariano_testi(maliyet_referans, maliyet_aday, gecikme_h=24):
    """
    Diebold-Mariano (1995) İstatistiksel Anlamlılık Testi (Newey-West HAC varyans düzeltmeli).
    d_t = maliyet_referans_t - maliyet_aday_t
    Pozitif DM istatistiği, aday modelin referans modelden daha düşük maliyet ürettiğini gösterir.
    """
    d = np.asarray(maliyet_referans, dtype=float) - np.asarray(maliyet_aday, dtype=float)
    T = len(d)
    d_bar = np.mean(d)
    d_merkez = d - d_bar

    gamma_0 = np.sum(d_merkez ** 2) / T
    uzun_donem_var = gamma_0
    for k in range(1, gecikme_h + 1):
        agirlik = 1.0 - (k / (gecikme_h + 1.0))
        gamma_k = np.sum(d_merkez[k:] * d_merkez[:-k]) / T
        uzun_donem_var += 2.0 * agirlik * gamma_k

    if uzun_donem_var <= 1e-12:
        return 0.0, 1.0, float(d_bar)

    dm_istatistigi = d_bar / math.sqrt(uzun_donem_var / T)
    p_degeri = math.erfc(abs(dm_istatistigi) / math.sqrt(2.0))
    return float(dm_istatistigi), float(p_degeri), float(d_bar)


def p_degeri_formatla(p):
    """Akademik raporlama için p-değerini biçimlendirir."""
    if p < 0.0001:
        return "< 0.0001"
    return f"{p:.4f}"


def tum_analizi_calistir(dosya_yolu=None, hedef_sutun="ruzgar"):
    """
    Tüm veri hazırlama, model eğitimi, maliyet hesabı ve DM testlerini çalıştırıp
    sonuç tablolarını döndürür.
    """
    ham_df = veriyi_yukle(dosya_yolu)
    islenmis_df = oznitelikleri_uret(ham_df, hedef_sutun=hedef_sutun)
    train_df, test_df = egitim_test_ayir(islenmis_df, egitim_orani=0.80)

    tahminler, ek_bilgiler = modelleri_egit_ve_tahmin_et(
        train_df, test_df, hedef_sutun=hedef_sutun
    )

    y_test = test_df[hedef_sutun].values
    pi_fazla = test_df["pi_fazla"].values
    pi_eksik = test_df["pi_eksik"].values
    tau_gercek = test_df["tau_gercek"].values
    toplam_uretim_mwh = np.sum(y_test)

    ref_isim = "Sabit tau = 0.50 (Simetrik MAE)"
    ref_maliyet_serisi = saatlik_dengesizlik_maliyeti(
        y_test, tahminler[ref_isim], pi_fazla, pi_eksik
    )
    ref_toplam_maliyet = np.sum(ref_maliyet_serisi)

    tablo1_satirlar = []
    tablo2_satirlar = []
    tablo3_satirlar = []
    maliyet_serileri = {}

    for model_adi, y_pred in tahminler.items():
        maliyet_serisi = saatlik_dengesizlik_maliyeti(y_test, y_pred, pi_fazla, pi_eksik)
        maliyet_serileri[model_adi] = maliyet_serisi

        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        ort_pinball = float(np.mean(pinball_loss(y_test, y_pred, tau_gercek)))
        toplam_maliyet = float(np.sum(maliyet_serisi))
        birim_maliyet = toplam_maliyet / toplam_uretim_mwh
        tasarruf_tl = ref_toplam_maliyet - toplam_maliyet
        tasarruf_yuzde = (tasarruf_tl / ref_toplam_maliyet) * 100.0

        tablo1_satirlar.append({
            "Model": model_adi,
            "MAE (MWh)": round(mae, 2),
            "RMSE (MWh)": round(rmse, 2),
            "Dinamik Pinball Kaybı": round(ort_pinball, 2),
            "Toplam Dengesizlik Maliyeti (TL)": round(toplam_maliyet, 0),
            "Birim Maliyet (TL/MWh)": round(birim_maliyet, 2),
            "Simetrik Modele Göre Tasarruf (%)": round(tasarruf_yuzde, 2),
        })

        if model_adi != ref_isim:
            dm_stat, p_val, saatlik_fark = diebold_mariano_testi(
                ref_maliyet_serisi, maliyet_serisi, gecikme_h=24
            )
            tablo2_satirlar.append({
                "Karşılaştırılan Model (vs. Sabit tau=0.50)": model_adi,
                "Saatlik Ort. Maliyet Farkı (TL/saat)": round(saatlik_fark, 2),
                "DM Test İstatistiği": round(dm_stat, 3),
                "p-Değeri": p_degeri_formatla(p_val),
                "p-Değeri (Sayısal)": p_val,
                "İstatistiksel Yorum": (
                    "Anlamlı Tasarruf (p < 0.05)"
                    if (p_val < 0.05 and dm_stat > 0)
                    else ("Anlamlı Daha Yüksek Maliyet" if (p_val < 0.05 and dm_stat < 0) else "Fark Anlamsız")
                ),
            })

        if "yon" in test_df.columns:
            fazla_mask = test_df["yon"] == "Enerji Fazlası"
            acik_mask = test_df["yon"] == "Enerji Açığı"
            tablo3_satirlar.append({
                "Model": model_adi,
                "Enerji Fazlası Saatlik Ort. Maliyet (TL)": round(float(np.mean(maliyet_serisi[fazla_mask])), 2),
                "Enerji Açığı Saatlik Ort. Maliyet (TL)": round(float(np.mean(maliyet_serisi[acik_mask])), 2),
                "Genel Saatlik Ort. Maliyet (TL)": round(float(np.mean(maliyet_serisi)), 2),
            })

    tablo1_df = pd.DataFrame(tablo1_satirlar)
    tablo2_df = pd.DataFrame(tablo2_satirlar)
    tablo3_df = pd.DataFrame(tablo3_satirlar)

    return {
        "tablo1_performans": tablo1_df,
        "tablo2_dm_testi": tablo2_df,
        "tablo3_sistem_yonu": tablo3_df,
        "test_df": test_df,
        "tahminler": tahminler,
        "maliyet_serileri": maliyet_serileri,
        "ek_bilgiler": ek_bilgiler,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("ASIMOPT - MODEL DEĞERLENDİRME VE DIEBOLD-MARIANO TEST SONUÇLARI")
    print("=" * 80)
    sonuclar = tum_analizi_calistir()

    ek = sonuclar["ek_bilgiler"]
    print(f"\n[BİLGİ] Teklif Anında Sistem Yönü Tahmin Doğruluğu : %{ek['yon_tahmin_dogrulugu'] * 100:.2f}")
    print(f"[BİLGİ] Eğitim Seti Ortalama tau (Enerji Fazlası)  : {ek['tau_ort_fazla']:.4f}")
    print(f"[BİLGİ] Eğitim Seti Ortalama tau (Enerji Açığı)    : {ek['tau_ort_acik']:.4f}")

    t1 = sonuclar["tablo1_performans"]
    t2 = sonuclar["tablo2_dm_testi"]
    t3 = sonuclar["tablo3_sistem_yonu"]

    print("\n[TABLO 1] Tüm Modellerin Performans ve Dengesizlik Maliyeti Karşılaştırması:")
    print(t1.to_string(index=False))

    print("\n[TABLO 2] Diebold-Mariano (DM) İstatistiksel Anlamlılık Testi Sonuçları:")
    print(t2.drop(columns=["p-Değeri (Sayısal)"]).to_string(index=False))

    if not t3.empty:
        print("\n[TABLO 3] Sistem Yönüne Göre Saatlik Ortalama Dengesizlik Maliyetleri:")
        print(t3.to_string(index=False))

    t1.to_csv("tablo1_model_karsilastirma.csv", index=False, encoding="utf-8-sig")
    t2.to_csv("tablo2_dm_testi.csv", index=False, encoding="utf-8-sig")
    if not t3.empty:
        t3.to_csv("tablo3_sistem_yonu.csv", index=False, encoding="utf-8-sig")
    print("\nTablolar CSV olarak kaydedildi: tablo1_model_karsilastirma.csv, tablo2_dm_testi.csv, tablo3_sistem_yonu.csv")
