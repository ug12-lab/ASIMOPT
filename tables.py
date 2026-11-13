# -*- coding: utf-8 -*-
"""
tables.py
=========
ASIMOPT - Rapor Tabloları, Dengesizlik Maliyeti, Diebold-Mariano (DM) Testi,
Tasarruf-Doğruluk Eğrisi ve Duyarlılık Analizi (±%10, ±%20).

Çalıştırma:
    python tables.py
Ana sonuç olarak 48 saat gecikmeli (min_gecikme=48) Rüzgar ve Güneş tablolarını,
ek olarak 24 saat gecikmeli (min_gecikme=24) Rüzgar karşılaştırmasını üretir.
"""

import os
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import veriyi_yukle, oznitelikleri_uret, egitim_test_ayir, TAU_MIN, TAU_MAX
from model import modelleri_egit_ve_tahmin_et, pinball_loss, dinamik_kuantil_sec


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


def tasarruf_dogruluk_egrisi_hesapla(
    y_test,
    pi_fazla,
    pi_eksik,
    ref_toplam_maliyet,
    ek_bilgiler,
    dogruluk_seviyeleri=None,
    tekrar_sayisi=10,
):
    """
    Danışman isteği (Bölüm 3): Gerçekleşmiş sistem yönünü yapay olarak bozarak
    hedeflenen doğruluk seviyelerinde (%50..%100) tau_hat'ı yeniden üretir ve
    maliyet tasarrufunu hesaplar.
    """
    if dogruluk_seviyeleri is None:
        dogruluk_seviyeleri = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    gercek_yon = ek_bilgiler["yon_test_etiket"]
    tau_ort_fazla = ek_bilgiler["tau_ort_fazla"]
    tau_ort_acik = ek_bilgiler["tau_ort_acik"]
    kuantil_matrisi = ek_bilgiler["kuantil_tahminleri"]
    kuantil_seviyeleri = ek_bilgiler["kuantil_seviyeleri"]
    n_ornek = len(gercek_yon)

    satirlar = []
    for p in dogruluk_seviyeleri:
        tasarruf_tekrar = []
        maliyet_tekrar = []
        gerceklesen_dogruluk_tekrar = []

        n_tekrar = 1 if p >= 0.9999 else tekrar_sayisi
        for rep in range(n_tekrar):
            rng = np.random.default_rng(42 + int(round(p * 1000)) * 100 + rep)
            n_bozulacak = int(round((1.0 - p) * n_ornek))
            yon_bozuk = gercek_yon.copy()
            if n_bozulacak > 0:
                secilen_idx = rng.choice(n_ornek, size=n_bozulacak, replace=False)
                yon_bozuk[secilen_idx] = 1 - yon_bozuk[secilen_idx]

            gerceklesen_dogruluk_tekrar.append(float(np.mean(yon_bozuk == gercek_yon)) * 100.0)

            # Eğitim setinden öğrenilen sınıf ortalamalarıyla tau_hat yeniden üretimi
            tau_yapay = np.where(yon_bozuk == 1, tau_ort_fazla, tau_ort_acik)
            tau_yapay = np.clip(tau_yapay, TAU_MIN, TAU_MAX)

            y_pred_yapay = dinamik_kuantil_sec(
                kuantil_matrisi, kuantil_seviyeleri, tau_yapay
            )
            maliyet_yapay = float(
                np.sum(saatlik_dengesizlik_maliyeti(y_test, y_pred_yapay, pi_fazla, pi_eksik))
            )
            tasarruf_yuzde = ((ref_toplam_maliyet - maliyet_yapay) / ref_toplam_maliyet) * 100.0

            maliyet_tekrar.append(maliyet_yapay)
            tasarruf_tekrar.append(tasarruf_yuzde)

        satirlar.append({
            "Hedef Yön Doğruluğu (%)": round(p * 100.0, 1),
            "Gerçekleşen Yön Doğruluğu (%)": round(float(np.mean(gerceklesen_dogruluk_tekrar)), 2),
            "Ortalama Tasarruf (%)": round(float(np.mean(tasarruf_tekrar)), 2),
            "Tasarruf Std (%)": round(float(np.std(tasarruf_tekrar)), 2),
            "Ortalama Maliyet (TL)": round(float(np.mean(maliyet_tekrar)), 0),
        })

    egri_df = pd.DataFrame(satirlar)

    # Başabaş (Tasarruf = %0) doğruluk eşiğinin doğrusal enterpolasyonla hesabı
    x_vals = egri_df["Hedef Yön Doğruluğu (%)"].values
    y_vals = egri_df["Ortalama Tasarruf (%)"].values
    basabas_dogruluk = None
    for i in range(len(y_vals) - 1):
        if y_vals[i] <= 0.0 <= y_vals[i + 1] or y_vals[i] >= 0.0 >= y_vals[i + 1]:
            if abs(y_vals[i + 1] - y_vals[i]) > 1e-9:
                oran = (0.0 - y_vals[i]) / (y_vals[i + 1] - y_vals[i])
                basabas_dogruluk = float(x_vals[i] + oran * (x_vals[i + 1] - x_vals[i]))
                break

    return egri_df, basabas_dogruluk


def duyarlilik_analizi_hesapla(y_test, tahminler, pi_fazla, pi_eksik, ek_bilgiler):
    """
    Dengesizlik cezaları (pi_eksik ve pi_fazla) ile tahmin edilen tau_hat parametresinin
    ±%10 ve ±%20 değişimlerine karşı modellerin maliyet ve tasarruf duyarlılığını ölçer.
    """
    ref_isim = "Sabit tau = 0.50 (Simetrik MAE)"
    gercekci_isim = "Gerçekçi Kuantil Seçimi (Tahmin tau_hat)"
    oracle_isim = "Üst Sınır / Oracle (Gerçekleşmiş tau*)"
    sabit07_isim = "Sabit tau = 0.70 (Kuantil)"

    degisimler = [-0.20, -0.10, 0.0, 0.10, 0.20]
    satirlar = []

    # 1. Eksik üretim cezası (pi_eksik) ±%10, ±%20 değişimi
    for d in degisimler:
        yeni_pi_eksik = pi_eksik * (1.0 + d)
        m_ref = float(np.sum(saatlik_dengesizlik_maliyeti(y_test, tahminler[ref_isim], pi_fazla, yeni_pi_eksik)))
        m_07 = float(np.sum(saatlik_dengesizlik_maliyeti(y_test, tahminler[sabit07_isim], pi_fazla, yeni_pi_eksik)))
        m_gercekci = float(np.sum(saatlik_dengesizlik_maliyeti(y_test, tahminler[gercekci_isim], pi_fazla, yeni_pi_eksik)))
        m_oracle = float(np.sum(saatlik_dengesizlik_maliyeti(y_test, tahminler[oracle_isim], pi_fazla, yeni_pi_eksik)))

        satirlar.append({
            "Parametre": "pi_eksik (Eksik Üretim Cezası)",
            "Değişim (%)": f"{d * 100:+.0f}%",
            "Degisim_Orani": d,
            "Referans Maliyet tau=0.50 (TL)": round(m_ref, 0),
            "Sabit tau=0.70 Tasarruf (%)": round(((m_ref - m_07) / m_ref) * 100.0, 2),
            "Gerçekçi (tau_hat) Tasarruf (%)": round(((m_ref - m_gercekci) / m_ref) * 100.0, 2),
            "Üst Sınır (tau*) Tasarruf (%)": round(((m_ref - m_oracle) / m_ref) * 100.0, 2),
        })

    # 2. Fazla üretim fırsat maliyeti (pi_fazla) ±%10, ±%20 değişimi
    for d in degisimler:
        yeni_pi_fazla = pi_fazla * (1.0 + d)
        m_ref = float(np.sum(saatlik_dengesizlik_maliyeti(y_test, tahminler[ref_isim], yeni_pi_fazla, pi_eksik)))
        m_07 = float(np.sum(saatlik_dengesizlik_maliyeti(y_test, tahminler[sabit07_isim], yeni_pi_fazla, pi_eksik)))
        m_gercekci = float(np.sum(saatlik_dengesizlik_maliyeti(y_test, tahminler[gercekci_isim], yeni_pi_fazla, pi_eksik)))
        m_oracle = float(np.sum(saatlik_dengesizlik_maliyeti(y_test, tahminler[oracle_isim], yeni_pi_fazla, pi_eksik)))

        satirlar.append({
            "Parametre": "pi_fazla (Fazla Üretim Maliyeti)",
            "Değişim (%)": f"{d * 100:+.0f}%",
            "Degisim_Orani": d,
            "Referans Maliyet tau=0.50 (TL)": round(m_ref, 0),
            "Sabit tau=0.70 Tasarruf (%)": round(((m_ref - m_07) / m_ref) * 100.0, 2),
            "Gerçekçi (tau_hat) Tasarruf (%)": round(((m_ref - m_gercekci) / m_ref) * 100.0, 2),
            "Üst Sınır (tau*) Tasarruf (%)": round(((m_ref - m_oracle) / m_ref) * 100.0, 2),
        })

    # 3. Tahmin edilen tau_hat ±%10, ±%20 kayması
    m_ref_baz = float(np.sum(saatlik_dengesizlik_maliyeti(y_test, tahminler[ref_isim], pi_fazla, pi_eksik)))
    for d in degisimler:
        kaymis_tau_hat = np.clip(ek_bilgiler["tau_hat_test"] * (1.0 + d), TAU_MIN, TAU_MAX)
        y_pred_kaymis = dinamik_kuantil_sec(
            ek_bilgiler["kuantil_tahminleri"], ek_bilgiler["kuantil_seviyeleri"], kaymis_tau_hat
        )
        m_kaymis = float(np.sum(saatlik_dengesizlik_maliyeti(y_test, y_pred_kaymis, pi_fazla, pi_eksik)))

        kaymis_tau_star = np.clip(ek_bilgiler["tau_gercek_test"] * (1.0 + d), TAU_MIN, TAU_MAX)
        y_pred_oracle_kaymis = dinamik_kuantil_sec(
            ek_bilgiler["kuantil_tahminleri"], ek_bilgiler["kuantil_seviyeleri"], kaymis_tau_star
        )
        m_oracle_kaymis = float(np.sum(saatlik_dengesizlik_maliyeti(y_test, y_pred_oracle_kaymis, pi_fazla, pi_eksik)))

        satirlar.append({
            "Parametre": "tau Parametresi Sapması",
            "Değişim (%)": f"{d * 100:+.0f}%",
            "Degisim_Orani": d,
            "Referans Maliyet tau=0.50 (TL)": round(m_ref_baz, 0),
            "Sabit tau=0.70 Tasarruf (%)": round(
                ((m_ref_baz - float(np.sum(saatlik_dengesizlik_maliyeti(y_test, tahminler[sabit07_isim], pi_fazla, pi_eksik)))) / m_ref_baz) * 100.0,
                2,
            ),
            "Gerçekçi (tau_hat) Tasarruf (%)": round(((m_ref_baz - m_kaymis) / m_ref_baz) * 100.0, 2),
            "Üst Sınır (tau*) Tasarruf (%)": round(((m_ref_baz - m_oracle_kaymis) / m_ref_baz) * 100.0, 2),
        })

    return pd.DataFrame(satirlar)


def tum_analizi_calistir(dosya_yolu=None, hedef_sutun="ruzgar", min_gecikme=48):
    """
    Tüm veri hazırlama, model eğitimi, maliyet hesabı, DM testi,
    Tasarruf-Doğruluk eğrisi ve Duyarlılık analizini çalıştırıp döndürür.
    """
    ham_df = veriyi_yukle(dosya_yolu)
    islenmis_df = oznitelikleri_uret(ham_df, hedef_sutun=hedef_sutun, min_gecikme=min_gecikme)
    train_df, test_df = egitim_test_ayir(islenmis_df, egitim_orani=0.80)

    tahminler, ek_bilgiler = modelleri_egit_ve_tahmin_et(
        train_df, test_df, hedef_sutun=hedef_sutun, min_gecikme=min_gecikme
    )

    y_test = test_df[hedef_sutun].values
    pi_fazla = test_df["pi_fazla"].values
    pi_eksik = test_df["pi_eksik"].values
    tau_gercek = test_df["tau_gercek"].values
    toplam_uretim_mwh = float(np.sum(y_test))

    ref_isim = "Sabit tau = 0.50 (Simetrik MAE)"
    ref_maliyet_serisi = saatlik_dengesizlik_maliyeti(
        y_test, tahminler[ref_isim], pi_fazla, pi_eksik
    )
    ref_toplam_maliyet = float(np.sum(ref_maliyet_serisi))

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
        birim_maliyet = toplam_maliyet / max(toplam_uretim_mwh, 1.0)
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

    # Tablo 4: Sistem Yönü Tahmini Karışıklık Matrisi
    cm = ek_bilgiler["karisiklik_matrisi"]
    tablo4_cm = pd.DataFrame(
        cm,
        index=["Gerçek: Enerji Açığı (tau <= 0.5)", "Gerçek: Enerji Fazlası (tau > 0.5)"],
        columns=["Tahmin: Enerji Açığı", "Tahmin: Enerji Fazlası"],
    )

    # Tablo 5: Tasarruf-Doğruluk Eğrisi
    egri_df, basabas_dogruluk = tasarruf_dogruluk_egrisi_hesapla(
        y_test, pi_fazla, pi_eksik, ref_toplam_maliyet, ek_bilgiler
    )

    # Tablo 6: Duyarlılık Analizi (±%10, ±%20)
    duyarlilik_df = duyarlilik_analizi_hesapla(
        y_test, tahminler, pi_fazla, pi_eksik, ek_bilgiler
    )

    return {
        "hedef_sutun": hedef_sutun,
        "min_gecikme": min_gecikme,
        "tablo1_performans": tablo1_df,
        "tablo2_dm_testi": tablo2_df,
        "tablo3_sistem_yonu": tablo3_df,
        "tablo4_karisiklik": tablo4_cm,
        "tablo5_tasarruf_dogruluk": egri_df,
        "basabas_dogruluk": basabas_dogruluk,
        "tablo6_duyarlilik": duyarlilik_df,
        "ham_df": ham_df,
        "train_df": train_df,
        "test_df": test_df,
        "tahminler": tahminler,
        "maliyet_serileri": maliyet_serileri,
        "ek_bilgiler": ek_bilgiler,
    }


def tablolari_disa_aktar(cikti_klasoru="cikti_rapor"):
    """
    Rüzgar (48s ana + 24s ek) ve Güneş (48s ek) sonuçlarını CSV dosyalarına kaydeder.
    """
    os.makedirs(cikti_klasoru, exist_ok=True)
    ana = tum_analizi_calistir(hedef_sutun="ruzgar", min_gecikme=48)
    ek_24 = tum_analizi_calistir(hedef_sutun="ruzgar", min_gecikme=24)
    gunes_48 = tum_analizi_calistir(hedef_sutun="gunes", min_gecikme=48)

    ana["tablo1_performans"].to_csv(os.path.join(cikti_klasoru, "tablo1_ruzgar_48s_ana.csv"), index=False, encoding="utf-8-sig")
    ana["tablo2_dm_testi"].to_csv(os.path.join(cikti_klasoru, "tablo2_ruzgar_48s_dm_testi.csv"), index=False, encoding="utf-8-sig")
    ana["tablo3_sistem_yonu"].to_csv(os.path.join(cikti_klasoru, "tablo3_ruzgar_48s_sistem_yonu.csv"), index=False, encoding="utf-8-sig")
    ana["tablo4_karisiklik"].to_csv(os.path.join(cikti_klasoru, "tablo4_ruzgar_48s_karisiklik_matrisi.csv"), encoding="utf-8-sig")
    ana["tablo5_tasarruf_dogruluk"].to_csv(os.path.join(cikti_klasoru, "tablo5_tasarruf_dogruluk_egrisi.csv"), index=False, encoding="utf-8-sig")
    ana["tablo6_duyarlilik"].to_csv(os.path.join(cikti_klasoru, "tablo6_duyarlilik_analizi.csv"), index=False, encoding="utf-8-sig")

    ek_24["tablo1_performans"].to_csv(os.path.join(cikti_klasoru, "ek_tablo_ruzgar_24s.csv"), index=False, encoding="utf-8-sig")
    ek_24["tablo2_dm_testi"].to_csv(os.path.join(cikti_klasoru, "ek_tablo_ruzgar_24s_dm.csv"), index=False, encoding="utf-8-sig")

    gunes_48["tablo1_performans"].to_csv(os.path.join(cikti_klasoru, "ek_tablo_gunes_48s.csv"), index=False, encoding="utf-8-sig")
    gunes_48["tablo2_dm_testi"].to_csv(os.path.join(cikti_klasoru, "ek_tablo_gunes_48s_dm.csv"), index=False, encoding="utf-8-sig")
    return ana, ek_24, gunes_48


if __name__ == "__main__":
    print("=" * 88)
    print("ASIMOPT - ANA SONUÇ (RÜZGAR, >=48 SAAT GECİKME) VE EK TABLOLAR")
    print("=" * 88)
    ana, ek_24, gunes_48 = tablolari_disa_aktar()

    ek = ana["ek_bilgiler"]
    print(f"\nTest Dönemi                : {ana['test_df']['ts'].min()} -> {ana['test_df']['ts'].max()} ({len(ana['test_df'])} saat)")
    print(f"Yön Tahmin Doğruluğu       : %{ek['yon_tahmin_dogrulugu'] * 100:.2f}")
    print(f"Çoğunluk Sınıfı Doğruluğu  : %{ek['cogunluk_sinifi_dogrulugu'] * 100:.2f}")
    if ana["basabas_dogruluk"] is not None:
        print(f"Başabaş Yön Doğruluğu Eşiği: %{ana['basabas_dogruluk']:.2f}")

    print("\n[ANA TABLO 1] Rüzgar (>=48s Gecikme) Model Karşılaştırması:")
    print(ana["tablo1_performans"].to_string(index=False))

    print("\n[ANA TABLO 2] Diebold-Mariano (DM) Testi:")
    print(ana["tablo2_dm_testi"].drop(columns=["p-Değeri (Sayısal)"]).to_string(index=False))

    print("\n[TABLO 5] Tasarruf-Doğruluk Eğrisi:")
    print(ana["tablo5_tasarruf_dogruluk"].to_string(index=False))

    print("\n[EK TABLO] Rüzgar (>=24s Gecikme) Karşılaştırması:")
    print(ek_24["tablo1_performans"].to_string(index=False))

    print("\n[EK TABLO] Güneş (>=48s Gecikme) Karşılaştırması:")
    print(gunes_48["tablo1_performans"].to_string(index=False))
    print("\nTüm CSV dosyaları 'cikti_rapor/' klasörüne kaydedildi.")
