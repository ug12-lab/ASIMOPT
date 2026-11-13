# -*- coding: utf-8 -*-
"""
figures.py
==========
ASIMOPT - Akademik Yayın Kalitesinde (300 DPI PNG) Grafik Üretici ve ZIP Paketleyici.

Danışman değerlendirme raporunda istenen 8 temel grafiği matplotlib (Agg) kullanarak
300 DPI çözünürlükte PNG olarak üretir, `cikti_rapor/grafikler/` altına kaydeder ve
Streamlit arayüzü üzerinden tekil / toplu ZIP indirme desteği sağlar.

Çalıştırma:
    python figures.py
"""

import io
import os
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from features import TAU_MIN, TAU_MAX
from tables import tum_analizi_calistir, tablolari_disa_aktar


# Yayın kalitesi genel stil ayarları
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#CBD5E1",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
})


MODEL_KISA_ADLAR = {
    "Klasik Simetrik (MSE)": "Simetrik (MSE)",
    "Sabit tau = 0.30 (Kuantil)": "Sabit τ = 0.30",
    "Sabit tau = 0.50 (Simetrik MAE)": "Sabit τ = 0.50 (Ref.)",
    "Sabit tau = 0.70 (Kuantil)": "Sabit τ = 0.70",
    "Eğitime Gömülü Özel Kayıp (Custom Obj.)": "Özel Kayıp (Custom Obj.)",
    "Gerçekçi Kuantil Seçimi (Tahmin tau_hat)": "Gerçekçi Seçim (τ̂)",
    "Üst Sınır / Oracle (Gerçekleşmiş tau*)": "Üst Sınır / Oracle (τ*)",
}


def _kisa_model_adi(uzun_ad):
    return MODEL_KISA_ADLAR.get(uzun_ad, uzun_ad)


def _fig_to_png_bytes(fig, dpi=300):
    """Matplotlib figürünü 300 DPI PNG byte dizisine çevirir ve belleği serbest bırakır."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def grafik_01_tau_histogram(sonuclar):
    """Grafik 1: Optimal Asimetri Oranı (tau*) Histogramı."""
    test_df = sonuclar["test_df"]
    train_df = sonuclar["train_df"]
    tau_test = test_df["tau_gercek"].values
    tau_train = train_df["tau_gercek"].values

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(tau_train, bins=bins, density=True, alpha=0.45, color="#64748B", edgecolor="white", label=f"Eğitim Seti (%80, n={len(tau_train):,})")
    ax.hist(tau_test, bins=bins, density=True, alpha=0.65, color="#2563EB", edgecolor="white", label=f"Test Seti (%20, n={len(tau_test):,})")

    ax.axvline(0.50, color="#DC2626", linestyle="--", linewidth=1.6, label="Simetrik Eşik (τ = 0.50)")
    ax.axvline(float(np.mean(tau_test)), color="#0F766E", linestyle="-.", linewidth=1.6, label=f"Test Ortalaması (τ* = {np.mean(tau_test):.3f})")
    ax.axvspan(TAU_MIN, TAU_MAX, alpha=0.06, color="#10B981", label=f"Kırpma Aralığı [{TAU_MIN:.2f}, {TAU_MAX:.2f}]")

    ax.set_title("Grafik 1: Saatlik Optimal Asimetri Oranı (τ*) Dağılımı")
    ax.set_xlabel("Optimal Asimetri Parametresi τ* = π_eksik / (π_eksik + π_fazla)")
    ax.set_ylabel("Olasılık Yoğunluğu")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    return _fig_to_png_bytes(fig)


def grafik_02_ornek_hafta(sonuclar, baslangic_saat=None):
    """Grafik 2: Test Setinden Örnek 1 Haftalık (168 Saat) Gerçekleşen vs. Model Tahminleri."""
    test_df = sonuclar["test_df"].reset_index(drop=True)
    hedef = sonuclar["hedef_sutun"]
    tahminler = sonuclar["tahminler"]
    n_test = len(test_df)
    pencere = min(168, n_test)

    if baslangic_saat is None:
        # En yüksek varyanslı ve dinamik 168 saatlik haftayı seç
        seri = test_df[hedef].values
        en_iyi_idx = 0
        en_yuksek_std = -1.0
        for s in range(0, max(1, n_test - pencere), 24):
            std_val = float(np.std(seri[s:s + pencere]))
            if std_val > en_yuksek_std and float(np.mean(seri[s:s + pencere])) > 0:
                en_yuksek_std = std_val
                en_iyi_idx = s
        baslangic_saat = en_iyi_idx

    bitis_saat = min(baslangic_saat + pencere, n_test)
    zaman = pd.to_datetime(test_df["ts"].iloc[baslangic_saat:bitis_saat])
    gercek = test_df[hedef].iloc[baslangic_saat:bitis_saat].values

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(zaman, gercek, color="#0F172A", linewidth=2.2, label="Gerçekleşen Üretim", zorder=5)

    cizim_ayar = [
        ("Sabit tau = 0.50 (Simetrik MAE)", "#64748B", "--", 1.5),
        ("Sabit tau = 0.70 (Kuantil)", "#D97706", ":", 1.7),
        ("Eğitime Gömülü Özel Kayıp (Custom Obj.)", "#7C3AED", "-.", 1.5),
        ("Gerçekçi Kuantil Seçimi (Tahmin tau_hat)", "#2563EB", "-", 1.8),
        ("Üst Sınır / Oracle (Gerçekleşmiş tau*)", "#059669", "--", 1.6),
    ]
    for model_adi, renk, stil, kalinlik in cizim_ayar:
        if model_adi in tahminler:
            ax.plot(
                zaman,
                tahminler[model_adi][baslangic_saat:bitis_saat],
                color=renk,
                linestyle=stil,
                linewidth=kalinlik,
                label=_kisa_model_adi(model_adi),
                alpha=0.92,
            )

    hedef_etiket = "Rüzgar Üretimi" if hedef == "ruzgar" else ("Güneş Üretimi" if hedef == "gunes" else hedef)
    ax.set_title(f"Grafik 2: Örnek 168 Saatlik Test Penceresi — {hedef_etiket} ({sonuclar['min_gecikme']}s Gecikme)")
    ax.set_xlabel("Tarih / Saat")
    ax.set_ylabel("Üretim (MWh)")
    ax.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", framealpha=0.92)
    fig.autofmt_xdate()
    return _fig_to_png_bytes(fig)


def grafik_03_model_maliyet_karsilastirma(sonuclar):
    """Grafik 3: Modellerin Toplam Dengesizlik Maliyeti ve Tasarruf (%) Karşılaştırması."""
    t1 = sonuclar["tablo1_performans"].copy()
    t1["KisaAd"] = t1["Model"].apply(_kisa_model_adi)
    t1 = t1.sort_values("Toplam Dengesizlik Maliyeti (TL)", ascending=True).reset_index(drop=True)

    renkler = []
    for ad in t1["Model"]:
        if "Oracle" in ad:
            renkler.append("#059669")
        elif "tau_hat" in ad:
            renkler.append("#2563EB")
        elif "Custom Obj" in ad:
            renkler.append("#7C3AED")
        elif "0.70" in ad:
            renkler.append("#D97706")
        elif "0.50" in ad:
            renkler.append("#475569")
        else:
            renkler.append("#94A3B8")

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    milyon_tl = t1["Toplam Dengesizlik Maliyeti (TL)"].values / 1e6
    bars = ax.barh(t1["KisaAd"], milyon_tl, color=renkler, height=0.62, edgecolor="#1E293B", linewidth=0.5)

    xmax = float(np.max(milyon_tl)) * 1.24
    ax.set_xlim(0, xmax)
    for bar, (_, row) in zip(bars, t1.iterrows()):
        tasarruf = row["Simetrik Modele Göre Tasarruf (%)"]
        etiket = f"{bar.get_width():.2f} M TL ({tasarruf:+.2f}%)"
        ax.text(
            bar.get_width() + xmax * 0.012,
            bar.get_y() + bar.get_height() / 2.0,
            etiket,
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold" if ("0.70" in row["Model"] or "tau_hat" in row["Model"] or "Oracle" in row["Model"]) else "normal",
            color="#0F172A",
        )

    ax.set_title(f"Grafik 3: Test Seti Toplam Dengesizlik Maliyeti Karşılaştırması ({sonuclar['min_gecikme']}s Gecikme)")
    ax.set_xlabel("Toplam Dengesizlik Maliyeti (Milyon TL) — Referansa Göre Tasarruf (%)")
    ax.invert_yaxis()
    return _fig_to_png_bytes(fig)


def grafik_04_sistem_yonune_gore_maliyet(sonuclar):
    """Grafik 4: Sistem Yönüne Göre (Enerji Fazlası vs. Enerji Açığı) Saatlik Ortalama Maliyet."""
    t3 = sonuclar["tablo3_sistem_yonu"].copy()
    t3["KisaAd"] = t3["Model"].apply(_kisa_model_adi)

    x = np.arange(len(t3))
    genislik = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(
        x - genislik / 2,
        t3["Enerji Fazlası Saatlik Ort. Maliyet (TL)"],
        width=genislik,
        color="#0284C7",
        label="Enerji Fazlası Saatleri (τ* > 0.50)",
    )
    ax.bar(
        x + genislik / 2,
        t3["Enerji Açığı Saatlik Ort. Maliyet (TL)"],
        width=genislik,
        color="#EA580C",
        label="Enerji Açığı Saatleri (τ* <= 0.50)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(t3["KisaAd"], rotation=22, ha="right")
    ax.set_ylabel("Saatlik Ortalama Dengesizlik Maliyeti (TL / saat)")
    ax.set_title(f"Grafik 4: Sistem Yönüne Göre Saatlik Ortalama Maliyetler ({sonuclar['min_gecikme']}s Gecikme)")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    return _fig_to_png_bytes(fig)


def grafik_05_yon_karisiklik_matrisi(sonuclar):
    """Grafik 5: Sistem Yönü Tahmini Karışıklık Matrisi ve Çoğunluk Sınıfı Karşılaştırması."""
    cm = sonuclar["ek_bilgiler"]["karisiklik_matrisi"]
    yon_acc = sonuclar["ek_bilgiler"]["yon_tahmin_dogrulugu"] * 100.0
    maj_acc = sonuclar["ek_bilgiler"]["cogunluk_sinifi_dogrulugu"] * 100.0

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Saat Sayısı")

    siniflar = ["Enerji Açığı\n(τ <= 0.50)", "Enerji Fazlası\n(τ > 0.50)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Tahmin:\n{s}" for s in siniflar])
    ax.set_yticklabels([f"Gerçek:\n{s}" for s in siniflar])

    satir_toplam = cm.sum(axis=1, keepdims=True)
    esik = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            deger = int(cm[i, j])
            yuzde = (deger / max(int(satir_toplam[i, 0]), 1)) * 100.0
            renk = "white" if deger > esik else "#0F172A"
            ax.text(
                j, i, f"{deger:,}\n(%{yuzde:.1f})",
                ha="center", va="center", color=renk, fontsize=11, fontweight="bold"
            )

    ax.grid(False)
    ax.set_title(
        f"Grafik 5: Sistem Yönü Tahmini Karışıklık Matrisi ({sonuclar['min_gecikme']}s)\n"
        f"Model Doğruluğu: %{yon_acc:.2f}  |  Çoğunluk Sınıfı Tabanı: %{maj_acc:.2f} (+{yon_acc - maj_acc:.2f} puan)"
    )
    return _fig_to_png_bytes(fig)


def grafik_06_tasarruf_dogruluk_egrisi(sonuclar):
    """Grafik 6: Yön Tahmin Doğruluğu vs. Maliyet Tasarrufu Eğrisi ve Başabaş Eşiği."""
    egri_df = sonuclar["tablo5_tasarruf_dogruluk"]
    basabas = sonuclar["basabas_dogruluk"]
    yon_acc = sonuclar["ek_bilgiler"]["yon_tahmin_dogrulugu"] * 100.0
    maj_acc = sonuclar["ek_bilgiler"]["cogunluk_sinifi_dogrulugu"] * 100.0

    t1 = sonuclar["tablo1_performans"]
    gercekci_row = t1[t1["Model"] == "Gerçekçi Kuantil Seçimi (Tahmin tau_hat)"]
    gercekci_tasarruf = float(gercekci_row["Simetrik Modele Göre Tasarruf (%)"].iloc[0]) if len(gercekci_row) else 0.0

    x = egri_df["Hedef Yön Doğruluğu (%)"].values
    y = egri_df["Ortalama Tasarruf (%)"].values
    std = egri_df["Tasarruf Std (%)"].values

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(x, y, marker="o", linewidth=2.2, color="#2563EB", label="Simüle Edilen Tasarruf Eğrisi (Ortalama)")
    ax.fill_between(x, y - std, y + std, color="#93C5FD", alpha=0.35, label="±1 Std. Sapma Bandı (10 tekrar)")

    ax.axhline(0.0, color="#DC2626", linestyle="--", linewidth=1.4, label="Başabaş Çizgisi (%0 Tasarruf)")
    ax.axvline(maj_acc, color="#64748B", linestyle=":", linewidth=1.4, label=f"Çoğunluk Sınıfı Tabanı (%{maj_acc:.1f})")

    if basabas is not None:
        ax.axvline(basabas, color="#EA580C", linestyle="-.", linewidth=1.6, label=f"Başabaş Doğruluk Eşiği (%{basabas:.1f})")
        ax.scatter([basabas], [0.0], color="#EA580C", s=70, zorder=6)

    ax.scatter(
        [yon_acc], [gercekci_tasarruf],
        color="#7C3AED", marker="D", s=85, zorder=7,
        label=f"Mevcut Gerçekçi Model (%{yon_acc:.1f} → {gercekci_tasarruf:+.2f}%)"
    )

    ax.set_title(f"Grafik 6: Sistem Yönü Doğruluğuna Bağlı Tasarruf Eğrisi ({sonuclar['min_gecikme']}s Gecikme)")
    ax.set_xlabel("Sistem Yönü Tahmin Doğruluğu (%)")
    ax.set_ylabel("Simetrik Referans Modele (τ=0.50) Göre Tasarruf (%)")
    ax.set_xlim(48, 102)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92)
    return _fig_to_png_bytes(fig)


def grafik_07_dm_testi_ozet(sonuclar):
    """Grafik 7: Diebold-Mariano (DM) İstatistiksel Anlamlılık Özeti."""
    t2 = sonuclar["tablo2_dm_testi"].copy()
    t2["KisaAd"] = t2["Karşılaştırılan Model (vs. Sabit tau=0.50)"].apply(_kisa_model_adi)
    t2 = t2.sort_values("DM Test İstatistiği", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    dm_vals = t2["DM Test İstatistiği"].values
    renkler = [
        "#059669" if (p < 0.05 and d > 0) else ("#DC2626" if (p < 0.05 and d < 0) else "#94A3B8")
        for d, p in zip(dm_vals, t2["p-Değeri (Sayısal)"].values)
    ]

    bars = ax.barh(t2["KisaAd"], dm_vals, color=renkler, height=0.58, edgecolor="#1E293B", linewidth=0.5)
    ax.axvline(0.0, color="#0F172A", linewidth=1.0)
    ax.axvline(1.96, color="#059669", linestyle="--", linewidth=1.3, label="Anlamlı Tasarruf Sınırı (+1.96, p<0.05)")
    ax.axvline(-1.96, color="#DC2626", linestyle="--", linewidth=1.3, label="Anlamlı Maliyet Artışı Sınırı (-1.96)")

    min_x = min(-3.5, float(np.min(dm_vals)) * 1.45 - 0.8)
    max_x = max(3.5, float(np.max(dm_vals)) * 1.45 + 0.8)
    ax.set_xlim(min_x, max_x)

    for bar, (_, row) in zip(bars, t2.iterrows()):
        val = row["DM Test İstatistiği"]
        p_str = str(row["p-Değeri"])
        p_metin = f"p {p_str}" if p_str.startswith("<") else f"p={p_str}"
        etiket = f"DM={val:+.2f} ({p_metin})"
        ha = "left" if val >= 0 else "right"
        offset = (max_x - min_x) * 0.01 * (1 if val >= 0 else -1)
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2.0, etiket, va="center", ha=ha, fontsize=9, fontweight="bold")

    ax.set_title(f"Grafik 7: Diebold-Mariano (DM) Testi Özeti — Sabit τ = 0.50 Referansına Karşı ({sonuclar['min_gecikme']}s)")
    ax.set_xlabel("DM Test İstatistiği (Pozitif = Referanstan Düşük Maliyet)")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92)
    return _fig_to_png_bytes(fig)


def grafik_08_duyarlilik_analizi(sonuclar):
    """Grafik 8: Parametre Duyarlılık Analizi (±%10, ±%20)."""
    df = sonuclar["tablo6_duyarlilik"]
    parametreler = [
        ("pi_eksik (Eksik Üretim Cezası)", "Eksik Üretim Cezası (π_eksik)"),
        ("pi_fazla (Fazla Üretim Maliyeti)", "Fazla Üretim Maliyeti (π_fazla)"),
        ("tau Parametresi Sapması", "Asimetri Parametresi (τ) Sapması"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for ax, (param_key, baslik) in zip(axes, parametreler):
        alt = df[df["Parametre"] == param_key].sort_values("Degisim_Orani")
        x = alt["Degisim_Orani"].values * 100.0
        ax.plot(x, alt["Üst Sınır (tau*) Tasarruf (%)"], marker="s", linewidth=2.0, color="#059669", label="Üst Sınır (τ*)")
        ax.plot(x, alt["Sabit tau=0.70 Tasarruf (%)"], marker="^", linewidth=2.0, color="#D97706", label="Sabit τ=0.70")
        ax.plot(x, alt["Gerçekçi (tau_hat) Tasarruf (%)"], marker="o", linewidth=2.0, color="#2563EB", label="Gerçekçi (τ̂)")
        ax.axhline(0.0, color="#64748B", linestyle="--", linewidth=1.0)
        ax.axvline(0.0, color="#94A3B8", linestyle=":", linewidth=1.0)
        ax.set_title(baslik, fontsize=10.5)
        ax.set_xlabel("Değişim (%)")
        ax.set_xticks([-20, -10, 0, 10, 20])

    axes[0].set_ylabel("Simetrik Modele (τ=0.50) Göre Tasarruf (%)")
    axes[0].legend(loc="best", frameon=True, facecolor="white", fontsize=8.5)
    fig.suptitle(f"Grafik 8: Dengesizlik Cezaları ve τ Parametresi Duyarlılık Analizi ({sonuclar['min_gecikme']}s Gecikme)", fontsize=12, fontweight="bold", y=1.03)
    return _fig_to_png_bytes(fig)


def tum_grafikleri_uret(sonuclar):
    """
    8 grafiğin tamamını 300 DPI PNG byte'ları olarak üretip sıralı sözlük döndürür.
    Anahtarlar dosya adlarıdır; değerler (baslik, png_bytes) şeklindedir.
    """
    return {
        "01_tau_histogram.png": ("Grafik 1: Optimal Asimetri Oranı (τ*) Dağılımı", grafik_01_tau_histogram(sonuclar)),
        "02_ornek_hafta_tahmin_gercek.png": ("Grafik 2: Örnek 168 Saatlik Gerçekleşen vs. Model Tahminleri", grafik_02_ornek_hafta(sonuclar)),
        "03_model_maliyet_karsilastirma.png": ("Grafik 3: Modellerin Toplam Dengesizlik Maliyeti Karşılaştırması", grafik_03_model_maliyet_karsilastirma(sonuclar)),
        "04_sistem_yonune_gore_maliyet.png": ("Grafik 4: Sistem Yönüne Göre Saatlik Ortalama Maliyetler", grafik_04_sistem_yonune_gore_maliyet(sonuclar)),
        "05_yon_karisiklik_matrisi.png": ("Grafik 5: Sistem Yönü Tahmini Karışıklık Matrisi", grafik_05_yon_karisiklik_matrisi(sonuclar)),
        "06_tasarruf_dogruluk_egrisi.png": ("Grafik 6: Yön Doğruluğu vs. Maliyet Tasarrufu Eğrisi", grafik_06_tasarruf_dogruluk_egrisi(sonuclar)),
        "07_dm_testi_ozet.png": ("Grafik 7: Diebold-Mariano (DM) Testi Özeti", grafik_07_dm_testi_ozet(sonuclar)),
        "08_duyarlilik_analizi.png": ("Grafik 8: Parametre Duyarlılık Analizi (±%10, ±%20)", grafik_08_duyarlilik_analizi(sonuclar)),
    }


def grafikleri_disa_aktar(sonuclar, cikti_klasoru="cikti_rapor/grafikler"):
    """8 grafiği 300 DPI PNG olarak diske kaydeder."""
    os.makedirs(cikti_klasoru, exist_ok=True)
    grafikler = tum_grafikleri_uret(sonuclar)
    for dosya_adi, (_, png_bytes) in grafikler.items():
        yol = os.path.join(cikti_klasoru, dosya_adi)
        with open(yol, "wb") as f:
            f.write(png_bytes)
    return grafikler


def tum_rapor_paketi_zip(ana, ek_24, gunes_48):
    """
    Tüm tabloları (CSV) ve 8 ana grafiği (300 DPI PNG) tek bir ZIP arşivi olarak bellek içinde paketler.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        csv_dosyalari = {
            "tablolar/tablo1_ruzgar_48s_ana.csv": ana["tablo1_performans"].to_csv(index=False).encode("utf-8-sig"),
            "tablolar/tablo2_ruzgar_48s_dm_testi.csv": ana["tablo2_dm_testi"].to_csv(index=False).encode("utf-8-sig"),
            "tablolar/tablo3_ruzgar_48s_sistem_yonu.csv": ana["tablo3_sistem_yonu"].to_csv(index=False).encode("utf-8-sig"),
            "tablolar/tablo4_ruzgar_48s_karisiklik_matrisi.csv": ana["tablo4_karisiklik"].to_csv().encode("utf-8-sig"),
            "tablolar/tablo5_tasarruf_dogruluk_egrisi.csv": ana["tablo5_tasarruf_dogruluk"].to_csv(index=False).encode("utf-8-sig"),
            "tablolar/tablo6_duyarlilik_analizi.csv": ana["tablo6_duyarlilik"].to_csv(index=False).encode("utf-8-sig"),
            "tablolar/ek_tablo_ruzgar_24s.csv": ek_24["tablo1_performans"].to_csv(index=False).encode("utf-8-sig"),
            "tablolar/ek_tablo_ruzgar_24s_dm.csv": ek_24["tablo2_dm_testi"].to_csv(index=False).encode("utf-8-sig"),
            "tablolar/ek_tablo_gunes_48s.csv": gunes_48["tablo1_performans"].to_csv(index=False).encode("utf-8-sig"),
            "tablolar/ek_tablo_gunes_48s_dm.csv": gunes_48["tablo2_dm_testi"].to_csv(index=False).encode("utf-8-sig"),
        }
        for yol, icerik in csv_dosyalari.items():
            zf.writestr(yol, icerik)

        grafikler_48 = tum_grafikleri_uret(ana)
        for dosya_adi, (_, png_bytes) in grafikler_48.items():
            zf.writestr(f"grafikler_ruzgar_48s/{dosya_adi}", png_bytes)

        grafikler_gunes = tum_grafikleri_uret(gunes_48)
        for dosya_adi, (_, png_bytes) in grafikler_gunes.items():
            zf.writestr(f"grafikler_gunes_48s/{dosya_adi}", png_bytes)

    buf.seek(0)
    return buf.getvalue()


if __name__ == "__main__":
    print("ASIMOPT 300 DPI Grafikler ve Rapor Tabloları Üretiliyor...")
    ana, ek_24, gunes_48 = tablolari_disa_aktar("cikti_rapor")
    grafikleri_disa_aktar(ana, "cikti_rapor/grafikler_ruzgar_48s")
    grafikleri_disa_aktar(gunes_48, "cikti_rapor/grafikler_gunes_48s")
    print("Tamamlandı: 8 adet 300 DPI PNG + tüm CSV tabloları 'cikti_rapor/' içine kaydedildi.")
