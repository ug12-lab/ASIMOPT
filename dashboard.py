# -*- coding: utf-8 -*-
"""
dashboard.py
============
ASIMOPT - İnteraktif Model Değerlendirme, Dengesizlik Maliyeti ve DM Testi Paneli.
"""

import importlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import features
import model
import tables

# Streamlit Cloud önbelleğinde eski modül kalmasını önlemek için zorunlu yenileme
importlib.reload(features)
importlib.reload(model)
importlib.reload(tables)

from features import veriyi_yukle
from tables import tum_analizi_calistir

st.set_page_config(
    page_title="ASIMOPT Akademik Analiz ve Optimizasyon Paneli",
    layout="wide",
    page_icon="⚡",
)

st.title("⚡ ASIMOPT - Asimetrik Kayıp Fonksiyonu ve Dengesizlik Maliyeti Optimizasyonu")
st.markdown(
    "**Gerçekleşmiş Uzlaştırma Fiyatlarından Hesaplanan Dengesizlik Maliyetleri, "
    "Sabit $\\tau$ ($0.3, 0.5, 0.7$), Tahmin Edilen $\\hat{\\tau}$ (Gerçekçi) ve "
    "Gerçekleşmiş $\\tau^*$ (Üst Sınır / Oracle) Karşılaştırması**"
)
st.markdown("---")


@st.cache_data(show_spinner=False)
def modeli_onbellekli_calistir(hedef_kaynak="ruzgar", versiyon="v2"):
    """Modelleri eğitir ve sonuç tablolarını önbelleğe (cache) alır."""
    ham_df = veriyi_yukle()
    sonuclar = tum_analizi_calistir(ham_df, hedef_sutun=hedef_kaynak)
    return ham_df, sonuclar


# Sol Menü
with st.sidebar:
    st.header("⚙️ Analiz Ayarları")
    secilen_kaynak = st.selectbox(
        "Tahmin Edilecek Üretim Kaynağı:",
        options=["ruzgar", "gunes", "uretim_toplam", "dogalgaz", "barajli", "akarsu"],
        format_func=lambda x: {
            "ruzgar": "🌬️ Rüzgar Üretimi (MWh)",
            "gunes": "☀️ Güneş Üretimi (MWh)",
            "uretim_toplam": "🏭 Toplam Üretim (MWh)",
            "dogalgaz": "🔥 Doğal Gaz Üretimi (MWh)",
            "barajli": "💧 Barajlı Hidroelektrik (MWh)",
            "akarsu": "🌊 Akarsu Üretimi (MWh)",
        }[x],
        index=0,
    )

    st.info(
        "📌 **2022–2026 Tam Veri Seti (35.088 Saat):**\n\n"
        "• **Eğitim Seti (%80):** 2022–2025 arası (~27.916 saat) geçmiş verilerle yapay zeka modelleri eğitilir.\n\n"
        "• **Test Seti (%20):** 2025–2026 arası (~6.980 saat) modelin hiç görmediği dönem üzerinde dengesizlik maliyetleri ve DM testi ölçülür."
    )

try:
    with st.spinner("2022-2026 (35.088 saat) verisi işleniyor, 6 model eğitiliyor ve Diebold-Mariano testi hesaplanıyor..."):
        df_ham, analiz = modeli_onbellekli_calistir(secilen_kaynak, versiyon="v2")

    tablo1 = analiz["tablo1_performans"]
    tablo2 = analiz["tablo2_dm_testi"]
    tablo3 = analiz["tablo3_sistem_yonu"]
    test_df = analiz["test_df"]
    tahminler = analiz["tahminler"]
    ek = analiz["ek_bilgiler"]

    ref_satir = tablo1[tablo1["Model"] == "Sabit tau = 0.50 (Simetrik MAE)"].iloc[0]
    gercekci_satir = tablo1[tablo1["Model"] == "ASIMOPT Gerçekçi (Tahmin Edilen tau_hat)"].iloc[0]
    oracle_satir = tablo1[tablo1["Model"] == "ASIMOPT Üst Sınır / Oracle (Gerçekleşmiş tau*)"].iloc[0]

    gercekci_kazanc_tl = ref_satir["Toplam Dengesizlik Maliyeti (TL)"] - gercekci_satir["Toplam Dengesizlik Maliyeti (TL)"]
    oracle_kazanc_tl = ref_satir["Toplam Dengesizlik Maliyeti (TL)"] - oracle_satir["Toplam Dengesizlik Maliyeti (TL)"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Toplam Veri (2022–2026)",
        f"{len(df_ham):,} Saat",
        f"Eğitim: {len(df_ham) - len(test_df):,} | Test: {len(test_df):,}",
    )
    col2.metric(
        "Sistem Yönü Tahmin Doğruluğu",
        f"%{ek['yon_tahmin_dogrulugu'] * 100:.2f}",
        "Teklif Anında (Gün Öncesi)",
    )
    col3.metric(
        "ASIMOPT Gerçekçi (Tahmin Edilen τ̂) Tasarrufu",
        f"%{gercekci_satir['Simetrik Modele Göre Tasarruf (%)']:.2f}",
        f"{gercekci_kazanc_tl:,.0f} ₺ Kazanç",
    )
    col4.metric(
        "ASIMOPT Üst Sınır (Gerçekleşmiş τ*) Tasarrufu",
        f"%{oracle_satir['Simetrik Modele Göre Tasarruf (%)']:.2f}",
        f"{oracle_kazanc_tl:,.0f} ₺ (Teorik Üst Sınır)",
    )

    st.markdown("---")

    sekme1, sekme2, sekme3 = st.tabs([
        "📋 Rapor Tabloları ve DM Testi",
        "📊 Maliyet ve Tahmin Grafikleri",
        "⚡ 2022–2026 Tüm Veri Seti (Fiyat, Üretim ve τ*)",
    ])

    with sekme1:
        st.subheader("Tablo 1: Tüm Modellerin Performans ve Dengesizlik Maliyeti Karşılaştırması")
        st.caption(
            "Sabit τ (0.30, 0.50, 0.70), sistem yönünü önceden tahmin ederek τ̂ belirleyen "
            "Gerçekçi ASIMOPT modeli ve gerçekleşmiş τ* kullanan Üst Sınır (Oracle) modeli."
        )
        st.dataframe(tablo1, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Tablo 1'i İndir (CSV)",
            data=tablo1.to_csv(index=False, encoding="utf-8-sig"),
            file_name="tablo1_model_karsilastirma.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.subheader("Tablo 2: Diebold-Mariano (DM) İstatistiksel Anlamlılık Testi Sonuçları")
        st.caption(
            "Referans Model: Sabit τ = 0.50 (Simetrik MAE). Newey-West (HAC) otokorelasyon düzeltmeli "
            "DM test istatistiği ve çift kuyruklu p-değerleri."
        )
        gosterilecek_t2 = tablo2.drop(columns=["p-Değeri (Sayısal)"])
        st.dataframe(gosterilecek_t2, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Tablo 2'yi İndir (CSV)",
            data=gosterilecek_t2.to_csv(index=False, encoding="utf-8-sig"),
            file_name="tablo2_dm_testi.csv",
            mime="text/csv",
        )

        if not tablo3.empty:
            st.markdown("---")
            st.subheader("Tablo 3: Sistem Yönüne Göre Saatlik Ortalama Dengesizlik Maliyetleri (TL/Saat)")
            st.dataframe(tablo3, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Tablo 3'ü İndir (CSV)",
                data=tablo3.to_csv(index=False, encoding="utf-8-sig"),
                file_name="tablo3_sistem_yonu.csv",
                mime="text/csv",
            )

    with sekme2:
        st.subheader("💰 Modellerin Toplam Dengesizlik Maliyeti Karşılaştırması (TL)")
        fig_bar = px.bar(
            tablo1,
            x="Model",
            y="Toplam Dengesizlik Maliyeti (TL)",
            color="Model",
            text="Simetrik Modele Göre Tasarruf (%)",
            title="Test Dönemi Toplam Dengesizlik Maliyeti ve Simetrik Modele Göre Tasarruf (%)",
        )
        fig_bar.update_traces(texttemplate="Tasarruf: %{text:.2f}%", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("📈 Saatlik Üretim Tahmini Karşılaştırması (Son 168 Saat / 1 Hafta)")
        son_n = 168
        zaman_dilimi = test_df["ts"].iloc[-son_n:]

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=zaman_dilimi,
            y=test_df[secilen_kaynak].iloc[-son_n:],
            mode="lines",
            name="Gerçekleşen Üretim",
            line=dict(color="black", width=2),
        ))
        fig_pred.add_trace(go.Scatter(
            x=zaman_dilimi,
            y=tahminler["Sabit tau = 0.50 (Simetrik MAE)"][-son_n:],
            mode="lines",
            name="Sabit τ = 0.50 (Simetrik MAE)",
            line=dict(color="#ef4444", dash="dot"),
        ))
        fig_pred.add_trace(go.Scatter(
            x=zaman_dilimi,
            y=tahminler["ASIMOPT Gerçekçi (Tahmin Edilen tau_hat)"][-son_n:],
            mode="lines",
            name="ASIMOPT Gerçekçi (Tahmin Edilen τ̂)",
            line=dict(color="#22c55e", width=2),
        ))
        fig_pred.add_trace(go.Scatter(
            x=zaman_dilimi,
            y=tahminler["ASIMOPT Üst Sınır / Oracle (Gerçekleşmiş tau*)"][-son_n:],
            mode="lines",
            name="ASIMOPT Üst Sınır (Gerçekleşmiş τ*)",
            line=dict(color="#3b82f6", dash="dash"),
            opacity=0.8,
        ))
        fig_pred.update_layout(hovermode="x unified", yaxis_title="Üretim (MWh)", xaxis_title="Tarih")
        st.plotly_chart(fig_pred, use_container_width=True)

    with sekme3:
        st.subheader("📅 2022–2026 Arası Dönem Seçimi (35.088 Saatlik Tam Veri)")
        yil_secimi = st.selectbox(
            "İncelenecek Yılı / Dönemi Seçin:",
            options=["2022", "2023", "2024", "2025", "Son 30 Gün (720 Saat)"],
            index=4,
        )

        if yil_secimi == "Son 30 Gün (720 Saat)":
            plot_df = df_ham.tail(720)
        else:
            plot_df = df_ham[df_ham["ts"].dt.year == int(yil_secimi)]

        st.subheader(f"📈 PTF, SMF, PDF ve NDF Fiyat Dinamikleri ({yil_secimi})")
        fig_fiyat = go.Figure()
        fig_fiyat.add_trace(go.Scatter(x=plot_df["ts"], y=plot_df["PTF"], mode="lines", name="PTF", line=dict(color="blue")))
        fig_fiyat.add_trace(go.Scatter(x=plot_df["ts"], y=plot_df["SMF"], mode="lines", name="SMF", line=dict(color="orange"), opacity=0.7))
        fig_fiyat.add_trace(go.Scatter(x=plot_df["ts"], y=plot_df["NDF_neg"], mode="lines", name="NDF (Eksik Üretim Fiyatı)", line=dict(color="red", dash="dot")))
        fig_fiyat.add_trace(go.Scatter(x=plot_df["ts"], y=plot_df["PDF_poz"], mode="lines", name="PDF (Fazla Üretim Fiyatı)", line=dict(color="green", dash="dot")))
        fig_fiyat.update_layout(hovermode="x unified", yaxis_title="Fiyat (TL/MWh)", xaxis_title="Tarih")
        st.plotly_chart(fig_fiyat, use_container_width=True)

        kaynaklar = ["dogalgaz", "ithal_komur", "barajli", "linyit", "ruzgar", "akarsu", "gunes"]
        mevcut_kaynaklar = [k for k in kaynaklar if k in plot_df.columns]
        if mevcut_kaynaklar:
            st.subheader(f"🏭 Saatlik Enerji Üretim Dağılımı ({yil_secimi})")
            fig_uretim = px.area(
                plot_df,
                x="ts",
                y=mevcut_kaynaklar,
                labels={"value": "Üretim (MWh)", "variable": "Kaynak", "ts": "Tarih"},
            )
            st.plotly_chart(fig_uretim, use_container_width=True)

        st.subheader("⚡ Gerçekleşmiş τ* ve Tahmin Edilen τ̂ Dinamikleri (Test Dönemi Son 168 Saat)")
        fig_tau = go.Figure()
        fig_tau.add_trace(go.Scatter(
            x=zaman_dilimi,
            y=ek["tau_gercek_test"][-son_n:],
            mode="lines",
            name="Gerçekleşmiş τ* (Uzlaştırma Fiyatlarından)",
            line=dict(color="#9333ea"),
            opacity=0.6,
        ))
        fig_tau.add_trace(go.Scatter(
            x=zaman_dilimi,
            y=ek["tau_hat_test"][-son_n:],
            mode="lines",
            name="Teklif Anında Tahmin Edilen τ̂ (Gerçekçi Model)",
            line=dict(color="#eab308", width=2),
        ))
        fig_tau.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="Simetrik Eşik (τ = 0.5)")
        fig_tau.update_layout(hovermode="x unified", yaxis_title="Asimetri Parametresi (τ)", xaxis_title="Tarih")
        st.plotly_chart(fig_tau, use_container_width=True)

except Exception as e:
    st.error(f"Analiz çalıştırılırken bir hata oluştu: {e}")
    st.info("Lütfen GitHub deponuzda `birlesik_veri.csv`, `features.py`, `model.py` ve `tables.py` dosyalarının bulunduğundan emin olun.")
