# -*- coding: utf-8 -*-
"""
dashboard.py
============
ASIMOPT - Asimetrik Kayıp Fonksiyonu ve Dengesizlik Maliyeti Optimizasyonu
Streamlit Akademik Karar Destek, Analiz ve 300 DPI Grafik Dışa Aktarım Arayüzü.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from features import TAU_MIN, TAU_MAX, BITIS_TARIHI
from tables import tum_analizi_calistir
from figures import tum_grafikleri_uret, tum_rapor_paketi_zip


st.set_page_config(
    page_title="ASIMOPT | Dengesizlik Maliyeti ve Asimetrik Kayıp Analizi",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


URETIM_SECENEKLERI = {
    "ruzgar": "🌬️ Rüzgar Üretimi (Ana Çalışma Serisi)",
    "gunes": "☀️ Güneş Üretimi (Ek Tablo & Kapsam Notlu)",
    "uretim_toplam": "🏭 Toplam Üretim (MWh)",
    "barajli": "💧 Barajlı Hidroelektrik (MWh)",
    "dogalgaz": "🔥 Doğal Gaz Üretimi (MWh)",
    "ithal_komur": "🪨 İthal Kömür (MWh)",
    "linyit": "⛏️ Linyit (MWh)",
    "akarsu": "🌊 Akarsu Hidroelektrik (MWh)",
}

GECIKME_SECENEKLERI = {
    48: "≥ 48 Saat Gecikme (Ana Sonuç — Sızıntısız GÖP Akışı)",
    24: "≥ 24 Saat Gecikme (Ek Karşılaştırma Tablosu)",
}


@st.cache_resource(show_spinner="Modeller eğitiliyor ve sonuçlar hesaplanıyor...")
def analiz_getir(hedef_sutun: str, min_gecikme: int):
    return tum_analizi_calistir(hedef_sutun=hedef_sutun, min_gecikme=min_gecikme)


@st.cache_data(show_spinner="8 adet 300 DPI akademik grafik hazırlanıyor...")
def grafik_paketi_getir(hedef_sutun: str, min_gecikme: int):
    sonuc = analiz_getir(hedef_sutun, min_gecikme)
    return tum_grafikleri_uret(sonuc)


def main():
    st.title("⚡ ASIMOPT — Asimetrik Kayıp Fonksiyonu ve Dengesizlik Maliyeti Optimizasyonu")
    st.markdown(
        "Gerçekleşmiş EPİAŞ PTF & SMF verileriyle **7 farklı tahmin yaklaşımının** (MSE, Sabit τ ∈ {0.30, 0.50, 0.70}, "
        "Eğitime Gömülü Özel Kayıp, Saatlik τ̂ ile Kuantil Seçimi ve Oracle τ*) maliyet, yön tahmini ve duyarlılık analizi."
    )
    st.divider()

    # Kenar çubuğu
    with st.sidebar:
        st.header("⚙️ Analiz Ayarları")
        hedef_sutun = st.selectbox(
            "Tahmin Edilecek Üretim Kaynağı:",
            options=list(URETIM_SECENEKLERI.keys()),
            format_func=lambda k: URETIM_SECENEKLERI[k],
            index=0,
        )
        min_gecikme = st.radio(
            "Öznitelik Gecikme Rejimi:",
            options=[48, 24],
            format_func=lambda k: GECIKME_SECENEKLERI[k],
            index=0,
        )
        st.info(
            f"📌 **Veri ve Kurulum Özeti:**\n\n"
            f"• **Tarih Aralığı:** 01.01.2022 – {BITIS_TARIHI.strftime('%d.%m.%Y %H:%M')} "
            f"(01.01.2026 yeni piyasa mekanizması hariç tutuldu).\n\n"
            f"• **Kronolojik Ayrım (%80 / %20):** Doğrulama (validation) seti kullanılmamıştır; "
            f"hiperparametreler literatür tabanlı sabitlenmiştir.\n\n"
            f"• **Kırpma Sınırı:** Tüm modellerde `τ ∈ [{TAU_MIN:.2f}, {TAU_MAX:.2f}]`."
        )
        if st.button("🔄 Önbelleği Temizle ve Yenile"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    try:
        sonuclar = analiz_getir(hedef_sutun, min_gecikme)
    except Exception as exc:
        st.error(f"Analiz çalıştırılırken bir hata oluştu: {exc}")
        st.info("Lütfen `birlesik_veri.csv`, `features.py`, `model.py`, `tables.py` ve `figures.py` dosyalarının eksiksiz olduğunu doğrulayın.")
        return

    tablo1 = sonuclar["tablo1_performans"]
    tablo2 = sonuclar["tablo2_dm_testi"]
    tablo3 = sonuclar["tablo3_sistem_yonu"]
    tablo4 = sonuclar["tablo4_karisiklik"]
    tablo5 = sonuclar["tablo5_tasarruf_dogruluk"]
    tablo6 = sonuclar["tablo6_duyarlilik"]
    basabas = sonuclar["basabas_dogruluk"]
    ek = sonuclar["ek_bilgiler"]
    test_df = sonuclar["test_df"]
    train_df = sonuclar["train_df"]

    if hedef_sutun == "gunes":
        st.warning(
            "☀️ **EPİAŞ Güneş Verisi Metodolojik Dipnotu:** EPİAŞ Şeffaflık Platformu'ndaki gerçek zamanlı güneş üretimi "
            "serisi ağırlıklı olarak **lisanslı (OSOS/telemetri kapsamındaki)** santralleri yansıtmaktadır; Türkiye'deki "
            "yüksek paylı lisanssız GES üretimi doğrudan net tüketimden düşmektedir. Gece saatlerindeki küçük negatif değerler "
            "santral iç tüketiminden (yardımcı yük) kaynaklanmaktadır ve bu nedenle Güneş serisi Rüzgar ana bulgularına **ek tablo** olarak sunulur."
        )

    # Üst Bilgi Şeridi
    st.caption(
        f"**Eğitim Seti:** {len(train_df):,} saat ({train_df['ts'].min()} → {train_df['ts'].max()})  |  "
        f"**Test Seti:** {len(test_df):,} saat ({test_df['ts'].min()} → {test_df['ts'].max()})  |  "
        f"**Aktif Gecikme Rejimi:** ≥ {min_gecikme} saat"
    )

    ref_row = tablo1[tablo1["Model"] == "Sabit tau = 0.50 (Simetrik MAE)"].iloc[0]
    sabit07_row = tablo1[tablo1["Model"] == "Sabit tau = 0.70 (Kuantil)"].iloc[0]
    custom_row = tablo1[tablo1["Model"] == "Eğitime Gömülü Özel Kayıp (Custom Obj.)"].iloc[0]
    gercekci_row = tablo1[tablo1["Model"] == "Gerçekçi Kuantil Seçimi (Tahmin tau_hat)"].iloc[0]
    oracle_row = tablo1[tablo1["Model"] == "Üst Sınır / Oracle (Gerçekleşmiş tau*)"].iloc[0]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Simetrik Referans (τ=0.50)", f"{ref_row['Toplam Dengesizlik Maliyeti (TL)']:,.0f} TL")
    k2.metric(
        "Sabit Asimetrik (τ=0.70)",
        f"{sabit07_row['Toplam Dengesizlik Maliyeti (TL)']:,.0f} TL",
        delta=f"{sabit07_row['Simetrik Modele Göre Tasarruf (%)']:+.2f}% Tasarruf",
    )
    k3.metric(
        "Özel Kayıp (Custom Obj.)",
        f"{custom_row['Toplam Dengesizlik Maliyeti (TL)']:,.0f} TL",
        delta=f"{custom_row['Simetrik Modele Göre Tasarruf (%)']:+.2f}% Tasarruf",
    )
    k4.metric(
        "Gerçekçi Seçim (τ̂)",
        f"{gercekci_row['Toplam Dengesizlik Maliyeti (TL)']:,.0f} TL",
        delta=f"{gercekci_row['Simetrik Modele Göre Tasarruf (%)']:+.2f}% Tasarruf",
    )
    k5.metric(
        "Oracle Üst Sınır (τ*)",
        f"{oracle_row['Toplam Dengesizlik Maliyeti (TL)']:,.0f} TL",
        delta=f"{oracle_row['Simetrik Modele Göre Tasarruf (%)']:+.2f}% Tasarruf",
    )

    st.divider()

    sekme1, sekme2, sekme3, sekme4, sekme5 = st.tabs([
        "📊 1. Ana Performans & DM Testi",
        "🧭 2. Yön Tahmini & Tasarruf Eğrisi",
        "📐 3. Duyarlılık Analizi (±%10, ±%20)",
        "🔁 4. Rüzgar (48s vs 24s) & Güneş Karşılaştırması",
        "🖼️ 5. 300 DPI Grafikler & ZIP İndir",
    ])

    # ------------------------------------------------------------------
    # SEKME 1: Ana Tablolar, DM Testi ve Örnek Hafta İnteraktif Görünümü
    # ------------------------------------------------------------------
    with sekme1:
        st.subheader(f"📋 Tablo 1: Test Seti Performans ve Maliyet Karşılaştırması ({min_gecikme}s Gecikme)")
        st.dataframe(tablo1, use_container_width=True, hide_index=True)

        c1, c2 = st.columns([1.15, 0.85])
        with c1:
            st.subheader("🧪 Tablo 2: Diebold-Mariano (DM) İstatistiksel Anlamlılık Testi")
            st.dataframe(tablo2.drop(columns=["p-Değeri (Sayısal)"]), use_container_width=True, hide_index=True)
        with c2:
            st.subheader("⚖️ Tablo 3: Sistem Yönüne Göre Saatlik Ortalama Maliyet")
            st.dataframe(tablo3, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("📈 İnteraktif Test Seti Tahmin İncelemesi (168 Saatlik Pencere)")
        max_baslangic = max(0, len(test_df) - 168)
        baslangic = st.slider("Başlangıç Saati İndeksi:", min_value=0, max_value=max_baslangic, value=0, step=24)
        pencere_df = test_df.iloc[baslangic:baslangic + 168].copy()

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=pencere_df["ts"], y=pencere_df[hedef_sutun],
            name="Gerçekleşen", line=dict(color="#0F172A", width=2.5),
        ))
        renkler = {
            "Sabit tau = 0.50 (Simetrik MAE)": ("#64748B", "dash"),
            "Sabit tau = 0.70 (Kuantil)": ("#D97706", "dot"),
            "Eğitime Gömülü Özel Kayıp (Custom Obj.)": ("#7C3AED", "dashdot"),
            "Gerçekçi Kuantil Seçimi (Tahmin tau_hat)": ("#2563EB", "solid"),
            "Üst Sınır / Oracle (Gerçekleşmiş tau*)": ("#059669", "dash"),
        }
        for model_adi, (renk, dash_stili) in renkler.items():
            if model_adi in sonuclar["tahminler"]:
                fig_line.add_trace(go.Scatter(
                    x=pencere_df["ts"],
                    y=sonuclar["tahminler"][model_adi][baslangic:baslangic + 168],
                    name=model_adi,
                    line=dict(color=renk, width=1.9, dash=dash_stili),
                ))
        fig_line.update_layout(
            height=420,
            hovermode="x unified",
            xaxis_title="Tarih / Saat",
            yaxis_title="Üretim (MWh)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # ------------------------------------------------------------------
    # SEKME 2: Yön Tahmini Karışıklık Matrisi & Tasarruf-Doğruluk Eğrisi
    # ------------------------------------------------------------------
    with sekme2:
        yon_acc = ek["yon_tahmin_dogrulugu"] * 100.0
        maj_acc = ek["cogunluk_sinifi_dogrulugu"] * 100.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Yön Tahmin Doğruluğu", f"%{yon_acc:.2f}")
        m2.metric("Çoğunluk Sınıfı Tabanı (Always Enerji Açığı)", f"%{maj_acc:.2f}", delta=f"+{yon_acc - maj_acc:.2f} puan model katkısı")
        m3.metric(
            "Başabaş Yön Doğruluğu Eşiği (%0 Tasarruf)",
            f"%{basabas:.2f}" if basabas is not None else "Ulaşılamadı / Daima Pozitif",
        )

        st.markdown(
            f"> **Bulgu Yorumu:** Test döneminde gerçek saatlik yön tahmin doğruluğu **%{yon_acc:.2f}** olup, "
            f"her saati çoğunluk sınıfı olan *Enerji Açığı* seçmekle ulaşılan **%{maj_acc:.2f}** tabanının "
            f"yalnızca **{yon_acc - maj_acc:.2f} puan** üzerindedir. Bu sonuç, gerçekçi yön belirsizliği altında "
            f"dinamik kuantil seçiminin neden sabit asimetrik yaklaşıma (`τ=0.70`) göre düşük kaldığını açıklamaktadır."
        )

        sol, sag = st.columns([0.9, 1.1])
        with sol:
            st.subheader("🧩 Tablo 4: Sistem Yönü Karışıklık Matrisi")
            st.dataframe(tablo4, use_container_width=True)
            st.download_button(
                "⬇️ Karışıklık Matrisini İndir (CSV)",
                data=tablo4.to_csv().encode("utf-8-sig"),
                file_name=f"karisiklik_matrisi_{hedef_sutun}_{min_gecikme}s.csv",
                mime="text/csv",
            )
        with sag:
            st.subheader("📈 Tablo 5: Tasarruf–Doğruluk Eğrisi (%50 – %100)")
            st.dataframe(tablo5, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Tasarruf–Doğruluk Eğrisini İndir (CSV)",
                data=tablo5.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"tasarruf_dogruluk_{hedef_sutun}_{min_gecikme}s.csv",
                mime="text/csv",
            )

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=tablo5["Hedef Yön Doğruluğu (%)"],
            y=tablo5["Ortalama Tasarruf (%)"] + tablo5["Tasarruf Std (%)"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_curve.add_trace(go.Scatter(
            x=tablo5["Hedef Yön Doğruluğu (%)"],
            y=tablo5["Ortalama Tasarruf (%)"] - tablo5["Tasarruf Std (%)"],
            mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(37,99,235,0.18)",
            name="±1 Std. Sapma",
        ))
        fig_curve.add_trace(go.Scatter(
            x=tablo5["Hedef Yön Doğruluğu (%)"],
            y=tablo5["Ortalama Tasarruf (%)"],
            mode="lines+markers", line=dict(color="#2563EB", width=3),
            name="Ortalama Tasarruf (%)",
        ))
        fig_curve.add_hline(y=0.0, line_dash="dash", line_color="#DC2626", annotation_text="Başabaş (%0)")
        if basabas is not None:
            fig_curve.add_vline(x=basabas, line_dash="dashdot", line_color="#EA580C", annotation_text=f"Başabaş Eşiği: %{basabas:.1f}")
        fig_curve.update_layout(
            height=400,
            xaxis_title="Sistem Yönü Tahmin Doğruluğu (%)",
            yaxis_title="Simetrik Modele (τ=0.50) Göre Tasarruf (%)",
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    # ------------------------------------------------------------------
    # SEKME 3: Duyarlılık Analizi (±%10, ±%20)
    # ------------------------------------------------------------------
    with sekme3:
        st.subheader(f"📐 Tablo 6: Dengesizlik Cezaları ve τ Parametresi Duyarlılık Analizi ({min_gecikme}s)")
        st.markdown(
            "Bu analiz, `π_eksik`, `π_fazla` ve tahmin edilen `τ` parametrelerinde **±%10 ve ±%20** sapma olması durumunda "
            "`Sabit τ=0.70`, `Gerçekçi (τ̂)` ve `Üst Sınır (τ*)` modellerinin yüzde tasarrufunun nasıl değiştiğini gösterir."
        )
        st.dataframe(tablo6.drop(columns=["Degisim_Orani"]), use_container_width=True, hide_index=True)

        fig_sens = px.line(
            tablo6,
            x="Degisim_Orani",
            y=["Üst Sınır (tau*) Tasarruf (%)", "Sabit tau=0.70 Tasarruf (%)", "Gerçekçi (tau_hat) Tasarruf (%)"],
            facet_col="Parametre",
            markers=True,
            labels={"Degisim_Orani": "Değişim Oranı", "value": "Tasarruf (%)", "variable": "Model"},
        )
        fig_sens.update_xaxes(tickformat=".0%")
        fig_sens.update_layout(height=420)
        st.plotly_chart(fig_sens, use_container_width=True)

        st.download_button(
            "⬇️ Duyarlılık Analizi Tablosunu İndir (CSV)",
            data=tablo6.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"duyarlilik_analizi_{hedef_sutun}_{min_gecikme}s.csv",
            mime="text/csv",
        )

    # ------------------------------------------------------------------
    # SEKME 4: Rüzgar (48s Ana vs 24s Ek) & Güneş (48s Ek) Karşılaştırması
    # ------------------------------------------------------------------
    with sekme4:
        st.subheader("🔁 Akademik Rapor Karşılaştırma Tabloları (48s Ana vs. 24s Ek & Güneş)")
        st.markdown(
            "Danışman değerlendirme notu gereği **≥ 48 saat gecikmeli Rüzgar serisi** sızıntısız ana bulgu olarak, "
            "**≥ 24 saat gecikmeli Rüzgar serisi** geçmiş çalışma karşılaştırması için ek tablo olarak, "
            "**≥ 48 saat gecikmeli Güneş serisi** ise ayrı kaynak değerlendirmesi olarak sunulur."
        )

        if "karsilastirma_acik" not in st.session_state:
            st.session_state["karsilastirma_acik"] = False

        if st.button("⚙️ Karşılaştırma Tablolarını Hesapla / Göster (Rüzgar 48s & 24s + Güneş 48s)", type="primary"):
            st.session_state["karsilastirma_acik"] = True

        if st.session_state["karsilastirma_acik"]:
            r48 = analiz_getir("ruzgar", 48)
            r24 = analiz_getir("ruzgar", 24)
            g48 = analiz_getir("gunes", 48)

            st.markdown("#### 1) Ana Bulgular — Rüzgar (≥ 48 Saat Gecikme)")
            st.dataframe(r48["tablo1_performans"], use_container_width=True, hide_index=True)

            st.markdown("#### 2) Ek Tablo — Rüzgar (≥ 24 Saat Gecikme Karşılaştırması)")
            st.dataframe(r24["tablo1_performans"], use_container_width=True, hide_index=True)

            st.markdown("#### 3) Ayrı Kaynak Değerlendirmesi — Güneş (≥ 48 Saat Gecikme)")
            st.caption(
                "*Dipnot:* EPİAŞ güneş üretim serisi ağırlıklı olarak lisanslı santralleri kapsamakta olup "
                "lisanssız GES üretimi net yükten düşmektedir; gece saatlerindeki küçük negatif gözlemler iç tüketim kaynaklıdır."
            )
            st.dataframe(g48["tablo1_performans"], use_container_width=True, hide_index=True)
        else:
            st.info("İlk açılışı hızlandırmak için 24s ve Güneş karşılaştırma modelleri yalnızca yukarıdaki butona bastığınızda eğitilir.")

    # ------------------------------------------------------------------
    # SEKME 5: 300 DPI Akademik Grafikler & Toplu ZIP Paket İndirme
    # ------------------------------------------------------------------
    with sekme5:
        st.subheader(f"🖼️ 300 DPI Akademik Grafikler ({URETIM_SECENEKLERI[hedef_sutun]} — {min_gecikme}s)")

        if "grafikler_acik" not in st.session_state:
            st.session_state["grafikler_acik"] = False

        col_zip1, col_zip2 = st.columns([0.55, 0.45])
        with col_zip1:
            if st.button("🎨 8 Adet 300 DPI Grafiği Üret ve Göster", type="primary"):
                st.session_state["grafikler_acik"] = True
        with col_zip2:
            if st.button("📦 Tam Rapor ZIP Paketi Oluştur (Rüzgar 48s+24s + Güneş 48s)"):
                zip_bytes = tum_rapor_paketi_zip(
                    analiz_getir("ruzgar", 48),
                    analiz_getir("ruzgar", 24),
                    analiz_getir("gunes", 48),
                )
                st.download_button(
                    label="⬇️ ASIMOPT_Rapor_Paketi_300DPI.zip İndir",
                    data=zip_bytes,
                    file_name="ASIMOPT_Rapor_Paketi_300DPI.zip",
                    mime="application/zip",
                )

        if st.session_state["grafikler_acik"]:
            grafikler = grafik_paketi_getir(hedef_sutun, min_gecikme)
            st.divider()
            dosya_adlari = list(grafikler.keys())
            for i in range(0, len(dosya_adlari), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(dosya_adlari):
                        dosya_adi = dosya_adlari[i + j]
                        baslik, png_bytes = grafikler[dosya_adi]
                        with cols[j]:
                            st.markdown(f"**{baslik}**")
                            st.image(png_bytes, use_container_width=True)
                            st.download_button(
                                label=f"⬇️ {dosya_adi} (300 DPI PNG)",
                                data=png_bytes,
                                file_name=f"{hedef_sutun}_{min_gecikme}s_{dosya_adi}",
                                mime="image/png",
                                key=f"dl_{hedef_sutun}_{min_gecikme}_{dosya_adi}",
                            )
        else:
            st.info("8 adet 300 DPI PNG grafiği görmek için yukarıdaki **'8 Adet 300 DPI Grafiği Üret ve Göster'** butonuna basın.")


if __name__ == "__main__":
    main()
