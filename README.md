[ASIMOPT_Proje_Rapor_Rehberi.md](https://github.com/user-attachments/files/34805291/ASIMOPT_Proje_Rapor_Rehberi.md)
# ASIMOPT Proje ve Makale Rapor Yazım Rehberi

Bu rehber, **ASIMOPT** projesinin bitirme raporu ve akademik makale taslağını yazacak ekip arkadaşları için danışman değerlendirme notları (`ASIMOPT_Sonuc_Degerlendirmesi_ve_Rapor_Notu.pdf`) ve örnek referans makale yapısı (`Symmetry 2026 — Slice-Aware Adversarial Vulnerability Analysis...`) esas alınarak güncellenmiştir.

---

## 1. Güncel Metodolojik Çerçeve (Kod ile %100 Tutarlılık)

Rapor yazılırken aşağıdaki kesin teknik kararlar **istisnasız** kullanılmalıdır; eski taslaklardaki XGBoost / LSTM veya 2026 yılı ifadeleri kesinlikle temizlenmelidir:

1. **Veri Aralığı:** `birlesik_veri.csv` içinden **01 Ocak 2022 00:00 – 31 Aralık 2025 23:00** (`35.064` saat) kullanılmıştır. **01 Ocak 2026 tarihli 24 saatlik dilim**, yeni piyasa mekanizmasına geçiş sebebiyle homojenlik için dışarıda bırakılmıştır.
2. **Eğitim / Test Ayrımı ve Doğrulama Seti Notu:** Kronolojik **%80 Eğitim / %20 Test** ayrımı kullanılmıştır. Model hiperparametreleri (yön/tau için `n_estimators=150`; MSE, Custom Obj. ve kuantil modelleri için `n_estimators=200`; `learning_rate=0.05`, `num_leaves=31`, `n_jobs=1`, `random_state=42`) literatür ve standart başlangıç değerleriyle **sabit** tutulduğu için test seti üzerinde herhangi bir ayar yapılmamış, ayrıca bir **doğrulama (validation) seti** ayrılmamıştır. *Bu gerekçe Bölüm 3'te açıkça yazılmalıdır.*
3. **Ana Kaynak vs. Ek Kaynak:**
   - **Rüzgar (`ruzgar`)**: Ana çalışma serisidir; tüm ana tablolar, istatistiksel testler ve tartışmalar Rüzgar serisi üzerinden kurulur.
   - **Güneş (`gunes`)**: Ayrı bir **ek tablo** olarak raporlanır ve altına mutlaka şu **dipnot** eklenir: *"EPİAŞ Şeffaflık Platformu gerçekleşen güneş üretim verisi ağırlıklı olarak lisanslı (OSOS/telemetrili) santralleri kapsamakta olup lisanssız GES üretimi doğrudan net tüketimden düşmektedir; gece saatlerindeki küçük negatif okumalar ise santral iç tüketiminden kaynaklanmaktadır."*
4. **Öznitelik Gecikmesi (Veri Sızıntısı Önleme):**
   - **Ana Sonuç (`≥ 48 saat` gecikme):** Gün Öncesi Piyasası (GÖP) saat 12:00 kapı kapanışı gereği yarının 24 saati tahmin edilirken yalnızca 48 saat ve daha eski üretim/fiyat gecikmeleri kullanılır.
   - **Ek Tablo (`≥ 24 saat` gecikme):** 48s kısıtı gevşetildiğinde tahmin hatasının (MAE/RMSE) nasıl düştüğü, ancak asimetrik maliyet dinamiğinin korunup korunmadığını göstermek için **ek karşılaştırma tablosu** olarak sunulur.
5. **Tutarlı Kırpma Sınırı:** Tüm kodda ($\tau^*$ hedefi, $\hat{\tau}$ tahminleri, eğitim örnek ağırlıkları ve duyarlılık kaydırmaları) kırpma sınırı **`[0.05, 0.95]`** olarak sabitlenmiştir.
6. **Yöntem İsimlendirmesi (Doğru Ayrım):**
   - Kuantil ızgarasından $\hat{\tau}$ ile enterpolasyon yapan yönteme **"eğitime gömülü"** DENMEZ; doğrusu **"Saatlik tahmin edilen $\hat{\tau}$'a göre kuantil seçimi"**dir.
   - **Eğitime gömülü yöntem** ise ayrı bir model olan **5. Model (`custom_asymmetric_objective`)** yani `lgb.train` içinde saatlik $\tau_i^*$ örnek ağırlıklı özel pinball gradyanı/hessianı ile eğitilen modeldir.
   - Kullanılan algoritma ailesi **XGBoost veya LSTM değildir**; tüm regresyon/kuantil/özel kayıp modelleri **LightGBM (`4.5.0`)** tabanlıdır.

---

## 2. Önerilen 6 Bölümlük Akademik Makale / Rapor İskeleti (*Symmetry 2026* Yapısına Uygun)

### Bölüm 1: Giriş (*Introduction*)
- Türkiye elektrik piyasasında (EPİAŞ GÖP & DGP) yenilenebilir enerji üreticilerinin karşılaştığı çift taraflı dengesizlik cezası problemi ($\pi_{\text{eksik}} \neq \pi_{\text{fazla}}$).
- Klasik simetrik kayıp fonksiyonlarının (MSE / MAE) enerji açığı ağırlıklı fiyat rejimlerinde neden alt-optimal tekliflere yol açtığı.
- Çalışmanın **4 somut katkısı** (örnek şablon):
  1. 2022–2025 EPİAŞ PTF/SMF verileriyle GÖP kapı kapanışına sadık ($\ge 48$ saat gecikmeli) ve sızıntısız LightGBM tahmin hattı.
  2. Hem **eğitime gömülü özel asimetrik kayıp fonksiyonu** hem de **saatlik $\hat{\tau}$ tabanlı dinamik kuantil seçimi** dahil 7 farklı yaklaşımın sistematik ve Newey-West düzeltmeli Diebold-Mariano (DM) testi ile karşılaştırılması.
  3. Gerçekçi yön tahmini doğruluğu ile çoğunluk sınıfı tabanı (*Enerji Açığı*) arasındaki dar marjın **Tasarruf–Doğruluk Eğrisi** ve **başabaş doğruluk eşiği** üzerinden deneysel olarak açıklanması (sabit $\tau=0.70$'in gerçekçi dinamik seçimi neden geride bıraktığının nedensel analizi).
  4. Ceza katsayıları ($\pi_{\text{eksik}}, \pi_{\text{fazla}}$) ve $\tau$ sapmalarına karşı **$\pm\%10, \pm\%20$ duyarlılık analizi** ile Güneş enerjisi ek değerlendirmesi.

### Bölüm 2: İlgili Çalışmalar (*Related Work*) + Karşılaştırma Tablosu
Metin özetinin ardından **mutlaka** aşağıdaki sütunlara sahip bir **Literatür Karşılaştırma Tablosu** ekleyin:

| Çalışma | Piyasa / Veri Dönemi | Kaynak & Algoritma | Kayıp / Maliyet Yaklaşımı | Operasyonel Gecikme & Yön Analizi | İstatistiksel Test & Duyarlılık |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Zugno vd. (2013)** | Nord Pool / İskandinavya | Rüzgar — Stokastik / Oyun-Teorik Denge | Fiyat-duyarlı stratejik teklif | Teklif kapısı uyumlu; analitik | Karşılaştırmalı vaka analizi |
| **Dinler (2021)** | Türkiye EPİAŞ | Rüzgar / Piyasa — Kuantil / ML | İki aşamalı dengesizlik azaltımı | Saatlik piyasa tahmini | Temel hata metrikleri |
| **Akpınar vd. (2026)** | Türkiye EPİAŞ | Yenilenebilir — Güncel ML/DL | Enerji tahmini + maliyet | Güncel zaman serisi kıyaslaması | Standart test metrikleri |
| **ASIMOPT (Bu Çalışma)** | EPİAŞ (2022–2025, 35.064s) | Rüzgar (Ana) & Güneş (Ek) — **LightGBM** (7 Model) | Sabit Kuantil, **Özel Asimetrik Objective**, $\hat{\tau}$ Kuantil Seçimi & Oracle $\tau^*$ | **$\ge 48$s (Ana)** ve **$\ge 24$s (Ek)**; Karışıklık Matrisi + **Tasarruf–Doğruluk Eğrisi** | **Newey-West DM Testi** + **$\pm\%10, \pm\%20$ Duyarlılık** |

*(Not: Yukarıdaki yazarların tam başlık, dergi ve yıl bilgilerini bibliyografyadan kontrol edip kesinleştirin.)*

### Bölüm 3: Materyal ve Yöntem (*Materials and Methods*)
- **3.1 Veri Seti ve Ön İşleme:** 2022–2025 (35.064 saat), 1 Ocak 2026'nın çıkarılma nedeni, saatlik optimal asimetri oranı:
  $$\pi_{\text{fazla},t} = \max(\text{PTF}_t - 0.97 \cdot \min(\text{PTF}_t, \text{SMF}_t), 0), \quad \pi_{\text{eksik},t} = \max(1.03 \cdot \max(\text{PTF}_t, \text{SMF}_t) - \text{PTF}_t, 0)$$
  $$\tau_t^* = \operatorname{clip}\left(\frac{\pi_{\text{eksik},t}}{\pi_{\text{eksik},t} + \pi_{\text{fazla},t}},\ 0.05,\ 0.95\right)$$
- **3.2 Öznitelik Mühendisliği ve Sızıntı Önleme:** $\ge 48$ saat gecikme seti (`lag_48`, `lag_72`, `lag_168`, kayar 24s/168s istatistikler, takvim ve PTF/SMF spread/yön gecikmeleri) ile ek $\ge 24$ saat gecikme farkı.
- **3.3 Modeller ve Sabit Hiperparametreler:** 7 modelin tanımı + tüm modellerin LightGBM sabit hiperparametre tablosu + doğrulama seti kullanılmama gerekçesi.
- **3.4 Değerlendirme Metrikleri ve Deney Protokolleri:**
  - MAE, RMSE, Dinamik Pinball Kaybı, Toplam ve Birim (TL/MWh) Dengesizlik Maliyeti.
  - Newey-West ($h=24$) Diebold-Mariano testi ($H_0: \mathbb{E}[d_t] = 0$).
  - Tasarruf–Doğruluk Eğrisi (%50–%100 yapay yön bozma, 10 Monte Carlo tekrarı, başabaş doğruluk hesabı).
  - $\pm\%10, \pm\%20$ Duyarlılık analizi ($\pi_{\text{eksik}}, \pi_{\text{fazla}}, \tau$).

### Bölüm 4: Deneysel Bulgular (*Experimental Results*)
Streamlit arayüzündeki **Sekme 5** üzerinden veya `python tables.py && python figures.py` ile üreteceğiniz **6 Tablo** ve **8 Grafik (300 DPI PNG)** doğrudan bu bölüme yerleştirilir:
- **Tablo 1 & Grafik 1–3:** Rüzgar ($\ge 48$s) 7 model maliyet ve hata karşılaştırması.
- **Tablo 2 & Grafik 7:** Diebold-Mariano ($DM$ ve $p$-değeri) sonuçları.
- **Tablo 3–4 & Grafik 4–5:** Sistem yönüne göre maliyetler ve Karışıklık Matrisi (Model doğruluğu vs. Çoğunluk sınıfı tabanı).
- **Tablo 5 & Grafik 6:** Tasarruf–Doğruluk Eğrisi ve **Başabaş Yön Doğruluğu Eşiği**.
- **Tablo 6 & Grafik 8:** $\pm\%10, \pm\%20$ Duyarlılık Analizi sonuçları.
- **Ek Tablolar:** Rüzgar ($\ge 24$s) karşılaştırma tablosu ve Güneş ($\ge 48$s) kapsam dipnotlu tablo.

### Bölüm 5: Tartışma (*Discussion*)
- **Neden Sabit $\tau = 0.70$, Gerçekçi $\hat{\tau}$ Seçimini Yeniyor?** Karışıklık matrisindeki çoğunluk sınıfı etkisi ve Tasarruf–Doğruluk eğrisindeki başabaş eşiği referans alınarak sayılarla açıklanmalıdır.
- **48 Saat vs. 24 Saat Gecikme Etkisi:** Gerçekçi GÖP zaman çizelgesine geçildiğinde hata artışı ve maliyet duyarlılığı yorumlanmalıdır.
- **Güneş Verisi Kapsam Sınırı:** Lisanslı/lisanssız GES ayrımı tartışılmalıdır.

### Bölüm 6: Sonuç ve Gelecek Çalışmalar (*Conclusions*)
- Ana sayısal bulguların özeti, piyasa katılımcıları için operasyonel öneri ve gelecekte eksojen meteorolojik (NWP) verilerle yön sınıflandırıcı başarımının başabaş eşiğinin üzerine taşınması önerisi.
