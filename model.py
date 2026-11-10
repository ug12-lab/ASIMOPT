# -*- coding: utf-8 -*-
"""
model.py
========
ASIMOPT - Asimetrik Kayıp Fonksiyonu ve Kuantil Optimizasyon Modelleri.

Hocanın geri bildirimi doğrultusunda karşılaştırılan 6 model:
1. Klasik Simetrik Model (MSE)
2. Sabit tau = 0.30 Modeli
3. Sabit tau = 0.50 Modeli (Simetrik MAE / Medyan)
4. Sabit tau = 0.70 Modeli
5. ASIMOPT Gerçekçi Model (Sistem yönünü önceden tahmin ederek tau_hat belirleyen model)
6. ASIMOPT Üst Sınır / Oracle Modeli (Gerçekleşmiş uzlaştırma fiyatlarından hesaplanan tau* ile)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from features import oznitelik_listeleri


def pinball_loss(y_true, y_pred, tau):
    """Asimetrik Pinball Kayıp Fonksiyonu."""
    hata = y_true - y_pred
    return np.where(hata >= 0, tau * hata, (tau - 1.0) * hata)


def custom_asymmetric_objective(y_true, y_pred, dataset):
    """
    LightGBM için dinamik tau ağırlıklı özel asimetrik amaç fonksiyonu (Custom Objective).
    Her saatin kendi tau değerine göre birinci (gradient) ve ikinci (hessian) türevleri üretir.
    """
    tau = dataset.get_weight()
    hata = y_true - y_pred
    grad = np.where(hata >= 0, -tau, (1.0 - tau))
    hess = np.ones_like(y_true) * 1.0
    return grad, hess


def dinamik_kuantil_sec(kuantil_matrisi, kuantil_seviyeleri, hedef_tau_dizisi):
    """
    Her saat t için hedef_tau_dizisi[t] değerine karşılık gelen koşullu üretim kuantilini
    (Q_y(tau_t | X_t)) doğrusal enterpolasyon ile hesaplar.
    """
    n_ornek = len(hedef_tau_dizisi)
    secilen_tahminler = np.zeros(n_ornek, dtype=float)
    hedef_tau = np.clip(hedef_tau_dizisi, kuantil_seviyeleri[0], kuantil_seviyeleri[-1])

    for i in range(n_ornek):
        secilen_tahminler[i] = np.interp(hedef_tau[i], kuantil_seviyeleri, kuantil_matrisi[i, :])
    return secilen_tahminler


def modelleri_egit_ve_tahmin_et(train_df, test_df, hedef_sutun="ruzgar"):
    """
    Tüm karşılaştırma modellerini eğitir ve test seti tahminlerini döndürür.
    """
    uretim_cols, tau_cols = oznitelik_listeleri(hedef_sutun)

    X_train = train_df[uretim_cols]
    y_train = train_df[hedef_sutun].values
    X_test = test_df[uretim_cols]

    # -------------------------------------------------------------------------
    # ADIM 1: Sistem Yönünü Önceden Tahmin Ederek tau_hat Belirleyen Gerçekçi Model
    # -------------------------------------------------------------------------
    # Sadece teklif anında (t-24 ve öncesi) bilinen veriler kullanılır (veri sızıntısı yoktur).
    X_tau_train = train_df[tau_cols]
    X_tau_test = test_df[tau_cols]

    # 1A. Sistem Yönü Sınıflandırması (1: Enerji Fazlası / tau > 0.5, 0: Enerji Açığı / tau <= 0.5)
    yon_train_etiket = (train_df["tau_gercek"] > 0.5).astype(int).values
    yon_test_etiket = (test_df["tau_gercek"] > 0.5).astype(int).values

    yon_siniflandirici = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    yon_siniflandirici.fit(X_tau_train, yon_train_etiket)
    olas_enerji_fazlasi = yon_siniflandirici.predict_proba(X_tau_test)[:, 1]
    yon_tahmin_dogrulugu = accuracy_score(yon_test_etiket, (olas_enerji_fazlasi >= 0.5).astype(int))

    # Eğitim setindeki Enerji Fazlası ve Enerji Açığı durumlarının ortalama tau değerleri
    tau_ort_fazla = float(train_df.loc[yon_train_etiket == 1, "tau_gercek"].mean())
    tau_ort_acik = float(train_df.loc[yon_train_etiket == 0, "tau_gercek"].mean())

    # 1B. Doğrudan Tau Regresyonu ile Sistem Yönü Olasılığının Birleştirilmesi
    tau_regresor = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    tau_regresor.fit(X_tau_train, train_df["tau_gercek"].values)
    tau_reg_tahmin = tau_regresor.predict(X_tau_test)

    # Sistem yönü olasılığından beklenen tau ve regresyon tahmininin hibrit birleşimi
    tau_yon_beklenen = olas_enerji_fazlasi * tau_ort_fazla + (1.0 - olas_enerji_fazlasi) * tau_ort_acik
    tau_hat_test = np.clip(0.5 * tau_yon_beklenen + 0.5 * tau_reg_tahmin, 0.10, 0.90)
    tau_gercek_test = test_df["tau_gercek"].values

    # -------------------------------------------------------------------------
    # ADIM 2: Klasik Simetrik Model (MSE)
    # -------------------------------------------------------------------------
    mse_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    mse_model.fit(X_train, y_train)
    pred_mse = np.clip(mse_model.predict(X_test), 0.0, None)

    # -------------------------------------------------------------------------
    # ADIM 3: Kuantil Regresyon Ailesi (Sabit 0.3, 0.5, 0.7 ve Dinamik ASIMOPT İçin)
    # -------------------------------------------------------------------------
    kuantil_seviyeleri = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    kuantil_tahminleri = np.zeros((len(test_df), len(kuantil_seviyeleri)))

    for idx, q in enumerate(kuantil_seviyeleri):
        q_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=float(q),
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        q_model.fit(X_train, y_train)
        kuantil_tahminleri[:, idx] = np.clip(q_model.predict(X_test), 0.0, None)

    # Kuantil sıralaması monotonluğunu garanti altına al (quantile crossing önleme)
    kuantil_tahminleri = np.sort(kuantil_tahminleri, axis=1)

    # Sabit tau modelleri (0.30, 0.50, 0.70)
    pred_tau_03 = kuantil_tahminleri[:, 2]  # alpha = 0.30
    pred_tau_05 = kuantil_tahminleri[:, 4]  # alpha = 0.50 (Simetrik MAE / Medyan)
    pred_tau_07 = kuantil_tahminleri[:, 6]  # alpha = 0.70

    # -------------------------------------------------------------------------
    # ADIM 4: ASIMOPT Modelleri (Gerçekçi Tahmin Edilen tau_hat vs Gerçekleşmiş tau* Üst Sınır)
    # -------------------------------------------------------------------------
    pred_asimopt_gercekci = dinamik_kuantil_sec(
        kuantil_tahminleri, kuantil_seviyeleri, tau_hat_test
    )

    pred_asimopt_ust_sinir = dinamik_kuantil_sec(
        kuantil_tahminleri, kuantil_seviyeleri, tau_gercek_test
    )

    tahmin_sozlugu = {
        "Klasik Simetrik (MSE)": pred_mse,
        "Sabit tau = 0.30": pred_tau_03,
        "Sabit tau = 0.50 (Simetrik MAE)": pred_tau_05,
        "Sabit tau = 0.70": pred_tau_07,
        "ASIMOPT Gerçekçi (Tahmin Edilen tau_hat)": pred_asimopt_gercekci,
        "ASIMOPT Üst Sınır / Oracle (Gerçekleşmiş tau*)": pred_asimopt_ust_sinir,
    }

    ek_bilgiler = {
        "tau_hat_test": tau_hat_test,
        "tau_gercek_test": tau_gercek_test,
        "yon_tahmin_dogrulugu": float(yon_tahmin_dogrulugu),
        "tau_ort_fazla": tau_ort_fazla,
        "tau_ort_acik": tau_ort_acik,
    }

    return tahmin_sozlugu, ek_bilgiler
