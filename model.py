# -*- coding: utf-8 -*-
"""
model.py
========
ASIMOPT - Asimetrik Kayıp Fonksiyonu ve Kuantil Optimizasyon Modelleri.
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
    tau = dataset.get_weight()
    hata = y_true - y_pred
    grad = np.where(hata >= 0, -tau, (1.0 - tau))
    hess = np.ones_like(y_true) * 1.0
    return grad, hess


def dinamik_kuantil_sec(kuantil_matrisi, kuantil_seviyeleri, hedef_tau_dizisi):
    n_ornek = len(hedef_tau_dizisi)
    secilen_tahminler = np.zeros(n_ornek, dtype=float)
    hedef_tau = np.clip(hedef_tau_dizisi, kuantil_seviyeleri[0], kuantil_seviyeleri[-1])

    for i in range(n_ornek):
        secilen_tahminler[i] = np.interp(hedef_tau[i], kuantil_seviyeleri, kuantil_matrisi[i, :])
    return secilen_tahminler


def modelleri_egit_ve_tahmin_et(train_df, test_df, hedef_sutun="ruzgar"):
    uretim_cols, tau_cols = oznitelik_listeleri(hedef_sutun)
    uretim_cols = list(dict.fromkeys([c for c in uretim_cols if c in train_df.columns]))
    tau_cols = list(dict.fromkeys([c for c in tau_cols if c in train_df.columns]))

    X_train = train_df[uretim_cols].loc[:, ~train_df[uretim_cols].columns.duplicated()]
    y_train = train_df[hedef_sutun].values
    X_test = test_df[uretim_cols].loc[:, ~test_df[uretim_cols].columns.duplicated()]

    X_tau_train = train_df[tau_cols].loc[:, ~train_df[tau_cols].columns.duplicated()]
    X_tau_test = test_df[tau_cols].loc[:, ~test_df[tau_cols].columns.duplicated()]

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

    tau_ort_fazla = float(train_df.loc[yon_train_etiket == 1, "tau_gercek"].mean())
    tau_ort_acik = float(train_df.loc[yon_train_etiket == 0, "tau_gercek"].mean())

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

    tau_yon_beklenen = olas_enerji_fazlasi * tau_ort_fazla + (1.0 - olas_enerji_fazlasi) * tau_ort_acik
    tau_hat_test = np.clip(0.5 * tau_yon_beklenen + 0.5 * tau_reg_tahmin, 0.10, 0.90)
    tau_gercek_test = test_df["tau_gercek"].values

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

    kuantil_tahminleri = np.sort(kuantil_tahminleri, axis=1)

    pred_tau_03 = kuantil_tahminleri[:, 2]
    pred_tau_05 = kuantil_tahminleri[:, 4]
    pred_tau_07 = kuantil_tahminleri[:, 6]

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
