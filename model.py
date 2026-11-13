# -*- coding: utf-8 -*-
"""
model.py
========
ASIMOPT - Saatlik tau*'a Göre Kuantil Seçimi ve Asimetrik Kayıp Modelleri.

Karşılaştırılan 7 model:
1. Klasik Simetrik (MSE)
2. Sabit tau = 0.30 (Kuantil)
3. Sabit tau = 0.50 (Simetrik MAE / Medyan)
4. Sabit tau = 0.70 (Kuantil)
5. Eğitime Gömülü Asimetrik Kayıp (Custom Objective - LightGBM lgb.train)
6. Gerçekçi Kuantil Seçimi (Teklif anında tahmin edilen tau_hat ile)
7. Üst Sınır / Oracle (Gerçekleşmiş uzlaştırma fiyatlarından hesaplanan tau* ile saatlik kuantil seçimi)
"""

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix
from features import oznitelik_listeleri, TAU_MIN, TAU_MAX

KUANTIL_SEVIYELERI = np.array(
    [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
)


def pinball_loss(y_true, y_pred, tau):
    """Asimetrik Pinball Kayıp Fonksiyonu."""
    hata = y_true - y_pred
    return np.where(hata >= 0, tau * hata, (tau - 1.0) * hata)


def custom_asymmetric_objective(preds, dataset):
    """
    LightGBM (v4+) için dinamik tau ağırlıklı özel asimetrik amaç fonksiyonu.
    Eğitim sırasında gerçekleşmiş tau* değerini örnek ağırlığı üzerinden okuyarak
    birinci (grad) ve ikinci (hess) türevleri üretir.
    """
    y_true = dataset.get_label()
    tau = dataset.get_weight()
    hata = y_true - preds
    grad = np.where(hata >= 0, -tau, (1.0 - tau))
    hess = np.ones_like(y_true)
    return grad, hess


def dinamik_kuantil_sec(kuantil_matrisi, kuantil_seviyeleri, hedef_tau_dizisi):
    """
    Her saat t için hedef_tau_dizisi[t] değerine karşılık gelen koşullu üretim kuantilini
    (Q_y(tau_t | X_t)) doğrusal enterpolasyon ile seçer.
    """
    n_ornek = len(hedef_tau_dizisi)
    secilen_tahminler = np.zeros(n_ornek, dtype=float)
    hedef_tau = np.clip(hedef_tau_dizisi, kuantil_seviyeleri[0], kuantil_seviyeleri[-1])

    for i in range(n_ornek):
        secilen_tahminler[i] = np.interp(
            hedef_tau[i], kuantil_seviyeleri, kuantil_matrisi[i, :]
        )
    return secilen_tahminler


def modelleri_egit_ve_tahmin_et(train_df, test_df, hedef_sutun="ruzgar", min_gecikme=48):
    """
    Tüm karşılaştırma modellerini sabit hiperparametrelerle eğitir
    (hiperparametre araması yapılmadığı için doğrulama kümesi ayrılmamıştır)
    ve test seti tahminlerini döndürür.
    """
    uretim_cols, tau_cols = oznitelik_listeleri(hedef_sutun, min_gecikme=min_gecikme)
    uretim_cols = list(dict.fromkeys([c for c in uretim_cols if c in train_df.columns]))
    tau_cols = list(dict.fromkeys([c for c in tau_cols if c in train_df.columns]))

    X_train = train_df[uretim_cols].loc[:, ~train_df[uretim_cols].columns.duplicated()]
    y_train = train_df[hedef_sutun].values
    X_test = test_df[uretim_cols].loc[:, ~test_df[uretim_cols].columns.duplicated()]

    # -------------------------------------------------------------------------
    # ADIM 1: Teklif Anında Sistem Yönü ve tau_hat Tahmini (Gerçekçi Model İçin)
    # -------------------------------------------------------------------------
    X_tau_train = train_df[tau_cols].loc[:, ~train_df[tau_cols].columns.duplicated()]
    X_tau_test = test_df[tau_cols].loc[:, ~test_df[tau_cols].columns.duplicated()]

    # 1: Enerji Fazlası (tau > 0.5), 0: Enerji Açığı / Dengede (tau <= 0.5)
    yon_train_etiket = (train_df["tau_gercek"] > 0.5).astype(int).values
    yon_test_etiket = (test_df["tau_gercek"] > 0.5).astype(int).values

    yon_siniflandirici = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=1,
        random_state=42,
        verbose=-1,
    )
    yon_siniflandirici.fit(X_tau_train, yon_train_etiket)
    olas_enerji_fazlasi = yon_siniflandirici.predict_proba(X_tau_test)[:, 1]
    yon_tahmin_etiket = (olas_enerji_fazlasi >= 0.5).astype(int)

    yon_tahmin_dogrulugu = float(accuracy_score(yon_test_etiket, yon_tahmin_etiket))
    cogunluk_sinifi_dogrulugu = float(
        max(np.mean(yon_test_etiket == 0), np.mean(yon_test_etiket == 1))
    )
    karisiklik_matrisi = confusion_matrix(yon_test_etiket, yon_tahmin_etiket, labels=[0, 1])

    tau_ort_fazla = float(train_df.loc[yon_train_etiket == 1, "tau_gercek"].mean())
    tau_ort_acik = float(train_df.loc[yon_train_etiket == 0, "tau_gercek"].mean())

    tau_regresor = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=1,
        random_state=42,
        verbose=-1,
    )
    tau_regresor.fit(X_tau_train, train_df["tau_gercek"].values)
    tau_reg_tahmin = tau_regresor.predict(X_tau_test)

    tau_yon_beklenen = (
        olas_enerji_fazlasi * tau_ort_fazla
        + (1.0 - olas_enerji_fazlasi) * tau_ort_acik
    )
    tau_hat_test = np.clip(
        0.5 * tau_yon_beklenen + 0.5 * tau_reg_tahmin, TAU_MIN, TAU_MAX
    )
    tau_gercek_test = test_df["tau_gercek"].values

    # -------------------------------------------------------------------------
    # ADIM 2: Klasik Simetrik Model (MSE)
    # -------------------------------------------------------------------------
    mse_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=1,
        random_state=42,
        verbose=-1,
    )
    mse_model.fit(X_train, y_train)
    pred_mse = np.clip(mse_model.predict(X_test), 0.0, None)

    # -------------------------------------------------------------------------
    # ADIM 3: Eğitime Gömülü Özel Asimetrik Kayıp Modeli (Custom Objective)
    # -------------------------------------------------------------------------
    # Sayısal gradyan kararlılığı için eğitim hedefi z-skor ölçeğine alınır (Pinball loss ölçek-değişmezdir)
    y_mean = float(np.mean(y_train))
    y_std = max(float(np.std(y_train)), 1.0)
    y_train_scaled = (y_train - y_mean) / y_std
    tau_train = train_df["tau_gercek"].values

    custom_ds = lgb.Dataset(X_train, label=y_train_scaled, weight=tau_train)
    custom_params = {
        "objective": custom_asymmetric_objective,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "num_threads": 1,
        "seed": 42,
        "verbosity": -1,
    }
    custom_model = lgb.train(custom_params, custom_ds, num_boost_round=200)
    pred_custom = np.clip(custom_model.predict(X_test) * y_std + y_mean, 0.0, None)

    # -------------------------------------------------------------------------
    # ADIM 4: Kuantil Regresyon Ailesi (Sabit 0.3, 0.5, 0.7 ve Saatlik Kuantil Seçimi)
    # -------------------------------------------------------------------------
    kuantil_tahminleri = np.zeros((len(test_df), len(KUANTIL_SEVIYELERI)))

    for idx, q in enumerate(KUANTIL_SEVIYELERI):
        q_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=float(q),
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=1,
            random_state=42,
            verbose=-1,
        )
        q_model.fit(X_train, y_train)
        kuantil_tahminleri[:, idx] = np.clip(q_model.predict(X_test), 0.0, None)

    # Monotonluk koruması (quantile crossing önleme)
    kuantil_tahminleri = np.sort(kuantil_tahminleri, axis=1)
    q_index = {round(float(q), 2): i for i, q in enumerate(KUANTIL_SEVIYELERI)}

    pred_tau_03 = kuantil_tahminleri[:, q_index[0.30]]
    pred_tau_05 = kuantil_tahminleri[:, q_index[0.50]]
    pred_tau_07 = kuantil_tahminleri[:, q_index[0.70]]

    # -------------------------------------------------------------------------
    # ADIM 5: Saatlik Kuantil Seçimi (Gerçekçi tau_hat vs. Üst Sınır tau*)
    # -------------------------------------------------------------------------
    pred_asimopt_gercekci = dinamik_kuantil_sec(
        kuantil_tahminleri, KUANTIL_SEVIYELERI, tau_hat_test
    )
    pred_asimopt_ust_sinir = dinamik_kuantil_sec(
        kuantil_tahminleri, KUANTIL_SEVIYELERI, tau_gercek_test
    )

    tahmin_sozlugu = {
        "Klasik Simetrik (MSE)": pred_mse,
        "Sabit tau = 0.30 (Kuantil)": pred_tau_03,
        "Sabit tau = 0.50 (Simetrik MAE)": pred_tau_05,
        "Sabit tau = 0.70 (Kuantil)": pred_tau_07,
        "Eğitime Gömülü Özel Kayıp (Custom Obj.)": pred_custom,
        "Gerçekçi Kuantil Seçimi (Tahmin tau_hat)": pred_asimopt_gercekci,
        "Üst Sınır / Oracle (Gerçekleşmiş tau*)": pred_asimopt_ust_sinir,
    }

    ek_bilgiler = {
        "tau_hat_test": tau_hat_test,
        "tau_gercek_test": tau_gercek_test,
        "yon_test_etiket": yon_test_etiket,
        "yon_tahmin_etiket": yon_tahmin_etiket,
        "olas_enerji_fazlasi": olas_enerji_fazlasi,
        "yon_tahmin_dogrulugu": yon_tahmin_dogrulugu,
        "cogunluk_sinifi_dogrulugu": cogunluk_sinifi_dogrulugu,
        "karisiklik_matrisi": karisiklik_matrisi,
        "tau_ort_fazla": tau_ort_fazla,
        "tau_ort_acik": tau_ort_acik,
        "kuantil_tahminleri": kuantil_tahminleri,
        "kuantil_seviyeleri": KUANTIL_SEVIYELERI,
    }

    return tahmin_sozlugu, ek_bilgiler
