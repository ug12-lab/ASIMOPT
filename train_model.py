import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def pinball_loss(y_true, y_pred, tau):
    """
    Asimetrik Kayıp Fonksiyonu (Pinball Loss)
    tau: 0 ile 1 arasında değişen, her satır için dinamik asimetri parametresi
    """
    error = y_true - y_pred
    return np.where(error >= 0, tau * error, (tau - 1) * error)

def custom_asymmetric_objective(y_true, y_pred, dataset):
    """
    LightGBM için özel asimetrik amaç fonksiyonu (Custom Objective)
    Dinamik 'tau' parametresini kullanır.
    """
    # Veri setine ağırlık (weight) olarak tau'yu vereceğiz ve oradan çekeceğiz
    tau = dataset.get_weight() 
    
    error = y_true - y_pred
    
    # Gradient hesaplama (Pinball loss türevi)
    # L = tau * error if error >= 0 else (tau - 1) * error
    # dL/dy_pred = -tau if error >= 0 else (1 - tau)
    grad = np.where(error >= 0, -tau, (1 - tau))
    
    # Hessian (İkinci türev)
    # Mutlak değer türevsiz olduğu için 0'a yakınsayan sabit bir sayı veriyoruz
    hess = np.ones_like(y_true) * 1e-4 
    
    return grad, hess

def train_asimopt_model(csv_path="epias_veri.csv"):
    if not os.path.exists(csv_path):
        print(f"HATA: {csv_path} bulunamadı. Lütfen CSV dosyanızı bu dizine ekleyin.")
        return

    print("1. Veri yükleniyor ve ön işleme yapılıyor...")
    df = pd.read_csv(csv_path)
    
    # Zaman kolonunu datetime yap
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
        df['Saat'] = df['ts'].dt.hour
        df['Ay'] = df['ts'].dt.month
        df['Haftanin_Gunu'] = df['ts'].dt.dayofweek
    
    # Eksik verileri doldur (özellikle tau)
    df = df.interpolate(method='linear', limit_direction='both')
    
    # Hedef Değişken (Rüzgar Üretimini tahmin ettiğimizi varsayalım)
    target_col = 'ruzgar' 
    
    # Modele girecek öznitelikler (Geleceği gören fiyatları değil, sadece zaman ve dışsal verileri alıyoruz)
    # Gerçek senaryoda buraya hava durumu (ruzgar_hizi) da eklenmelidir.
    features = ['Saat', 'Ay', 'Haftanin_Gunu']
    
    X = df[features]
    y = df[target_col]
    
    # ASIMOPT'un kalbi: Dinamik tau parametresi
    # Eğer tau yoka, varsayılan 0.5 verilir (Simetrik MAE gibi davranır)
    tau_values = df['tau'].values if 'tau' in df.columns else np.ones(len(df)) * 0.5

    # Train-Test Split (Zaman serisi olduğu için sırayı bozmadan bölüyoruz)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    tau_train, tau_test = tau_values[:split_idx], tau_values[split_idx:]

    print("2. LightGBM Veri Seti oluşturuluyor (Dinamik Tau weight olarak veriliyor)...")
    # Tau değerlerini modele weight üzerinden paslıyoruz ki custom objective içinde okuyabilelim
    train_data = lgb.Dataset(X_train, label=y_train, weight=tau_train)
    test_data = lgb.Dataset(X_test, label=y_test, weight=tau_test, reference=train_data)

    params = {
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbosity': -1,
        'seed': 42
    }

    print("3. Asimetrik Kayıp Fonksiyonu (ASIMOPT) ile Model Eğitiliyor...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, test_data],
        fobj=custom_asymmetric_objective,  # KENDİ YAZDIĞIMIZ PINBALL LOSS
    )

    print("4. Test seti üzerinde tahminler yapılıyor...")
    preds = model.predict(X_test)
    
    # Performans Ölçümü
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    asym_loss = np.mean(pinball_loss(y_test.values, preds, tau_test))
    
    print("\n--- MODEL SONUÇLARI ---")
    print(f"Klasik MAE (Simetrik Hata) : {mae:.2f}")
    print(f"ASIMOPT Kayıp (Pinball)    : {asym_loss:.2f}")
    print("------------------------")
    print("Harika! Modeliniz artık cezaların (Pi_eksik, Pi_fazla) durumuna göre dinamik olarak daha kârlı olan yöne doğru (fazla/eksik üretim) taraflı tahminler üretecek.")
    
    return model

if __name__ == "__main__":
    train_asimopt_model()
