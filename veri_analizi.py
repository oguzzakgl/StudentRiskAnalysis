
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

def veri_analizi_ve_model():
    print("--- 1. VERİ YÜKLEME ---")
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dosya_yolu = os.path.join(current_dir, "ogrenci_risk_verisi.csv")

    try:
        df = pd.read_csv(dosya_yolu)
    except FileNotFoundError:
        print(f"HATA: '{dosya_yolu}' bulunamadı. Önce veri_uretme.py çalıştırın.")
        return

    print(f"Veri Seti Boyutu: {df.shape}")
    
    print("\n--- 2. VERİ ÖN İŞLEME ---")
    # Riskli -> 1, Düşük Riskli -> 0
    le = LabelEncoder()
    df['Risk_Label'] = le.fit_transform(df['Risk_Durumu'])
    
    print("Sınıf dağılımı:")
    print(df['Risk_Durumu'].value_counts())
    
    # -------------------------------------------------------------
    print("\n--- 3. GÖRSELLEŞTİRME (Scatter Plot) ---")
    # -------------------------------------------------------------
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 7))
    
    sns.scatterplot(
        x='Onceki_Yoklama_Orani', 
        y='Donem_Ici_Basari_Notu', 
        hue='Risk_Durumu', 
        style='Risk_Durumu',
        data=df, 
        palette={'Riskli': '#FF4B4B', 'Düşük Riskli': '#4B4BFF'},
        s=100,
        alpha=0.8,
        edgecolor='w'
    )
    
    plt.title('Öğrenci Risk Analizi: Yoklama vs Başarı Notu', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Önceki Yoklama Oranı (%)', fontsize=12)
    plt.ylabel('Dönem İçi Başarı Notu', fontsize=12)
    
    plt.axvline(x=70, color='gray', linestyle='--', alpha=0.5) 
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    plt.fill_between([0, 70], 0, 100, color='#FF4B4B', alpha=0.1, label='Yüksek Risk Bölgesi (Yoklama < 70)')
    plt.fill_between([0, 100], 0, 50, color='#FF4B4B', alpha=0.1)
    
    plt.legend(title='Risk Durumu', title_fontsize='11', loc='upper left', frameon=True)
    plt.tight_layout()
    
    png_yolu = os.path.join(current_dir, '01_risk_dagilim_grafigi.png')
    plt.savefig(png_yolu, dpi=300)
    plt.close()

    
    print("\n--- 4. MAKİNE ÖĞRENMESİ MODELİ (LOGISTIC REGRESSION) ---")
    
    X = df[['Onceki_Yoklama_Orani', 'Donem_Ici_Basari_Notu', 'Ders_Katilimi']]
    y = df['Risk_Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nModel: Logistic Regression")
    print(f"Doğruluk (Accuracy): {acc:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    veri_analizi_ve_model()
