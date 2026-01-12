
import pandas as pd
import numpy as np
from faker import Faker
import random
import warnings
import os

warnings.filterwarnings("ignore")

fake = Faker('tr_TR')

np.random.seed(42)
random.seed(42)
Faker.seed(42)

def veri_uret(satir_sayisi=1000):
    print(f"{satir_sayisi} adet öğrenci verisi üretiliyor.")
    
    data = []
    
    for _ in range(satir_sayisi):
        ad_soyad = fake.name()
        
        # 1. Önceki Yoklama Oranı (%0 - %100)
        onceki_yoklama_orani = np.random.randint(50, 100)
        
        # 2. Dönem İçi Başarı Notu (0-100)
        donem_ici_basari_notu = np.random.randint(20, 100)
        
        # 3. Ders Katılımı (1-5 arası puan)
        ders_katilimi = np.random.randint(1, 6)
        
        # Notu düşük ve yoklaması düşükse -> RİSKLİ
        risk_score = 0
        if onceki_yoklama_orani < 70:
            risk_score += 4
        if donem_ici_basari_notu < 50:
            risk_score += 3
        if ders_katilimi <= 2:
            risk_score += 2
            
        risk_score += np.random.normal(0, 1)
        
        if risk_score > 4:
            risk_durumu = "Riskli"
        else:
            risk_durumu = "Düşük Riskli"
        
        data.append([ad_soyad, onceki_yoklama_orani, donem_ici_basari_notu, ders_katilimi, risk_durumu])
        
    df = pd.DataFrame(data, columns=['Ad_Soyad', 'Onceki_Yoklama_Orani', 'Donem_Ici_Basari_Notu', 'Ders_Katilimi', 'Risk_Durumu'])

    return df

if __name__ == "__main__":
    df_ogrenci = veri_uret()
    
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dosya_adi = os.path.join(current_dir, "ogrenci_risk_verisi.csv")
    
    df_ogrenci.to_csv(dosya_adi, index=False)
    
    print(f"\nVeri seti başarıyla oluşturuldu ve şu konuma kaydedildi:\n{dosya_adi}")
    print("\nVeriden İlk 5 Satır:")
    print(df_ogrenci.head())
    print("\nSınıf Dağılımı:")
    print(df_ogrenci['Risk_Durumu'].value_counts())
