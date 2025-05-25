import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Buat folder model jika belum ada
if not os.path.exists('model'):
    os.makedirs('model')

# Baca dataset
df = pd.read_csv('smartphones.csv')

# Ambil kolom yang relevan dan hapus data kosong
df = df[['RAM', 'Storage', 'Final Price']]
df = df.dropna()

# Ubah ke tipe numerik
df['RAM'] = pd.to_numeric(df['RAM'], errors='coerce')
df['Storage'] = pd.to_numeric(df['Storage'], errors='coerce')
df['Final Price'] = pd.to_numeric(df['Final Price'], errors='coerce')
df = df.dropna()

# Standarisasi data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Latih model K-Means
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(data_scaled)

# Simpan model dan scaler
with open('model/kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model K-Means dan Scaler berhasil disimpan di folder 'model'!") 