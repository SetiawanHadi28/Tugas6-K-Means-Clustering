import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Baca dataset
df = pd.read_csv('smartphones.csv')

# 2. Ambil kolom yang relevan dan hapus data kosong
df = df[['RAM', 'Storage', 'Final Price']]
df = df.dropna()

# 3. Ubah ke tipe numerik (jika belum)
df['RAM'] = pd.to_numeric(df['RAM'], errors='coerce')
df['Storage'] = pd.to_numeric(df['Storage'], errors='coerce')
df['Final Price'] = pd.to_numeric(df['Final Price'], errors='coerce')
df = df.dropna()

# 4. Standarisasi data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# 5. Elbow Method untuk menentukan jumlah cluster optimal
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method untuk Menentukan Jumlah Cluster Optimal')
plt.grid()
plt.show()

# 6. Lakukan clustering dengan 10 cluster
kmeans = KMeans(n_clusters=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(data_scaled)

# 7. Visualisasi hasil clustering
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['RAM'], df['Storage'], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('RAM')
plt.ylabel('Storage')
plt.title('Hasil Clustering K-Means (10 Cluster)')
plt.grid()
plt.show() 