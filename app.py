from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

app = Flask(__name__)

# Cek apakah file model ada
model_path = 'model/kmeans_model.pkl'
scaler_path = 'model/scaler.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print("File model belum ada. Silakan jalankan train_model.py terlebih dahulu!")
    exit()

# Load model dan scaler
with open(model_path, 'rb') as f:
    kmeans = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

def get_scaled_data():
    df = pd.read_csv('smartphones.csv')
    df = df[['RAM', 'Storage', 'Final Price']].dropna()
    df['RAM'] = pd.to_numeric(df['RAM'], errors='coerce')
    df['Storage'] = pd.to_numeric(df['Storage'], errors='coerce')
    df['Final Price'] = pd.to_numeric(df['Final Price'], errors='coerce')
    df = df.dropna()
    data_scaled = scaler.transform(df)
    return data_scaled, df

@app.route('/')
def index():
    cluster_result = request.args.get('cluster_result')
    ram = request.args.get('ram')
    storage = request.args.get('storage')
    price = request.args.get('final_price')
    return render_template('index.html', cluster_result=cluster_result, ram=ram, storage=storage, price=price)

@app.route('/elbow')
def elbow():
    data_scaled, _ = get_scaled_data()
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(6,4))
    plt.plot(k_range, inertia, marker='o')
    plt.xlabel('Jumlah Cluster (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid()
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return jsonify({'plot_url': plot_url})

@app.route('/silhouette')
def silhouette():
    data_scaled, _ = get_scaled_data()
    scores = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels)
        scores.append(score)
    plt.figure(figsize=(6,4))
    plt.plot(list(k_range), scores, marker='o', color='orange')
    plt.xlabel('Jumlah Cluster (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Jumlah Cluster')
    plt.grid()
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return jsonify({'plot_url': plot_url})

@app.route('/manual', methods=['POST'])
def manual():
    try:
        ram = request.form.get('ram')
        storage = request.form.get('storage')
        price = request.form.get('final_price')
        if not ram or not storage or not price:
            return redirect(url_for('index'))
        df = pd.DataFrame([[ram, storage, price]], columns=['RAM', 'Storage', 'Final Price'])
        df['RAM'] = pd.to_numeric(df['RAM'], errors='coerce')
        df['Storage'] = pd.to_numeric(df['Storage'], errors='coerce')
        df['Final Price'] = pd.to_numeric(df['Final Price'], errors='coerce')
        if df.isnull().any().any():
            return redirect(url_for('index'))
        data_scaled = scaler.transform(df)
        cluster = kmeans.predict(data_scaled)[0]
        # Redirect ke index dengan hasil cluster
        return redirect(url_for('index', cluster_result=cluster, ram=ram, storage=storage, final_price=price))
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        try:
            df = pd.read_csv(file)
            df = df[['RAM', 'Storage', 'Final Price']]
            df = df.dropna()
            df['RAM'] = pd.to_numeric(df['RAM'], errors='coerce')
            df['Storage'] = pd.to_numeric(df['Storage'], errors='coerce')
            df['Final Price'] = pd.to_numeric(df['Final Price'], errors='coerce')
            df = df.dropna()
            data_scaled = scaler.transform(df)
            clusters = kmeans.predict(data_scaled)
            df['Cluster'] = clusters
            # Sortir berdasarkan kolom Cluster
            df = df.sort_values(by='Cluster').reset_index(drop=True)
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(df['RAM'], df['Storage'], c=df['Cluster'], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel('RAM')
            plt.ylabel('Storage')
            plt.title('Hasil Clustering K-Means')
            plt.grid(True)
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            table_html = df.to_html(classes='table table-striped', index=False)
            return render_template('result.html', table=table_html, plot_url=plot_url)
        except Exception as e:
            return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
