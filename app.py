from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load dataset dan ekstrak data provinsi/kota
dataset = pd.read_csv("data_lele.csv")
# Membersihkan nama provinsi dan kota dari spasi ekstra dan kapitalisasi
dataset['Provinsi'] = dataset['Provinsi'].str.strip().str.upper()
dataset['Kabupaten/Kota'] = dataset['Kabupaten/Kota'].str.strip().str.upper()

# Filter provinsi JAWA TIMUR
print(dataset[dataset['Provinsi'] == 'JAWA TIMUR']['Kabupaten/Kota'].unique())

provinces = dataset['Provinsi'].unique()
cities_by_province = {
    province: dataset[dataset['Provinsi'] == province]['Kabupaten/Kota'].unique().tolist()
    for province in provinces
}

# Load model dan scaler
model = tf.keras.models.load_model('model_lele.h5')
scaler_X = joblib.load('scaler_lele_X.pkl')
scaler_y = joblib.load('scaler_lele_y.pkl')

@app.route('/get_regions', methods=['GET'])
def get_regions():
    return jsonify({
        "provinces": list(cities_by_province.keys()),
        "cities_by_province": cities_by_province
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        province = data['province'].strip().upper()  # Normalisasi input
        city = data['city'].strip().upper()  # Normalisasi input
        n_years = int(data['years'])

        if province not in cities_by_province:
            return jsonify({"error": f"Provinsi {province} tidak ditemukan."}), 400

        if city not in cities_by_province[province]:
            return jsonify({"error": f"Kota {city} tidak ditemukan dalam provinsi {province}."}), 400

        # Buat one-hot encoding untuk kota
        all_cities = [city for cities in cities_by_province.values() for city in cities]
        city_one_hot = np.zeros(len(all_cities))
        city_index = all_cities.index(city) if city in all_cities else -1
        if city_index != -1:
            city_one_hot[city_index] = 1

        # Menyiapkan data untuk model
        last_known_data = np.array([[1.0] * scaler_X.scale_.shape[0]])
        last_known_data[0, -len(city_one_hot):] = city_one_hot
        last_known_data = scaler_X.transform(last_known_data)

        # Prediksi menggunakan model
        future_predictions = predict_future(model, last_known_data, n_years, scaler_X, scaler_y)

        # Hitung rata-rata tahunan dari hasil prediksi bulanan
        yearly_averages = [float(np.mean(future_predictions[i * 12:(i + 1) * 12])) for i in range(n_years)]

        # Konversi prediksi dan rata-rata tahunan ke dalam bentuk list yang bisa diserialisasi ke JSON
        return jsonify({
            "monthly_predictions": future_predictions.tolist(),
            "yearly_averages": yearly_averages,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

def predict_future(model, last_known_data, n_years, scaler_X, scaler_y):
    predictions = []
    current_input = last_known_data.copy()

    for i in range(n_years * 12):  # Prediksi bulanan selama n_years
        pred = model.predict(current_input.reshape(1, 1, -1))  # Model prediksi
        predictions.append(pred[0][0])  # Menambahkan hasil prediksi
        current_input = np.roll(current_input, -1, axis=1)  # Geser data untuk input berikutnya
        current_input[0, -1] = pred  # Update dengan prediksi terbaru

    # Kembalikan hasil prediksi dengan unscale
    predictions_rescaled = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return np.where(predictions_rescaled < 0, 0, predictions_rescaled)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
