from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Model kustom
class MatrixFactorization(tf.keras.Model):
    def __init__(self, n_users, n_places, embedding_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.n_users = n_users
        self.n_places = n_places
        self.embedding_dim = embedding_dim

        self.user_embedding = tf.keras.layers.Embedding(n_users, embedding_dim)
        self.place_embedding = tf.keras.layers.Embedding(n_places, embedding_dim)

    def call(self, inputs):
        user_vec = self.user_embedding(inputs[:, 0])
        place_vec = self.place_embedding(inputs[:, 1])
        return tf.reduce_sum(user_vec * place_vec, axis=1)

    def get_config(self):
        return {
            "n_users": self.n_users,
            "n_places": self.n_places,
            "embedding_dim": self.embedding_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load model
model = load_model("model_rekomendasi.keras", custom_objects={"MatrixFactorization": MatrixFactorization})

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or "user_id" not in data or "candidates" not in data:
        return jsonify({"error": "Harap kirim 'user_id' dan 'candidates'"}), 400

    user_id = data["user_id"]
    candidates = data["candidates"]  # List of place IDs (int)

    input_data = np.array([[user_id, place_id] for place_id in candidates])
    predictions = model.predict(input_data, verbose=0)

    # Ambil 3 tempat dengan skor prediksi tertinggi
    top_indices = predictions.argsort()[-3:][::-1]
    rekomendasi = [candidates[i] for i in top_indices]

    return jsonify({
        "user_id": user_id,
        "rekomendasi_tempat_id": rekomendasi,
        "scores": [float(predictions[i]) for i in top_indices]
    })
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "API Rekomendasi Tempat sudah berjalan.",
        "usage": {
            "POST /predict": {
                "body": {
                    "user_id": "int",
                    "candidates": "[list of int]"
                }
            }
        }
    })
if __name__ == "__main__":
    app.run(debug=True)
