from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix

# --- INITIALIZATION ---
app = Flask(__name__)
CORS(app) 

# --- LOAD MODELS AND DATA ---
print("Loading models and data... Please wait.")
try:
    with open('drug_encoder.pkl', 'rb') as f:
        drug_encoder = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    
    final_df = pd.read_csv('final_df_for_lookup.csv')
    print("Models and data loaded successfully!")
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not load a required file: {e}")
    exit()

# --- DEFINE CLUSTER THEMES & RISK ---
cluster_themes = {
    0: "Metabolism Interference (Decreased)", 1: "Serotonergic Effects (Increased Risk)",
    2: "General Adverse Effects (Increased Severity)", 3: "Cardiovascular Risk (QTc Prolongation)",
    4: "Antihypertensive Activity Interference", 5: "Metabolism Interference (Increased)",
    6: "CNS Depressant Effects (Increased Risk)", 7: "Bleeding Risk",
    8: "Reduced Therapeutic Efficacy", 9: "General Drug Concentration Increase"
}
cluster_risk_levels = {
    0: "Medium", 1: "High", 2: "Medium", 3: "High", 4: "Medium",
    5: "Medium", 6: "High", 7: "High", 8: "Low", 9: "Medium"
}

# --- API ENDPOINT: Get All Drugs ---
@app.route('/drugs', methods=['GET'])
def get_drugs():
    """Returns the full list of known drugs for autocomplete."""
    drug_list = drug_encoder.classes_.tolist()
    return jsonify(drug_list)

# --- API ENDPOINT: Predict Interaction ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    drug_a = data.get('drug_a', '').lower().strip()
    drug_b = data.get('drug_b', '').lower().strip()

    if not drug_a or not drug_b:
        return jsonify({'error': 'Please provide both drug names.'}), 400

    known_drugs = drug_encoder.classes_
    if drug_a not in known_drugs or drug_b not in known_drugs:
        return jsonify({'error': 'One or both drugs are not in the dataset.'}), 404

    drug_a_encoded = drug_encoder.transform([drug_a])
    drug_b_encoded = drug_encoder.transform([drug_b])
    
    drug_features = np.array([drug_a_encoded[0], drug_b_encoded[0]]).reshape(1, -1)
    drug_features_sparse = csr_matrix(drug_features)
    
    num_text_features = len(tfidf_vectorizer.get_feature_names_out())
    text_features_sparse = csr_matrix((1, num_text_features))
    
    new_vector = hstack([drug_features_sparse, text_features_sparse])

    predicted_cluster = kmeans_model.predict(new_vector)[0]
    distances = kmeans_model.transform(new_vector)[0]
    sorted_distances = np.sort(distances)
    confidence = (1 - (sorted_distances[0] / sorted_distances[1])) * 100

    example_desc = "No direct interaction description found in the dataset for this pair."
    match = final_df[((final_df['drug1'] == drug_a) & (final_df['drug2'] == drug_b)) |
                     ((final_df['drug1'] == drug_b) & (final_df['drug2'] == drug_a))]
    if not match.empty:
        example_desc = match['interaction'].iloc[0]

    result = {
        'cluster_id': int(predicted_cluster),
        'theme': cluster_themes.get(int(predicted_cluster), "Unknown Theme"),
        'risk_level': cluster_risk_levels.get(int(predicted_cluster), "Unknown"),
        'confidence': f"{confidence:.2f}",
        'example_description': example_desc.capitalize()
    }
    return jsonify(result)

# --- RUN THE APP ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
