from flask import Flask, request, render_template
import joblib
import pandas as pd
from urllib.parse import urlparse
import re
import json

app = Flask(__name__)

# --- Load All Three Artifacts ---
try:
    model = joblib.load('final_phishing_detection_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('columns.json', 'r') as f:
        MODEL_FEATURES = json.load(f)
    print("✅ Model, scaler, and feature list loaded successfully.")
except Exception as e:
    model, scaler, MODEL_FEATURES = None, None, []
    print(f"❌ Error loading artifacts: {e}")

def extract_features(url: str) -> dict:
    features = {key: 0 for key in MODEL_FEATURES}
    if not re.match(r'http(s)?://', url):
        url = 'http://' + url
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ''
    
    features.update({
        'ip': 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 0,
        'nb_dots': url.count('.'),
        'nb_hyphens': hostname.count('-'),
        'https_token': 1 if parsed_url.scheme == 'https' else 0,
        'ratio_digits_url': sum(c.isdigit() for c in url) / len(url) if url else 0,
        'ratio_digits_host': sum(c.isdigit() for c in hostname) / len(hostname) if hostname else 0,
        'prefix_suffix': 1 if '-' in hostname else 0,
        'nb_subdomains': hostname.count('.'),
        'domain_age': -1,
        'google_index': 0,
        'page_rank': 0,
    })
    # Return a dictionary with values in the canonical order
    return {key: features.get(key, 0) for key in MODEL_FEATURES}

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text, url_checked = "", ""
    if request.method == 'POST':
        if all([model, scaler, MODEL_FEATURES]):
            try:
                url_to_check = request.form['url']
                features_dict = extract_features(url_to_check)
                data_df = pd.DataFrame([features_dict])[MODEL_FEATURES]
                scaled_data = scaler.transform(data_df)
                prediction = model.predict(scaled_data)
                result = 'Phishing' if prediction[0] == 1 else 'Legitimate'
                prediction_text = f'The website is predicted to be: {result}'
                url_checked = url_to_check
            except Exception as e:
                prediction_text = f'Error during prediction: {e}'
    return render_template('index.html', prediction_text=prediction_text, url_checked=url_checked)

if __name__ == '__main__':
    app.run(debug=True)