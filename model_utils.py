import joblib

def load_model_and_vectorizer():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

def predict_job_post(text):
    model, vectorizer = load_model_and_vectorizer()
    vect = vectorizer.transform([text])
    prediction = model.predict(vect)[0]
    confidence = model.predict_proba(vect)[0][1]
    label = "Suspicious" if prediction == 1 else "Legit"
    return label, confidence
