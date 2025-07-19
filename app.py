import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load data (download and place fake_job_postings.csv in the same folder)
df = pd.read_csv("fake_job_postings.csv")

# Drop missing descriptions or labels
df = df.dropna(subset=["description", "fraudulent"])

# Combine useful text fields
text_data = df["title"].fillna("") + " " + df["company_profile"].fillna("") + " " + df["description"].fillna("")
labels = df["fraudulent"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
clf = LogisticRegression(max_iter=300)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(clf, "fake_job_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Try sample prediction
def predict_job_post(text):
    vect = vectorizer.transform([text])
    pred = clf.predict(vect)[0]
    prob = clf.predict_proba(vect)[0][1]
    return "Suspicious" if pred == 1 else "Legit", prob

# Example use
sample_text = "Work from home! Make $500/day. No experience required. Click now!"
label, confidence = predict_job_post(sample_text)
print(f"Prediction: {label} (Confidence: {confidence:.2f})")
