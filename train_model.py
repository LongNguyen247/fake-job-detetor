import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv("fake_job_postings.csv")
df = df.dropna(subset=["description", "fraudulent"])

# Prepare text and labels
text_data = df["title"].fillna("") + " " + df["company_profile"].fillna("") + " " + df["description"].fillna("")
labels = df["fraudulent"]

# Split
X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)   
X_test_vec = vectorizer.transform(X_test)

# Model
clf = LogisticRegression(max_iter=300)
clf.fit(X_train_vec, y_train)

# Evaluation
y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save artifacts
joblib.dump(clf, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
