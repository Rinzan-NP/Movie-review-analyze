# ----------- Import Libraries -----------
import re
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# ----------- Download NLTK Resources -----------
nltk.download('stopwords')
nltk.download('wordnet')

# ----------- Load & Preprocess Data -----------
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    tokens = [t for t in text.split() if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)
df = pd.read_csv("IMDB_Dataset.csv")
df['review'] = df['review'].apply(preprocess)
df['y'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# ----------- Vectorize and Train Models -----------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['review'].values)
y = df['y'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
nb = MultinomialNB().fit(X_train, y_train)
bagging = BaggingClassifier(estimator=LogisticRegression(max_iter=1000),
                            n_estimators=25, bootstrap=True, random_state=42).fit(X_train, y_train)

# Voting Classifier (Ensemble)
voting = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('nb', MultinomialNB()),
        ('bagging', BaggingClassifier(estimator=LogisticRegression(max_iter=1000), n_estimators=10, bootstrap=True, random_state=42))
    ],
    voting='soft'
).fit(X_train, y_train)

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/tfidf.pkl")
joblib.dump(lr, "models/lr.pkl")
joblib.dump(nb, "models/nb.pkl")
joblib.dump(bagging, "models/bagging.pkl")
joblib.dump(voting, "models/voting.pkl")

# ----------- Plot Graphs -----------
models = {"Logistic": lr, "Naive Bayes": nb, "Bagging": bagging, "Voting": voting}
accuracies = {}
classification_reports = {}
auc_scores = {}
precision_scores = {}
recall_scores = {}
f1_scores = {}

print("\n" + "="*60)
print("MODEL PERFORMANCE EVALUATION")
print("="*60)

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"\n{name} Accuracy: {acc:.4f}")
    
    # Get classification report as dictionary
    report = classification_report(y_test, y_pred, output_dict=True)
    classification_reports[name] = report
    print(classification_report(y_test, y_pred))
    
    # Extract precision, recall, f1 for visualization
    precision_scores[name] = report['weighted avg']['precision']
    recall_scores[name] = report['weighted avg']['recall']
    f1_scores[name] = report['weighted avg']['f1-score']

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = auc(fpr, tpr)
    auc_scores[name] = auc_val

    # Individual ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], '--', color='red', alpha=0.5)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"{name} ROC Curve", fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"models/roc_{name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Save metrics to JSON file
import json
metrics_data = {
    "accuracies": accuracies,
    "auc_scores": auc_scores,
    "precision_scores": precision_scores,
    "recall_scores": recall_scores,
    "f1_scores": f1_scores,
    "classification_reports": classification_reports
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=2)

# 1. Accuracy Comparison Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(accuracies.keys(), accuracies.values(), 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison", fontsize=16, fontweight='bold')
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Models", fontsize=12)
plt.xticks(rotation=45)
# Add value labels on bars
for i, (name, acc) in enumerate(accuracies.items()):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("models/accuracy_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. AUC Scores Comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(auc_scores.keys(), auc_scores.values(),
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.ylim(0, 1)
plt.title("Model AUC Scores Comparison", fontsize=16, fontweight='bold')
plt.ylabel("AUC Score", fontsize=12)
plt.xlabel("Models", fontsize=12)
plt.xticks(rotation=45)
# Add value labels on bars
for i, (name, auc_val) in enumerate(auc_scores.items()):
    plt.text(i, auc_val + 0.01, f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("models/auc_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Precision, Recall, F1-Score Comparison
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Precision
bars1 = ax1.bar(precision_scores.keys(), precision_scores.values(),
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax1.set_title("Precision Comparison", fontsize=14, fontweight='bold')
ax1.set_ylabel("Precision", fontsize=12)
ax1.set_ylim(0, 1)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3, axis='y')
for i, (name, val) in enumerate(precision_scores.items()):
    ax1.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Recall
bars2 = ax2.bar(recall_scores.keys(), recall_scores.values(),
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax2.set_title("Recall Comparison", fontsize=14, fontweight='bold')
ax2.set_ylabel("Recall", fontsize=12)
ax2.set_ylim(0, 1)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')
for i, (name, val) in enumerate(recall_scores.items()):
    ax2.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# F1-Score
bars3 = ax3.bar(f1_scores.keys(), f1_scores.values(),
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax3.set_title("F1-Score Comparison", fontsize=14, fontweight='bold')
ax3.set_ylabel("F1-Score", fontsize=12)
ax3.set_ylim(0, 1)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3, axis='y')
for i, (name, val) in enumerate(f1_scores.items()):
    ax3.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("models/metrics_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Combined Performance Radar Chart
from math import pi
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Metrics for radar chart
metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score']
model_names = list(accuracies.keys())
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# Calculate angles for each metric
angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
angles += angles[:1]  # Complete the circle

for i, model in enumerate(model_names):
    values = [
        accuracies[model],
        auc_scores[model],
        precision_scores[model],
        recall_scores[model],
        f1_scores[model]
    ]
    values += values[:1]  # Complete the circle
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.set_title("Model Performance Radar Chart", size=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)
plt.tight_layout()
plt.savefig("models/performance_radar.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. All ROC Curves on One Plot
plt.figure(figsize=(10, 8))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for i, (name, model) in enumerate(models.items()):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})', 
             linewidth=2, color=colors[i])

plt.plot([0, 1], [0, 1], '--', color='red', alpha=0.5, label='Random Classifier')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves Comparison", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("models/all_roc_curves.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("GRAPHS GENERATED SUCCESSFULLY!")
print("="*60)
print("Generated Charts:")
print("1. accuracy_comparison.png - Model accuracy comparison")
print("2. auc_comparison.png - AUC scores comparison")
print("3. metrics_comparison.png - Precision, Recall, F1-Score comparison")
print("4. performance_radar.png - Combined performance radar chart")
print("5. all_roc_curves.png - All ROC curves on one plot")
print("6. Individual ROC curves for each model")
print("="*60)

# ----------- Flask App -----------
def create_app():
    app = Flask(__name__)
    
    # Load models and vectorizer
    vectorizer = joblib.load("models/tfidf.pkl")
    models_loaded = {
        "logistic": joblib.load("models/lr.pkl"),
        "naive_bayes": joblib.load("models/nb.pkl"),
        "bagging": joblib.load("models/bagging.pkl"),
        "voting": joblib.load("models/voting.pkl")
    }

    def preprocess_simple(text):
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
        return text

    @app.route("/")
    def index():
        return render_template('index.html')

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json()
        text = preprocess_simple(data["review"])
        X = vectorizer.transform([text])
        model = models_loaded[data["method"]]
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0,1]
        return jsonify({"label":"Positive" if pred==1 else "Negative","probability":float(prob)})

    @app.route("/metrics")
    def get_metrics():
        import json
        try:
            with open("models/metrics.json", "r") as f:
                metrics = json.load(f)
            return jsonify(metrics)
        except:
            return jsonify({"error": "Metrics not available"})
    
    return app

if __name__ == "__main__":
    from flask import send_from_directory
    import shutil

    # Serve saved images from /static directory
    if not os.path.exists("static"):
        os.makedirs("static")
    for fname in os.listdir("models"):
        if fname.endswith(".png"):
            shutil.copy(f"models/{fname}", f"static/{fname}")
    
    app = create_app()
    app.run(debug=True)
