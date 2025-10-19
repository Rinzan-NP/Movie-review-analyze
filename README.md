# 🎬 Movie Review Sentiment Analysis

A comprehensive machine learning web application that analyzes the sentiment of movie reviews using multiple algorithms and ensemble methods. The application provides an interactive web interface with detailed performance metrics and visualizations.

## 🌟 Features

### **Machine Learning Models**
- **Logistic Regression**: Fast and interpretable linear model
- **Naive Bayes**: Probabilistic classifier with strong assumptions
- **Bagging Ensemble**: Bootstrap aggregating for improved accuracy
- **Voting Classifier**: Combines multiple models for best results

### **Web Interface**
- 🎨 **Modern UI**: Beautiful, responsive design with gradient backgrounds
- 📊 **Interactive Charts**: ROC curves and accuracy comparisons
- 📈 **Performance Metrics**: Real-time display of model statistics
- 🔍 **Live Analysis**: Test your own movie reviews
- 📱 **Mobile Responsive**: Works on all devices

### **Performance Analytics**
- **Accuracy Scores**: Percentage accuracy for each model
- **AUC Scores**: Area Under Curve metrics
- **Classification Reports**: Detailed precision, recall, F1-score
- **Visual Comparisons**: Side-by-side model performance

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Required Python packages (see requirements.txt)

### Installation

Download The Repo

   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (automatic on first run)
   - The application will automatically download required NLTK resources

4. **Prepare your dataset**
   - Place your `IMDB_Dataset.csv` file in the project root
   - The dataset should have columns: `review` and `sentiment`

### Running the Application

1. **Train the models and start the server**
   ```bash
   python sentimental.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:5000`
   - The application will automatically train models and start the web server

## 📁 Project Structure

```
sarvana-project/
├── sentimental.py              # Main application file
├── templates/
│   └── index.html              # Web interface template
├── models/                     # Generated model files
│   ├── tfidf.pkl              # TF-IDF vectorizer
│   ├── lr.pkl                 # Logistic Regression model
│   ├── nb.pkl                 # Naive Bayes model
│   ├── bagging.pkl            # Bagging model
│   ├── voting.pkl             # Voting Classifier model
│   ├── metrics.json           # Performance metrics
│   └── *.png                  # Generated charts
├── static/                     # Static files for web interface
│   └── *.png                  # Chart images
├── IMDB_Dataset.csv           # Training dataset
└── README.md                  # This file
```

## 🔧 Technical Details

### **Data Preprocessing**
- HTML tag removal
- Text normalization
- Stop word removal
- Lemmatization using WordNet
- TF-IDF vectorization (5000 features)

### **Model Architecture**
- **Logistic Regression**: Linear classifier with L2 regularization
- **Naive Bayes**: Multinomial implementation
- **Bagging**: 25 estimators with bootstrap sampling
- **Voting Classifier**: Soft voting ensemble of all models

### **Performance Metrics**
- **Accuracy**: Overall correct predictions
- **AUC**: Area Under ROC Curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 📊 Usage

### **Web Interface**

1. **Choose Analysis Method**
   - Select from 4 different algorithms
   - Each method has different strengths and characteristics

2. **Enter Movie Review**
   - Type or paste your movie review
   - Supports any length of text

3. **Get Results**
   - Instant sentiment prediction
   - Confidence score
   - Visual feedback with emojis

4. **View Performance Data**
   - Model accuracy comparisons
   - ROC curves for each algorithm
   - Detailed classification reports

### **API Endpoints**

- `GET /` - Main web interface
- `POST /predict` - Sentiment prediction API
- `GET /metrics` - Model performance metrics

### **API Usage Example**

```python
import requests

# Predict sentiment
response = requests.post('http://localhost:5000/predict', 
                        json={
                            'method': 'logistic',
                            'review': 'I loved this movie!'
                        })
result = response.json()
print(f"Sentiment: {result['label']}")
print(f"Confidence: {result['probability']:.3f}")
```

## 🎯 Model Performance

The application automatically evaluates all models and displays:

- **Accuracy Comparison**: Side-by-side accuracy scores
- **ROC Curves**: Visual performance comparison
- **Classification Reports**: Detailed metrics for each model
- **AUC Scores**: Area under curve for each algorithm

## 🛠️ Customization

### **Adding New Models**
1. Import your model in `sentimental.py`
2. Add to the models dictionary
3. Update the HTML template with new options

### **Modifying UI**
- Edit `templates/index.html` for interface changes
- CSS styles are embedded in the HTML file
- JavaScript handles dynamic content loading

### **Dataset Requirements**
Your CSV file should have:
- `review`: Text content of movie reviews
- `sentiment`: Labels ('positive' or 'negative')

## 📈 Performance Tips

- **Training Time**: Initial model training may take a few minutes
- **Memory Usage**: Large datasets may require more RAM
- **Model Persistence**: Trained models are saved and reused
- **Caching**: TF-IDF vectorizer is cached for faster predictions

## 🐛 Troubleshooting

### **Common Issues**

1. **NLTK Download Errors**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

2. **Missing Dataset**
   - Ensure `IMDB_Dataset.csv` is in the project root
   - Check column names match requirements

3. **Port Already in Use**
   - Change port in `app.run(debug=True, port=5001)`

4. **Model Loading Errors**
   - Delete `models/` folder and retrain
   - Check file permissions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Scikit-learn**: Machine learning algorithms
- **NLTK**: Natural language processing
- **Flask**: Web framework
- **Matplotlib**: Data visualization
- **IMDB Dataset**: Training data

---

**Happy Analyzing! 🎬✨**
