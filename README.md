Email/SMS Spam Detection System
A machine learning project that detects spam messages using Naive Bayes classifiers with TF-IDF vectorization, achieving a precision of 1.00 (100%) and accuracy of 97%.
🎯 Objective
To build a text classification system that accurately identifies spam emails and SMS messages using probabilistic machine learning, with a focus on minimizing false positives (marking legitimate messages as spam).
The Math Behind It
Naive Bayes — Bayes' Theorem
The classifier is based on:
Code
P(Spam | Message) = P(Message | Spam) × P(Spam) / P(Message)
where :
P(Spam) — Prior probability of spam in dataset
P(Message | Spam) — Likelihood of words appearing in spam
P(Spam | Message) — Posterior: final spam probability

TF-IDF Vectorization

Converts raw text into numerical features:
TF-IDF(word) = TF(word) × log(Total Docs / Docs containing word)
This rewards words that are unique to spam and penalizes common words.

Dataset

Source: UCI SMS Spam Collection Dataset (Kaggle)
Size: 5,572 messages
Classes: Ham (legitimate) and Spam
Distribution: ~87% Ham, ~13% Spam
⚙️ Tech Stack
Category
Tools
Language
Python
ML Library
Scikit-learn
NLP
NLTK
Data
Pandas, NumPy
Visualization
Matplotlib, Seaborn
Vectorization
TF-IDF
🔬 Models Compared
Three Naive Bayes variants were tested:
Model
Full Name
Best For
GNB
Gaussian Naive Bayes
Continuous features
MNB
Multinomial Naive Bayes
Word count / frequency data
BNB
Bernoulli Naive Bayes
Binary presence/absence of words
📊 Results & Comparison
Accuracy Scores
Model
Vectorizer
Accuracy
GNB
TF-IDF
~87%
BNB
TF-IDF
~97%
MNB
TF-IDF
97% ✅
Why MNB + TF-IDF Wins
MNB is designed for word frequency data — exactly what TF-IDF produces
Naturally handles the Bayes probability math on term frequencies
Achieves Precision = 1.00 meaning zero false positives — no legitimate message was ever marked as spam

Why Precision = 1.00 Matters
In spam detection, false positives are costly — you never want important emails marked as spam. A precision of 1.00 on spam means the model never wrongly flags a legitimate message, which is the most important real-world metric.
🗂️ Project Structure
Code
spam-detection/
│
├── data/
│   └── spam.csv
├── notebooks/
│   └── spam_detection.ipynb
├── model/
│   └── spam_model.pkl
│   └── tfidf_vectorizer.pkl
├── app.py
├── requirements.txt
└── README.md
kEY FINDING
MNB + TF-IDF is the best combination for text-based spam detection
Precision of 1.00 confirms zero false positives — the most critical metric for spam filters
GNB performed poorly because it assumes Gaussian distribution which doesn't suit word frequency data
BNB ignores word frequency (only presence/absence), making it less powerful than MNB for this task
🔮 Future Improvements:
Add deep learning model (LSTM) for comparison
Deploy as a web app using Flask
Experiment with Word2Vec / BERT embeddings
Handle multilingual spam detection
