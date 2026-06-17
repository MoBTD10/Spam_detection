# Email Spam-Ham Detection System using NLP & Random Forest

This repository features an end-to-end Natural Language Processing (NLP) pipeline designed to classify emails into **Spam** or **Ham (Not Spam)**. The project implements comprehensive text preprocessing and utilizes a **Random Forest Classifier** to achieve high-accuracy predictions.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **NLP & Text Processing:** NLTK (Natural Language Toolkit)
* **Feature Extraction:** Scikit-Learn (`CountVectorizer` / Bag of Words)
* **Machine Learning Model:** Scikit-Learn (`RandomForestClassifier`)
* **Data Manipulation:** Pandas & NumPy
* **Visualization:** Seaborn & Matplotlib

## ⚙️ Data Preprocessing & NLP Pipeline
To prepare the raw email text for the machine learning model, a robust cleaning pipeline was implemented using a custom `word_cleaner` function:
1. **Formatting Fixes:** Stripped out annoying newline characters (`\r\n`) to flatten the text into a clean string.
2. **Case Normalization:** Converted all text to lowercase to ensure consistency (e.g., "SPAM" and "spam" are treated identically).
3. **Punctuation Removal:** Eliminated all punctuation marks using Python's `string.punctuation` to focus purely on meaningful words.
4. **Stopwords Elimination:** Filtered out common English stopwords (like "the", "is", "at") using `NLTK` to remove noise from the dataset.
5. **Stemming:** Applied the **PorterStemmer** to reduce words to their base or root form (e.g., "running", "runs" -> "run"), which significantly optimizes the feature vocabulary.

## 📊 Feature Extraction & Modeling
* **Vectorization:** Transformed the cleaned text corpus into numerical vectors using the **Bag of Words (BoW)** model via `CountVectorizer`.
* **Data Splitting:** Divided the dataset into an 80/20 split for training and testing to guarantee proper evaluation.
* **Classifier:** Trained a **Random Forest Classifier** utilizing parallel processing (`n_jobs=-1`) to maximize training speed and efficiency across CPU cores.
* **Evaluation:** Generated a Confusion Matrix to thoroughly inspect False Positives and False Negatives beyond just standard accuracy scores.

## 🔮 Inference & Real-Time Testing
The pipeline includes a built-in inference step capable of taking a raw, unseen email string, passing it through the exact same preprocessing and vectorization pipeline, and outputting a live prediction (`Spam` or `Not Spam`).
