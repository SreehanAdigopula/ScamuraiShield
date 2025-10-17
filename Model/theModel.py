import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib
import os

# Hugging Face imports
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np

# Create Model directory if it doesn't exist
os.makedirs('Model', exist_ok=True)

# ---------- Load Data ----------
df = pd.read_csv(
    'Model/SMSSpamCollection.csv',
    sep='\t',
    header=None,
    names=['label', 'message'],
    encoding='latin-1'
)

print("Original label distribution:")
print(df['label'].value_counts())

# ---------- Preprocessing ----------
df = df.dropna(subset=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels
X = df['message']
y = df['label']

# ---------- Oversampling for Logistic Regression ----------
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X.values.reshape(-1, 1), y)

df_balanced = pd.DataFrame(X_resampled, columns=['message'])
df_balanced['label'] = y_resampled
df_balanced['message'] = df_balanced['message'].astype(str)
print("Balanced dataset distribution:")
print(df_balanced['label'].value_counts())

# ---------- Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['message'],
    df_balanced['label'],
    test_size=0.2,
    random_state=42,
    stratify=df_balanced['label']
)

# ========== BASELINE: TF-IDF + Logistic Regression ==========
print("\n===== Training Logistic Regression =====")
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train.fillna(''))
X_test_tfidf = vectorizer.transform(X_test.fillna(''))

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=500, random_state=42)
lr_model.fit(X_train_resampled, y_train_resampled)

# Evaluation
y_pred = lr_model.predict(X_test_tfidf)
print("\n===== Logistic Regression Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Save baseline model
joblib.dump(lr_model, 'Model/scam_lr_model.pkl')
joblib.dump(vectorizer, 'Model/scam_vectorizer.pkl')
print("\n✓ Saved Logistic Regression model to Model/scam_lr_model.pkl")
print("✓ Saved vectorizer to Model/scam_vectorizer.pkl")

# ========== ADVANCED: DistilBERT ==========
print("\n===== Training DistilBERT Model =====")

# Use original data (not oversampled)
X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Convert data into Hugging Face Dataset
train_dataset = Dataset.from_dict({"text": X_train_bert.tolist(), "label": y_train_bert.tolist()})
test_dataset = Dataset.from_dict({"text": X_test_bert.tolist(), "label": y_test_bert.tolist()})

# Tokenizer & Encoding
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load Model
bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",   # FIXED keyword
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,   # Debug fast run; increase to 3+ for real training
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Trainer
trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Evaluate
metrics = trainer.evaluate()
print("\n===== DistilBERT Results =====")
print(metrics)

# Save Model
bert_model.save_pretrained("Model/distilbert_spam_detector")
tokenizer.save_pretrained("Model/distilbert_spam_detector")
print("\n✓ Saved DistilBERT model to Model/distilbert_spam_detector")

print("\n✅ Training complete!")
