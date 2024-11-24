import os
import email
from email.policy import default
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import string

def read_eml_files(folder_path):
    emails = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.eml'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                msg = email.message_from_file(file, policy=default)
                # Törzs (szöveg) kinyerése
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if content_type == "text/plain":
                            emails.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
                else:
                    emails.append(msg.get_payload(decode=True).decode('utf-8', errors='ignore'))
    return emails


def extract_features(email_text):
    features = {}
    # Szóismétlés
    words = email_text.split()
    word_counts = pd.Series(words).value_counts()
    features['most_common_word_freq'] = word_counts.max() / len(words) if len(words) > 0 else 0

    # Reklám kulcsszavak száma
    keywords = ['akció', 'nyeremény', 'kedvezmény', 'ajándék','játék','különleges','leárazás',"ajánlat","kedvező","nyereményjáték","giveaway","prize","discount","sale","free","ingyenes","eladó","participate"]
    features['keyword_count'] = sum(1 for word in words if word in keywords)

    # E-mail hossza
    features['email_length'] = len(email_text)

    # Több címzett van-e
    if "To:" in email_text:
        to_section = email_text.split("To:")[1]
        features['multiple_recipients'] = int("," in to_section)
    else:
        features['multiple_recipients'] = 0  # Ha nincs "To:" fejléc

    # HTML tartalom van-e
    features['has_html'] = int('<html>' in email_text.lower())

    # Szófajgyakoriság - szavak száma
    features['word_count'] = len(words)

    # Linkek száma
    features['link_count'] = email_text.count('http')

    # Speciális karakterek száma
    special_chars = set(string.punctuation)
    features['special_char_count'] = sum(1 for char in email_text if char in special_chars)

    # Melléklet van-e
    features['has_attachment'] = int("Content-Disposition: attachment" in email_text)

    # Saját paraméterek
    features['unique_domains'] = len(set(['example.com', 'test.com']))  # Tesztérték
    features['send_time_hour'] = 12  # Példaérték
    features['image_count'] = email_text.count('<img')

    return features


def create_dataset_from_eml(folder_path, is_spam):
    emails = read_eml_files(folder_path)
    data = []
    for email_text in emails:
        features = extract_features(email_text)
        features['label'] = int(is_spam)  # 0: Nem spam, 1: Spam
        data.append(features)
    return pd.DataFrame(data)

"""
# Spam és nem spam e-mailek mappáinak megadása
spam_folder = 'path/to/spam_emails'
not_spam_folder = 'path/to/not_spam_emails'

# Tanítóadatok beolvasása
spam_data = create_dataset_from_eml(spam_folder, is_spam=True)
not_spam_data = create_dataset_from_eml(not_spam_folder, is_spam=False)

# Egyesítés
training_data = pd.concat([spam_data, not_spam_data]).reset_index(drop=True)

# A DataFrame keverése, hogy véletlenszerű sorrendben legyenek az e-mailek
training_data = training_data.sample(frac=1).reset_index(drop=True)

"""

def train_naive_bayes(df):
    X = df.drop(columns=['label'])
    y = df['label']
    model = MultinomialNB()
    model.fit(X, y)
    return model

def evaluate_model(model, test_data):
    X_test = test_data.drop(columns=['label'])
    y_test = test_data['label']
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def calculate_correlations(df):
    correlations = {}
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                corr, _ = pearsonr(df[col1], df[col2])
                correlations[f"{col1}-{col2}"] = corr
    return correlations

if __name__ == "__main__":
    # Spam és nem spam e-mailek mappájának megadása
    spam_folder = './/spamemailek'
    not_spam_folder = './/emailek'

    # Adatbázis létrehozása
    spam_data = create_dataset_from_eml(spam_folder, is_spam=True)
    not_spam_data = create_dataset_from_eml(not_spam_folder, is_spam=False)
    training_data = pd.concat([spam_data, not_spam_data]).reset_index(drop=True)

    # Tanítás és kiértékelés
    train_data, test_data = train_test_split(training_data, test_size=0.25, random_state=42)
    model = train_naive_bayes(train_data)
    evaluate_model(model, test_data)

    # Korreláció számítása
    correlations = calculate_correlations(training_data)
    print("Correlations:\n", correlations)
