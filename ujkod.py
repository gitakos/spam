import os
import email
from email.policy import default
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import string

# Törzset kinyerő és feature extrakciós függvények nem változnak
def read_eml_files(folder_path):
    emails = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.eml'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                msg = email.message_from_file(file, policy=default)
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            emails.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
                else:
                    emails.append(msg.get_payload(decode=True).decode('utf-8', errors='ignore'))
    return emails

def extract_features(email_text):
    features = {}
    words = email_text.split()
    word_counts = pd.Series(words).value_counts()
    features['most_common_word_freq'] = word_counts.max() / len(words) if len(words) > 0 else 0
    keywords = [
        # Marketing és promóció
        "ajánlat", "akció", "nyeremény", "kedvezmény", "leárazás", "ingyenes", "csak ma", "különleges", 
        "vásároljon most", "rendeljen", "kupon", "árleszállítás", "kedvező ár", "akciós ár", 
        "black friday", "cyber monday", "kiárusítás",

        # Jutalmak és nyeremények
        "nyereményjáték", "díj", "sorsolás", "nyert", "szerencsés nyertes", "ajándék", "prize", 
        "giveaway", "reward", "claim now",

        # Pénzügyi csalások
        "könnyű pénz", "gyors bevétel", "bank", "hitel", "befektetés", "örökség", "adomány", 
        "pénz", "nagy összeg", "gyors hitel", "hitelkártya", "biztosítás", "örökösödés", 
        "jutalék", "wealth", "lottery", "fund transfer", "millionaire", 

        # Phishing kulcsszavak
        "ellenőrzés", "fiókfrissítés", "bejelentkezés", "jelszó", "biztonsági frissítés", 
        "azonosító", "számla", "hozzáférés", "account", "verify", "update now", "login here", 
        "reset password", "unauthorized access",

        # E-mail formázással kapcsolatos kulcsszavak
        "urgent", "fontos", "important", "confidential", "action required", 
        "do not ignore", "time-sensitive", "response needed",

        # Hamis ígéretek
        "100% garantált", "ingyen", "no risk", "limitált ajánlat", "ne hagyja ki", "exkluzív", 
        "gyors eredmény", "titok", "biztos módszer", "csodálatos", "azonnal", "megoldás", 
        "tudományosan igazolt", "működik mindenkinél", 

        # Egyéb általános spamszavak
        "hirdetés", "promóció", "ingatlan", "lottó", "orvosi", "gyógyszer", "edzés", "fogyás", 
        "diéta", "nagy méret", "bővítés", "potencia", "kattints ide", "most kattints", 
        "megnövekedett méret", "szenzációs", "kedvezményes", "special deal", "offer", 
        "free trial", "buy now", "limited time"
    ]

    features['keyword_count'] = sum(1 for word in words if word in keywords)
    features['email_length'] = len(email_text)
    features['multiple_recipients'] = int("," in email_text.split("To:")[1]) if "To:" in email_text else 0
    features['has_html'] = int('<html>' in email_text.lower())
    features['word_count'] = len(words)
    features['link_count'] = email_text.count('http')
    special_chars = set(string.punctuation)
    features['special_char_count'] = sum(1 for char in email_text if char in special_chars)
    features['has_attachment'] = int("Content-Disposition: attachment" in email_text)
    return features

def create_dataset_from_eml(folder_path, is_spam):
    emails = read_eml_files(folder_path)
    data = []
    for email_text in emails:
        features = extract_features(email_text)
        features['label'] = int(is_spam)
        data.append(features)
    return pd.DataFrame(data)

def train_naive_bayes(df):
    X = df.drop(columns=['label'])
    y = df['label']
    model = MultinomialNB()
    model.fit(X, y)
    return model

def evaluate_email(email_text, model, feature_columns):
    features = extract_features(email_text)
    # Készíts DataFrame-t, hogy megőrizzük az oszlopneveket
    feature_vector = pd.DataFrame([features], columns=feature_columns)
    spam_probability = model.predict_proba(feature_vector)[0][1]
    return spam_probability

def evaluate_model(model, test_data):
    """
    A modell teljesítményének kiértékelése.
    """
    X_test = test_data.drop(columns=['label'])
    y_test = test_data['label']
    y_pred = model.predict(X_test)
    print("Modell teljesítménye:")
    print(f"Pontosság: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nRészletes osztályozási jelentés:")
    print(classification_report(y_test, y_pred))

def kiertekeles(folder_path, model, feature_columns):
    """
    E-mailek kiértékelése egy adott mappából.
    """
    test_emails = read_eml_files(folder_path)
    for idx, email_text in enumerate(test_emails):
        spam_probability = evaluate_email(email_text, model, feature_columns)
        print(f"E-mail {idx + 1}: {spam_probability * 100:.2f}% eséllyel spam.")

if __name__ == "__main__":
    spam_folder = './spamemailek'
    not_spam_folder = './emailek'

    # Adatok betöltése és feldolgozása
    spam_data = create_dataset_from_eml(spam_folder, is_spam=True)
    not_spam_data = create_dataset_from_eml(not_spam_folder, is_spam=False)
    training_data = pd.concat([spam_data, not_spam_data]).reset_index(drop=True)

    # Adatok szétválasztása tanító és teszt adatokra
    train_data, test_data = train_test_split(training_data, test_size=0.5, random_state=42)
    model = train_naive_bayes(train_data)
    feature_columns = train_data.drop(columns=['label']).columns

    # E-mailek kiértékelése
    print("Spam e-mailek értékelése:")
    kiertekeles(spam_folder, model, feature_columns)

    print("\nNem spam e-mailek értékelése:")
    kiertekeles(not_spam_folder, model, feature_columns)

    # Modell teljesítményének kiértékelése
    print("\nModell teljesítményének kiértékelése teszt adatokon:")
    evaluate_model(model, test_data)

