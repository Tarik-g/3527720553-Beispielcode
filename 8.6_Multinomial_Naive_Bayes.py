from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Trainingsdaten
spam_mails = [
    "Gewinnen Sie jetzt Geld!",
    "Klicken Sie hier für ein kostenloses iPhone!",
    "Exklusives Angebot nur für Sie!"
]

normale_mails = [
    "Können wir uns morgen für ein Meeting treffen?",
    "Hier sind die Dokumente, die Sie angefordert haben.",
    "Vergessen Sie nicht, die Rechnung zu bezahlen."
]

train_data = spam_mails + normale_mails
train_labels = ['spam'] * 3 + ['nicht-spam'] * 3

# Testdaten
test_data = [
    "Das Meeting wurde verschoben.",
    "Hier ist Ihr kostenloses Geschenk!",
    "Können wir das Meeting auf nächste Woche verschieben?"
]

# Text in Vektor umwandeln
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)

# Modell trainieren
classifier = MultinomialNB()
classifier.fit(X_train, train_labels)

# Testdaten klassifizieren
X_test = vectorizer.transform(test_data)
predictions = classifier.predict(X_test)
print(predictions)