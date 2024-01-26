# Importăm bibliotecile necesare
import numpy as np # pentru manipularea array-urilor și algebra liniară
import pandas as pd  # pentru procesarea datelor
import matplotlib.pyplot as plt # pentru crearea și vizualizarea datelor
import matplotlib as mpl   # pentru vizualizare
import seaborn as sns # pentru vizualizarea datelor
import matplotlib.colors as mcolors # pentru vizualizarea culorilor
import string # colecție de alfabete, cuvinte sau alte caractere
import re # suport pentru expresii regulate
import gensim # pentru reprezentarea documentelor ca vectori semantici
import nltk
from nltk import pos_tag # Part-of-Speech (POS) tagging
from nltk.stem import WordNetLemmatizer # lematizează un cuvânt
from nltk.stem.porter import PorterStemmer # pentru informații de recuperare
from nltk.tokenize import word_tokenize # împarte propoziția în cuvinte
from nltk.tokenize import WhitespaceTokenizer # împarte și elimină doar caracterele spațiu alb
from nltk.corpus import wordnet # bază de date lexicală mare a cuvintelor în limba engleză
from nltk.sentiment.vader import SentimentIntensityAnalyzer # analiza de sentiment

nltk.download('stopwords')  # descarcă stopword-urile
nltk.download('averaged_perceptron_tagger')  # descarcă POS tagger
nltk.download('wordnet')  # descarcă WordNet
nltk.download('omw-1.4')  # descarcă Open Multilingual Wordnet
nltk.download('vader_lexicon')  # descarcă lexicul VADER
from matplotlib.cm import ScalarMappable  # pentru colormap
from matplotlib.lines import Line2D  # segment de linie în spațiul de coordonate (x, y)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # oferă figurii noastre personalități suplimentare
from textwrap import wrap  # modificarea comportamentului funcțiilor
from wordcloud import WordCloud  # pentru a vizualiza textul
from sklearn.feature_extraction.text import TfidfVectorizer  # convertește o colecție de documente brute într-o matrice
from gensim.test.utils import common_texts  # corpus de antrenament
from gensim.models.doc2vec import Doc2Vec, TaggedDocument  # reprezintă documentele ca vectori
from PIL import Image  # import imagine

# Pentru Modelul de Învățare Automată
from sklearn import preprocessing  # funcții de utilitate
from sklearn.impute import SimpleImputer  # înlocuiește valorile lipsă
from sklearn.ensemble import RandomForestClassifier  # algoritm de învățare Random Forest
from sklearn.linear_model import LogisticRegression  # algoritm de învățare Logistic Regression
from sklearn.model_selection import train_test_split  # creează date de antrenare și testare
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix  # metrici pentru evaluarea modelului
from sklearn.metrics import roc_auc_score, recall_score  # scor AUC-ROC și recall
from sklearn import tree  # pentru modelare
from sklearn.metrics import classification_report  # pentru raport de clasificare
from sklearn.metrics import roc_curve, recall_score  # scor AUC-ROC și recall


# Încarcăm setul de date
df = pd.read_csv('reviews_data.csv', encoding='ISO-8859-1')
df.head()
print(df)

# Preluam un eșantion de 10% din setul de date
df = df.sample(frac = 0.1, replace = False, random_state=42)

# Elimină coloanele inutile din setul de date
df = df.drop(['name', 'location', 'Date', 'Image_Links'], axis=1)

# Deschidem fișierul de stopwords și îl stocam într-o listă
gist_file = open("gist_stopwords.txt", "r")
content = gist_file.read()
stopwords = content.split(",")
gist_file.close()

# Funcție pentru a obține POS (Part-of-Speech) tag pentru un cuvânt
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Funcție pentru curățarea textului
def clean_text(text):
    # elimină \t
    text = text.replace('\t', '')
    # transformă textul în litere mici
    text = text.lower()
    # tokenizează textul și elimină punctuația
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # elimină cuvintele care conțin cifre
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # elimină stopword-urile
    text = [x for x in text if x not in stopwords]
    # elimină token-urile goale
    text = [t for t in text if len(t) > 0]
    # POS tag pentru text
    pos_tags = pos_tag(text)
    # lematizează textul
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # elimină cuvintele cu o singură literă
    text = [t for t in text if len(t) > 1]
    # unește toate
    text = ' '.join(text)
    return text

# Aplicăm funcția de curățare a textului pe coloana 'Review' și cream o nouă coloană 'Clean_Review'
df['Clean_Review'] = df['Review'].apply(lambda x: clean_text(x))
df.head()
print(df)

# Inițializam SentimentIntensityAnalyzer si adaugăm scorurile de sentiment la setul de date
sid = SentimentIntensityAnalyzer()

df['Sentiments'] = df['Review'].apply(lambda x: sid.polarity_scores(x))
df = pd.concat([df.drop(['Sentiments'], axis=1), df['Sentiments'].apply(pd.Series)], axis=1)
df.head()
print(df)

# Cream coloanele pentru vectorii Doc2Vec
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df['Clean_Review'].apply(lambda x: x.split(' ')))]

# Antrenam un model Doc2Vec cu datele noastre de text
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# Transformăm fiecare document într-un vector de date
doc2vec_df = df['Clean_Review'].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ['doc2vec_vector_' + str(x) for x in doc2vec_df.columns]
df = pd.concat([df, doc2vec_df], axis=1)
df.head()
print(df)

# Cream reprezentarea TF-IDF pentru text
tfidf = TfidfVectorizer(min_df=10)
tfidf_result = tfidf.fit_transform(df['Clean_Review']).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names_out())
tfidf_df.columns = ['word_' + str(x) for x in tfidf_df.columns]
tfidf_df.index = df.index
df = pd.concat([df, tfidf_df], axis=1)
df.head()
print(df)

# Etichetam datele în funcție de Rating (0 pentru Rating < 5, 1 pentru Rating >= 5)
df['posneg'] = df['Rating'].apply(lambda x: 0 if x < 5 else 1)

# Selectam recenziile pozitive și negative
train_pos = df[df['posneg'] == 1]
train_pos = train_pos['Clean_Review']
train_neg = df[df['posneg'] == 0]
train_neg = train_neg['Clean_Review']

# Funcție pentru generarea de WordCloud
def wordCloud_generator(data, color, color_map):
    # Utilizam o mască pentru a da forma norului
    wave_mask = np.array(Image.open('shape.png'))

    # Cream un obiect WordCloud
    wordcloud = WordCloud(width=1000, height=1000,
                          background_color=color,
                          min_font_size=12,
                          colormap=color_map,
                          mask=wave_mask
                          ).generate(' '.join(data.values))

    # Afișam imaginea WordCloud
    plt.figure(figsize=(10, 10), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

# Generam WordCloud pentru recenziile pozitive și negative
wordCloud_generator(train_pos, 'white', 'ocean')

wordCloud_generator(train_neg, 'white', 'Reds')

# Eliminăm coloanele neutilizate
df = df.drop(['Review', 'Clean_Review'], axis=1)

# Definirea variabilelor (X) și (Y)
X = df.drop(['posneg'], axis=1)
Y = df['posneg']

# Împartim datele în set de antrenare și set de testare
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X,
                                                                Y,
                                                                test_size=0.3,
                                                                random_state=42)

# Inițializam modelul Logistic Regression
logreg = LogisticRegression(random_state=42)

# Antrenam modelul pe setul de antrenare
logreg.fit(X_train_lr, y_train_lr)

# Obținem rezultatele clasificării pentru setul de testare
classification_decision2 = (classification_report(y_test_lr,
                                                  logreg.predict(X_test_lr)))
print(classification_decision2)

# Obținem probabilitățile prezicerii clasei pozitive
y_pred_prob = logreg.predict_proba(X_test_lr)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test_lr, y_pred_prob)

# Calculam ROC-AUC
roc_auc = roc_auc_score(y_test_lr, y_pred_prob)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
print(f'AUC-ROC Score: {roc_auc:.2f}')
plt.show()

# Importăm biblioteca tkinter pentru crearea interfeței grafice
import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext

# Funcție pentru a crea scorul de sentiment
def create_sentiment(input_text):
    # Utilizam modelul de analiză a sentimentelor (SentimentIntensityAnalyzer)
    sentiment_score = sid.polarity_scores(input_text)['compound']
    return sentiment_score

# Funcție pentru a analiza sentimentul și a afișa rezultatul
def analyze_sentiment():
    input_text = text_entry.get("1.0", tk.END).strip()

    if not input_text:
        messagebox.showwarning("Avertisment", "Vă rugăm să introduceți un text pentru analiză.")
        return

    try:
        sentiment_score = create_sentiment(input_text)
    except ValueError:
        messagebox.showwarning("Avertisment", "Textul introdus trebuie să fie un număr valid.")
        return

    # Determinăm rezultatul sentimentului
    if 0.05 <= sentiment_score <= 1:
        sentiment_result = "POZITIV"
    elif -1 <= sentiment_score <= -0.05:
        sentiment_result = "NEGATIV"
    else:
        sentiment_result = "NEUTRU"
    show_result(sentiment_result)

# Funcție pentru a afișa rezultatul în interfața grafică
def show_result(result):
    result_label.config(text=f"Rezultat: {result}")

# Crearea ferestrei principale pentru interfața grafică
root = tk.Tk()
root.title("Analiză de Sentiment")

# Adaugarea un câmp de text pentru introducerea textului
text_label = tk.Label(root, text="Introduceți textul:")
text_label.pack(pady=10)
text_entry = scrolledtext.ScrolledText(root, height=5, width=40)
text_entry.pack(pady=10)

# Adăugarea unui buton pentru analiza sentimentului
analyze_button = tk.Button(root, text="Analizează sentimentul", command=analyze_sentiment)
analyze_button.pack(pady=10)

# Adăugarea unei etichete pentru afișarea rezultatului
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Rularea buclei principale a interfeței grafice
root.mainloop()