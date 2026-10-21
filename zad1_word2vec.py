import pandas as pd
import numpy as np
import glob
import re
from gensim.models import KeyedVectors
import pickle
from gensim.models.fasttext import load_facebook_model


# ==========================================
# 1. ŁĄCZENIE PLIKÓW CSV Z ANOTACJAMI
# ==========================================
print("Łączenie plików CSV...")
file_list = glob.glob("C:/Users/mateu/Downloads/NLP*.csv")
df = pd.concat([pd.read_csv(file) for file in file_list], ignore_index=True)

df = df.dropna(subset=['text']).reset_index(drop=True)
print(f"Załadowano {len(df)} tekstów do wektoryzacji.")

# ==========================================
# 2. FUNKCJA CZYSZCZĄCA TEKST
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==========================================
# 3. OSADZENIA - WORD2VEC (.txt)
# ==========================================
print("\nŁadowanie lokalnego modelu Word2Vec (.txt)...")
W2V_PATH = "C:/Users/mateu/Downloads/wiki-forms-restricted-100-cbow-ns.txt/wiki-forms-restricted-100-cbow-ns.txt"
w2v_model = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False)


def get_w2v_embedding(text, model):
    words = clean_text(text).split()
    vectors = [model[w] for w in words if w in model.key_to_index]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)


print("Tworzenie wektorów Word2Vec dla korpusu...")
w2v_embeddings = [get_w2v_embedding(text, w2v_model) for text in df['text']]


# ===========================================
# 4. OSADZENIA - FASTTEXT                | /
# ========================================
FT_PATH = "C:/Users/mateu/Downloads/kgr10.plain.skipgram.dim100.neg10.bin"

# Używamy dedykowanego loadera dla oryginalnych plików .bin z FastTexta
ft_model = load_facebook_model(FT_PATH)


def get_ft_embedding(text, model):
    words = clean_text(text).split()

    # W modelu FastText z gensim, dostęp do wektorów odbywa się przez atrybut .wv (WordVectors)
    vectors = [model.wv[w] for w in words if w in model.wv.key_to_index]

    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.wv.vector_size)


print("Tworzenie wektorów FastText dla korpusu...")
ft_embeddings = [get_ft_embedding(text, ft_model) for text in df['text']]

# ==========================================
# 5. POŁĄCZENIE WEKTORÓW Z ANOTACJAMI W JEDEN OBIEKT
# ==========================================
print("\nPrzygotowywanie ostatecznych tablic...")

X_word2vec = np.array(w2v_embeddings)
X_fasttext = np.array(ft_embeddings)

y_migracja = df['migracja'].values
y_nienawisc = df['mowa nienawiści'].values

print("GOTOWE! Kształty tablic:")
print(f"Dane Word2Vec: {X_word2vec.shape}")
print(f"Dane FastText: {X_fasttext.shape}")
print(f"Etykiety anotacji: {y_migracja.shape}")


# Zapisujemy nasz połączony i oczyszczony DataFrame z tekstami i anotacjami
df.to_csv("przygotowane_dane_teksty.csv", index=False)

# Zapisujemy wektory do lekkich plików numerycznych Numpy (.npy)
np.save("X_word2vec.npy", X_word2vec)
np.save("X_fasttext.npy", X_fasttext)

w2v_model.save("w2v_model.gensim")
ft_model.save("ft_model.gensim")

# (Opcjonalnie) Jeśli będziesz robił zadanie 3 w innym pliku, same modele są już na dysku,
# więc wystarczy je załadować w nowym skrypcie dokładnie tak, jak zrobiliśmy to tutaj.

print("Zapis zakończony sukcesem! Możesz przejść do t-SNE.")