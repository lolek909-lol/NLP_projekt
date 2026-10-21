import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText

# ==========================================
# 1. WCZYTANIE ZAPISANYCH DANYCH Z PLIKU 1
# ==========================================
print("Wczytywanie danych...")
df = pd.read_csv("przygotowane_dane_teksty.csv")
X_word2vec = np.load("X_word2vec.npy")
X_fasttext = np.load("X_fasttext.npy")

print("Wczytywanie modeli (to potrwa tylko chwilę)...")
w2v_model = KeyedVectors.load("w2v_model.gensim")

try:
    ft_model = FastText.load("ft_model.gensim")
except AttributeError:
    from gensim.models.fasttext import load_facebook_model
    # ft_model = load_facebook_model("C:/Users/mateu/Downloads/kgr10.plain.skipgram.dim100.neg10.bin")

print("Dane i modele gotowe!")

# ==========================================
# ZADANIE 2: WIZUALIZACJA t-SNE (Wykres interaktywny)
# ==========================================
print("\nRedukcja wymiarów za pomocą t-SNE (to może potrwać kilkanaście sekund)...")

# t-SNE (X, Y)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_w2v_2d = tsne.fit_transform(X_word2vec)

df_plot = pd.DataFrame({
    'x': X_w2v_2d[:, 0],
    'y': X_w2v_2d[:, 1],
    'Tekst': df['text'].str[:120] + '...',  # Ucinamy podgląd
    'Mowa_nienawisci': df['mowa nienawiści'].astype(str),
    'Migracja': df['migracja'].astype(str)
})

print("Generowanie interaktywnego wykresu Plotly...")
fig = px.scatter(
    df_plot,
    x='x',
    y='y',
    color='Mowa_nienawisci',
    hover_data=['Tekst', 'Migracja'],
    title="Wizualizacja t-SNE - Word2Vec (Mowa nienawiści)",
    color_discrete_map={'True': 'red', 'False': 'blue', 'nan': 'grey'}
)

# Otwiera wykres w przeglądarce
fig.show()

# ==========================================
# ZADANIE 3: LISTY k-NAJBARDZIEJ PODOBNYCH SŁÓW
# ==========================================
print("\n" + "=" * 50)
print("PORÓWNANIE PODOBIEŃSTWA SŁÓW (Word2Vec vs FastText)")
print("=" * 50)

# Słowa testowe (zmień według uznania)
slowa_testowe = ["migracja", "uchodźca", "granica", "polska", "murzyn"]
k = 5

for slowo in slowa_testowe:
    print(f"\n---> Słowo: {slowo.upper()}")

    # WORD2VEC
    print("Word2Vec:")
    try:
        w2v_podobne = w2v_model.most_similar(slowo, topn=k)
        for s, pod in w2v_podobne:
            print(f"  - {s} ({pod:.3f})")
    except KeyError:
        print(f"  [!] Brak słowa '{slowo}' w słowniku Word2Vec")

    # FASTTEXT
    print("FastText:")
    try:
        # Pamiętajmy, że model FastText przechowuje wektory w atrybucie .wv
        ft_podobne = ft_model.wv.most_similar(slowo, topn=k)
        for s, pod in ft_podobne:
            print(f"  - {s} ({pod:.3f})")
    except KeyError:
        print(f"  [!] Brak słowa '{slowo}' w słowniku FastText")
    except AttributeError:
        # Zabezpieczenie dla różnych sposobów ładowania FastTexta w gensim
        ft_podobne = ft_model.most_similar(slowo, topn=k)
        for s, pod in ft_podobne:
            print(f"  - {s} ({pod:.3f})")