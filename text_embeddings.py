import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.metrics import silhouette_score

# 1. Wczytanie danych i wektorów z Części I (Word2Vec oraz FastText)
df = pd.read_csv("przygotowane_dane_teksty.csv")
X_w2v = np.load("X_word2vec.npy")
X_ft = np.load("X_fasttext.npy")

# 2. Czyszczenie braków danych (NaN w etykietach oraz wektorów zerowych)
valid_mask = (
        df['mowa nienawiści'].notna() &
        df['migracja'].notna() &
        ~np.all(X_w2v == 0, axis=1) &
        ~np.all(X_ft == 0, axis=1)
)

df = df[valid_mask].reset_index(drop=True)
X_w2v = X_w2v[valid_mask]
X_ft = X_ft[valid_mask]

print(f"Liczba poprawnych tekstów do analizy: {len(df)}")


# 3. Funkcja do wizualizacji t-SNE (Zadanie 2)
def plot_tsne(X, df_data, model_name):
    print(f"Obliczanie t-SNE dla: {model_name}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    df_plot = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'Tekst': df_data['text'].str[:120] + '...',
        'Mowa_nienawisci': df_data['mowa nienawiści'].astype(bool).astype(str),
        'Migracja': df_data['migracja'].astype(bool).astype(str)
    })

    fig = px.scatter(
        df_plot, x='x', y='y',
        color='Mowa_nienawisci',
        hover_data=['Tekst', 'Migracja'],
        title=f"Wizualizacja t-SNE - {model_name}",
        color_discrete_map={'True': 'red', 'False': 'blue'}
    )
    fig.show()


plot_tsne(X_w2v, df, "Uśrednione wektory Word2Vec")
plot_tsne(X_ft, df, "Osadzenia zdań FastText")

# 4. Analiza jakości osadzeń pod dyskusję (Zadanie 3)
y_nienawisc = df['mowa nienawiści'].astype(bool).astype(int).values
y_migracja = df['migracja'].astype(bool).astype(int).values

print("\n--- WYNIKI SILHOUETTE SCORE (DO DYSKUSJI) ---")
print(
    f"Mowa nienawiści -> Word2Vec: {silhouette_score(X_w2v, y_nienawisc):.4f} | FastText: {silhouette_score(X_ft, y_nienawisc):.4f}")
print(
    f"Migracja        -> Word2Vec: {silhouette_score(X_w2v, y_migracja):.4f} | FastText: {silhouette_score(X_ft, y_migracja):.4f}")