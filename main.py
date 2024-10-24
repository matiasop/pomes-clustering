import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm


def load_dataframe(csv_filename, poems_filename):
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        print("Loaded existing DataFrame with embeddings.")
    else:
        df = pd.read_csv(poems_filename)
        print("Loaded original poems DataFrame.")
    return df


def generate_embeddings(df, model):
    start_time = time.time()

    tqdm.pandas(desc='Generating SBERT Embeddings')
    df['Poem_Embeddings'] = df['Poem'].progress_apply(lambda poem: model.encode(poem).tolist())

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Embeddings generated in {elapsed_time:.2f} seconds.")
    return df


def perform_kmeans_clustering(df, n_clusters):
    # Convert the string representations of embeddings back to lists
    embeddings = df['Poem_Embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',')).tolist()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(embeddings)

    print(f"K-Means clustering completed with {n_clusters} clusters.")
    return df


def save_dataframe(df, csv_filename):
    df.to_csv(csv_filename, index=False)
    print(f"DataFrame with embeddings saved to {csv_filename}.")


def main():
    csv_filename = 'poems_with_embeddings.csv'
    poems_filename = 'poems.csv'
    n_clusters = 5

    df = load_dataframe(csv_filename, poems_filename)

    if 'Poem_Embeddings' not in df.columns:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        df = generate_embeddings(df, model)
        save_dataframe(df, csv_filename)

    df = perform_kmeans_clustering(df, n_clusters)

    clustered_csv_filename = 'poems_with_clusters.csv'
    save_dataframe(df, clustered_csv_filename)


if __name__ == "__main__":
    main()
