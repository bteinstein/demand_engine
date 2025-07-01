import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_unsupervised_clustering(df: pd.DataFrame) -> dict:
    # Usage:
    X = df[['Latitude', 'Longitude']].values
    labels = df['cluster'].values
    scores = {
        "Silhouette Score":  silhouette_score(X, labels).round(2),
        "Davies-Bouldin Index": davies_bouldin_score(X, labels).round(2),
        "Calinski-Harabasz Score": calinski_harabasz_score(X, labels).round(2)
    }

    for key in scores:
        print(f"{key}: {scores[key]}")
    return scores