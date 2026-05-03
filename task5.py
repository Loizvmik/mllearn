import pandas as pd
import numpy as np
from numpy import mean, linalg
from sklearn.metrics import adjusted_rand_score
data=pd.read_csv("data/pca_clusters.csv")

def preproc(data):
    X, y = data.drop(columns=["true_cluster"]), data["true_cluster"]
    length=len(y)
    X=X-mean(X, axis=0)
    X_train,X_val, X_test = X.iloc[:int(length*0.7)], X.iloc[int(length*0.7):int(length*0.85)], X.iloc[int(length*0.85):]
    y_train, y_val, y_test = y.iloc[:int(length*0.7)], y.iloc[int(length*0.7):int(length*0.85)], y.iloc[int(length*0.85):]
    return X_train, X_val, X_test, y_train, y_val, y_test


def PCA(X):
    X = np.asarray(X)
    U, s, Vt = linalg.svd(X)
    Vt=Vt[:2]
    s=s[:2]
    return X @ Vt.T

def k_means_iter(X, centers):
    labels = []

    for x in X:
        best_label = 0
        best_dist = None

        for idx, center in enumerate(centers):
            diff = x - center
            dist = sum(diff * diff)

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_label = idx

        labels.append(best_label)

    return labels, centers

def k_means(X, k=2, max_iters=100):
    eps=1e-4
    centers = X[:k].copy()

    for _ in range(max_iters):
        labels, centers = k_means_iter(X, centers)
        new_centers = []

        for label in range(k):
            points_sum = None
            count = 0

            for i in range(len(X)):
                if labels[i] == label:
                    if points_sum is None:
                        points_sum = X[i].copy()
                    else:
                        points_sum += X[i]

                    count += 1

            if count == 0:
                new_centers.append(centers[label])
            else:
                new_centers.append(points_sum / count)

        new_centers = np.array(new_centers)

        if linalg.norm(centers - new_centers) < eps:
            return labels, new_centers

        centers = new_centers

    return labels, centers

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = preproc(data)
    X_train_pca = PCA(X_train)
    labels, centers = k_means(X_train_pca)
    for k in [2, 3, 4, 5,6]:
        labels, centers = k_means(X_train_pca, k=k)
        ari = adjusted_rand_score(y_train, labels)
        comparison = pd.crosstab(
            pd.Series(y_train.to_numpy(), name="true_cluster"),
            pd.Series(labels, name="kmeans_cluster"),
        )
        print(f"\nk={k}, found_clusters={len(set(labels))}, adjusted_rand={ari:.4f}")
        print(comparison)
        print("centers:")
        print(centers)

if __name__ == "__main__":
    main()
        
