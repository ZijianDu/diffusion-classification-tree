#class that is used to process and visualize classifications
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import offsetbox
from time import time
from sklearn.preprocessing import MinMaxScaler

## heavily inspired by sklearn manifold learning example
class visualizer:
    def __init__(self, n_neighbors, images, actual_prediction):
        self.n_neighbors = n_neighbors
        self.labels = [i for i in range(1000)]
        self.num_samples = len(images)
        self.images = images
        assert len(images) == len(actual_prediction)
        # class prediction
        self.y = actual_prediction
        self.embeddings = {
        "Random projection embedding": SparseRandomProjection(
            n_components=2, random_state=42
        )
        }
        """
        "Truncated SVD embedding": TruncatedSVD(n_components=2),
        "Linear Discriminant Analysis embedding": LinearDiscriminantAnalysis(
            n_components=2
        ),
        "Isomap embedding": Isomap(n_neighbors=self.n_neighbors, n_components=2),
        "Standard LLE embedding": LocallyLinearEmbedding(
            n_neighbors=self.n_neighbors, n_components=2, method="standard"
        ),
        "Modified LLE embedding": LocallyLinearEmbedding(
            n_neighbors=self.n_neighbors, n_components=2, method="modified"
        ),
        "Hessian LLE embedding": LocallyLinearEmbedding(
            n_neighbors=self.n_neighbors, n_components=2, method="hessian"
        ),
        "LTSA LLE embedding": LocallyLinearEmbedding(
            n_neighbors=self.n_neighbors, n_components=2, method="ltsa"
        ),
        "MDS embedding": MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2),
        "Random Trees embedding": make_pipeline(
            RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0),
            TruncatedSVD(n_components=2),
        ),
        "Spectral embedding": SpectralEmbedding(
            n_components=2, random_state=0, eigen_solver="arpack"
        ),
        "t-SNE embedding": TSNE(
            n_components=2,
            n_iter=500,
            n_iter_without_progress=150,
            n_jobs=2,
            random_state=0,
        ),
        "NCA embedding": NeighborhoodComponentsAnalysis(
            n_components=2, init="pca", random_state=0
        ),
        """
        
    def set_images_and_predictions(self, images, actual_prediction):
        assert len(actual_prediction) == self.num_samples
        assert len(images) == self.num_samples
        self.y = actual_prediction
        self.images = images
    

    # X: embedding, images: list of 2d images, labels: [0~999], y: predicted label
    def plot_embedding(self, X, title):
        _, ax = plt.subplots()
        X = MinMaxScaler().fit_transform(X)
        for label in self.labels:
            ax.scatter(
                *X[self.y == label].T,
                marker = f"${label}$",
                s = 60,
                color = plt.cm.Dark2(label),
                alpha = 0.425,
                zorder = 2,
            )
        shown_images = np.array([[1.0, 1.0]])
        for i in range(X.shape[0]):
            # plot every image on the embedding
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # do not show points too close
                continue
            shown_images = np.concatenate([shown_images, [X[i]]], axis = 0)
            imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(self.images[i], cmap=plt.cm.gray_r), X[i]
            )
            imagebox.set(zorder=1)
            ax.add_artist(imagebox)
        ax.set_title(title)
        ax.axis("off")

    def run_manifold_learning(self):
        projections, timing = {}, {}
        for name, transformer in self.embeddings.items():
            if name.startswith("Linear Discriminant Analysis"):
                data = self.images.copy()
                data.flat[:: self.images.shape[1] + 1] += 0.01  # Make X invertible
            else:
                data = self.images

            print(f"Computing {name}...")
            start_time = time()
            #print("starting to do transform", data, self.y)
            projections[name] = transformer.fit_transform(data, self.y)
            timing[name] = time() - start_time

        # Finally, we can plot the resulting projection given by each method.
        for name in timing:
            title = f"{name} (time {timing[name]:.3f}s)"
            self.plot_embedding(projections[name], title)
            plt.savefig(title + ".jpg")
