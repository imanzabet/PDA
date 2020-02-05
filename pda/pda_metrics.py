from __future__ import print_function

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import v_measure_score, homogeneity_score,\
    completeness_score, adjusted_rand_score, adjusted_mutual_info_score,\
    normalized_mutual_info_score, contingency_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

class Cluster_Metrics(object):

    def __init__(self, X=None, y=None, y_pred=None, centers=None, n_clusters=None):
        self.y = y
        self.y_pred = y_pred
        self.X = X
        self.centers = centers
        self.n_clusters = n_clusters

        self.silhouette_avg = None
        self.v_measure = None
        self.homogeneity = None
        self.completeness = None
        self.a_rand = None
        self.a_mutual_info = None
        self.n_mutual_info = None
        self.contingency_m = None
        self.confusion_m = None

    def silhouette_score(self):
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        self.silhouette_avg = silhouette_score(self.X, self.y_pred)
        # print("For n_clusters =", n_clusters,
        #       "The average silhouette_score is :", self.silhouette_avg)
        return self.silhouette_avg

    def v_measure_score(self):
        self.v_measure = v_measure_score(self.y, self.y_pred)
        # print('v_measure_score for %s is: %.8f' % (name, v_m))
        return self.v_measure

    def homogeneity_score(self):
        self.homogeneity = homogeneity_score(self.y, self.y_pred)
        # print('homogeneity_score for %s is: %.8f' % (name, h_s))
        return self.homogeneity

    def completeness_score(self):
        self.completeness = completeness_score(self.y, self.y_pred)
        # print('completeness_score for %s is: %.8f' % (name, c_s))
        return self.completeness

    def ari(self):
        self.a_rand = adjusted_rand_score(self.y, self.y_pred)
        # print('adjusted_rand_score for %s is: %.8f' % (name, ars))
        return self.a_rand
    def ami(self):
        self.a_mutual_info = adjusted_mutual_info_score(self.y, self.y_pred)
        # print('adjusted_mutual_info_score for %s is: %.8f' % (name, ami))
        return self.a_mutual_info

    def nmi(self):
        self.n_mutual_info = normalized_mutual_info_score(self.y, self.y_pred)
        # print('normalized_mutual_info_score for %s is: %.8f' % (name, nmi))
        return self.n_mutual_info

    def contingency_matrix(self):
        self.contingency_m = contingency_matrix(self.y, self.y_pred)
        return self.contingency_m

    def confusion_matrix(self):
        confusion_m = confusion_matrix(self.y, self.y_pred)
        return confusion_m

    def plot_silhouette(self):

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(10, 16)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(self.X) + (self.n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        # clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        # cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(self.X, self.y_pred)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(self.X, self.y_pred)

        y_lower = 10
        for i in range(self.n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[self.y_pred == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / self.n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for %d clusters \nSilhouette score=%f.\n"
                      %(self.n_clusters, silhouette_avg),
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(self.y_pred.astype(float) / self.n_clusters)
        ax2.scatter(self.X[:, 0], self.X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        # centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(self.centers[:, 0], self.centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(self.centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        # plt.suptitle()

        plt.show()
    def generate_report(self):
        report = {
            'silhouette_avg': self.silhouette_score(),
            'v_measure': self.v_measure_score(),
            'homogeneity': self.homogeneity_score(),
            'completeness': self.completeness_score(),
            'a_rand': self.ari(),
            'a_mutual_info': self.ami(),
            'n_mutual_info': self.nmi(),
            'contingency_m': self.confusion_matrix(),
            'confusion_m': self.confusion_matrix()
        }
        return report

