from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import logging
import os.path
import numpy as np
logger = logging.getLogger(__name__)
from matplotlib import pyplot as plt
import pickle
from art.poison_detection.activation_defence import ActivationDefence
from art.visualization import plot_3d
from pda.pda_visualization import plot_2d
from pda import model_path
import configparser
import argparse

tf.get_logger().setLevel(logging.ERROR)

class PDA(ActivationDefence):
    defence_params = ['nb_clusters', 'clustering_hyparam', 'clustering_method', 'nb_dims', 'reduce', 'cluster_analysis']
    valid_clustering = ['KMeans', 'DBSCAN', 'AgglomerativeClustering',
                        'AffinityPropagation', 'SpectralClustering']
    valid_DBSCAN_algos= ['auto', 'ball_tree', 'kd_tree', 'brute']
    valid_DBSCAN_leaf_size = [30, 100, 1000]
    valid_DBSCAN_eps = 0.5
    valid_DBSCAN_min_samples = 5
    valid_DBSCAN_metric = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
    # ['braycurtis', 'canberra', 'chebyshev',
    #   'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
    #   'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
    #   'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

    valid_reduce = ['PCA', 'FastICA', 'TSNE']
    valid_FastICA_algos= list({'parallel', 'deflation'})
    valid_analysis = ['smaller', 'distance', 'relative-size', 'silhouette-scores']

    def __init__(self, classifier, x_train, y_train):
        super(PDA, self).__init__(classifier, x_train, y_train)
        self.red_activations_by_class = [] # Activations N-D reduced by class
        self.red_2d_activations_by_class = [] # Activations 2-D reduced by class
        self.red_3d_activations_by_class = [] # Activations 3-D reduced by class
        self.centers = []       # centroids for main dim red
        self.centers_2d = []    # centroids for 2d
        self.centers_3d = []    # centroids for 3d


    def evaluate_defence(self, is_clean, **kwargs):
        """
        Returns confusion matrix.

        :param is_clean: Ground truth, where is_clean[i]=1 means that x_train[i] is clean and is_clean[i]=0 means
                         x_train[i] is poisonous.
        :type is_clean: :class `np.ndarray`
        :param kwargs: A dictionary of defence-specific parameters.
        :type kwargs: `dict`
        :return: JSON object with confusion matrix.
        :rtype: `jsonObject`
        """
        if is_clean is None or len(is_clean) == 0:
            raise ValueError("is_clean was not provided while invoking evaluate_defence.")

        self.set_params(**kwargs)

        if not self.activations_by_class:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)

        self.clusters_by_class, self.red_activations_by_class = self._cluster_activations()
        _, self.assigned_clean_by_class = self.analyze_clusters()

        # Now check ground truth:
        self.is_clean_by_class = self._segment_by_class(is_clean, self.y_train)
        self.errors_by_class, conf_matrix_json = self.evaluator.analyze_correctness(self.assigned_clean_by_class,
                                                                                    self.is_clean_by_class)
        return conf_matrix_json

    def detect_poison(self, **kwargs):
        """
        Returns poison detected and a report.

        :param kwargs: A dictionary of detection-specific parameters.
        :type kwargs: `dict`
        :return: (report, is_clean_lst):
                where a report is a dict object that contains information specified by the clustering analysis technique.
                where is_clean is a list, where is_clean_lst[i]=1 means that x_train[i]
                there is clean and is_clean_lst[i]=0, means that x_train[i] was classified as poison.
        :rtype: `tuple`
        """
        self.set_params(**kwargs)

        if not self.activations_by_class:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)
        self.clusters_by_class, self.red_activations_by_class = self._cluster_activations()

        # 1. create a report and store it of analyzing the clusters
        self.analyze_report, self.assigned_clean_by_class = self.analyze_clusters()
        # Here, assigned_clean_by_class[i][j] is 1 if the jth datapoint in the ith class was
        # determined to be clean by activation cluster

        # 2. Build an array that matches the original indexes of x_train
        n_train = len(self.x_train)
        indices_by_class = self._segment_by_class(np.arange(n_train), self.y_train)
        self.is_clean_lst = [0] * n_train
        for assigned_clean, dp in zip(self.assigned_clean_by_class, indices_by_class):
            for assignment, index_dp in zip(assigned_clean, dp):
                if assignment == 1:
                    self.is_clean_lst[index_dp] = 1

    # Broken pipeline to sub-pipelines
    def first_dim_red(self, nb_dims=10, reduce='FastICA',
                            **kwargs):
        self.set_params(**kwargs)
        if not self.activations_by_class:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)

        if not self.red_activations_by_class:
            for ac in self.activations_by_class:
                if ac.shape == (0,):
                    continue
                # Apply dimensionality reduction
                nb_activations = np.shape(ac)[1]
                if nb_activations > nb_dims:
                    reduced_activations = self.reduce_dimensionality(ac, nb_dims=nb_dims, reduce=reduce)
                else:
                    logger.info("Dimensionality of activations = %i less than nb_dims = %i. Not applying dimensionality "
                                "reduction.", nb_activations, nb_dims)
                    reduced_activations = ac
                self.red_activations_by_class.append(reduced_activations)

        return self.red_activations_by_class


    def second_clustering(self, clustering_hyparam, clustering_method='KMeans', **kwargs):

        self.set_params(**kwargs)
        self.clustering_hyparam = clustering_hyparam

        # self.clusters_by_class, self.red_activations_by_class = self.cluster_activations(

        from sklearn.cluster import KMeans
        from sklearn.cluster import DBSCAN
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.cluster import AffinityPropagation
        from sklearn.cluster import SpectralClustering

        separated_clusters = []
        separated_reduced_activations = []
        h = clustering_hyparam

        if clustering_method == 'KMeans':
            clusterer = KMeans(n_clusters=h['n_clusters'],
                               init=h['init'], n_init=h['n_init'],
                               max_iter=h['max_iter'], tol=h['tol'],
                               precompute_distances=h['precompute_distances'],
                               random_state=h['random_state'], algorithm=h['algorithm'])
        elif clustering_method == 'DBSCAN':
            clusterer = DBSCAN(eps=h['eps'], min_samples=h['min_samples'],
                               leaf_size=h['leaf_size'], algorithm=h['algorithm'])
        elif clustering_method == 'AgglomerativeClustering':
            clusterer = AgglomerativeClustering(n_clusters=h['n_clusters'],
                                                affinity=h['affinity'],
                                                linkage=h['linkage'])
        elif clustering_method == 'AffinityPropagation':
            clusterer = AffinityPropagation()
        elif clustering_method == 'SpectralClustering':
            clusterer = SpectralClustering()
        else:
            raise ValueError(clustering_method + " clustering method not supported.")

        # if not hasattr(clusterer, 'n_clusters'):
        #     # algorithm.n_clusters = lambda: None
        #     if type(clusterer.cluster_centers_) != 'numpy.ndarray':
        #         clusterer.cluster_centers_ = np.array([clusterer.cluster_centers_])
        #     if clusterer.cluster_centers_.shape.__len__() == 3:
        #         clusterer.cluster_centers_ = clusterer.cluster_centers_[0, :, :]
        #     setattr(clusterer, 'n_clusters', clusterer.cluster_centers_.shape[0])

        for reduced_activations in self.red_activations_by_class:
            # Get cluster assignments
            clusters = clusterer.fit_predict(reduced_activations)
            try:
                self.centers.append(clusterer.cluster_centers_)
            except:
                try:
                    self.centers.append(self.compute_centroids(reduced_X=reduced_activations,y_pred=clusters))
                except:
                    print('Centroids are not computable!')

            self.clusters_by_class.append(clusters)

        # post processing
        detected_nb_clusters = np.unique(list(self.clusters_by_class[0])).shape[0]
        print('Detected number of clusters for %s is: %d'
              % (clustering_method, detected_nb_clusters))
        # replace -1 with 1 if there is not exisiting any 1
        for cluster in self.clusters_by_class:
            if detected_nb_clusters == 2:
                if cluster[cluster == -1].shape[0] != 0:
                    cluster[cluster == -1] = 1
            else:
                raise ValueError('Detected number of clusters are: %d' % detected_nb_clusters)


        return self.clusters_by_class, self.red_activations_by_class


    def third_analysis(self):
        _, self.assigned_clean_by_class = self.analyze_clusters()

    def fourth_confusion_matrix(self, p_train):
        # Now check ground truth:
        is_clean = (p_train == 0)
        self.is_clean_by_class = self._segment_by_class(is_clean, self.y_train)
        self.errors_by_class, conf_matrix_json = self.evaluator.analyze_correctness(self.assigned_clean_by_class,
                                                                                    self.is_clean_by_class)
        return conf_matrix_json

    def plot_clusters(self, is_plot_3d=False, interactive=False, save=True, folder='.', **kwargs):
        """
        Creates 2D-plot and 3D-plot to visualize each cluster each cluster is assigned a different color in the plot.
        When save=True, it also stores the 3D-plot per cluster in DATA_PATH.

        :param save: Boolean specifying if image should be saved
        :type  save: `bool`
        :param folder: Directory where the sprites will be saved inside DATA_PATH folder
        :type folder: `str`
        :param kwargs: a dictionary of cluster-analysis-specific parameters
        :type kwargs: `dict`
        :return: None
        """
        self.set_params(**kwargs)

        if not self.clusters_by_class:
            self.cluster_activations()

        if not interactive:
            # Get activations reduced to 2-components:
            if not (is_plot_3d):
                # For each class generate a plot:
                for class_id, (labels, coordinates) in enumerate(zip(self.clusters_by_class, self.red_2d_activations_by_class)):
                    f_name = ''
                    if save:
                        f_name = os.path.join(folder, '2D_plot_class_' + str(class_id) + '.png')
                    plot_2d(coordinates, labels, save=save, f_name=f_name)

            if (is_plot_3d):
                # For each class generate a plot:
                for class_id, (labels, coordinates) in enumerate(zip(self.clusters_by_class, self.red_3d_activations_by_class)):
                    f_name = ''
                    if save:
                        f_name = os.path.join(folder, '3D_plot_class_' + str(class_id) + '.png')
                    plot_3d(coordinates, labels, save=save, f_name=f_name)
        # else: # for interactive with jupyter
        #     import plotly
        #     import plotly.plotly as py
        #     import plotly.graph_objs as go
        #     from plotly.offline import init_notebook_mode, iplot
        #     import numpy as np
        #
        #     init_notebook_mode(connected=True)
        #
        #     if not (is_plot_3d):
        #         trace1 = go.Scatter(
        #             x=self.red_2d_activations_by_class[0][self.clusters_by_class[0] == 0][:, 0],
        #             y=self.red_2d_activations_by_class[0][self.clusters_by_class[0] == 0][:, 1],
        #             mode='markers',
        #             marker=dict(
        #                 color='aqua',
        #                 size=12,
        #                 line=dict(
        #                     color='rgba(217, 217, 217, 0.14)',
        #                     width=0.2
        #                 ),
        #                 opacity=.8
        #             )
        #         )
        #         trace2 = go.Scatter(
        #             x=self.red_2d_activations_by_class[0][self.clusters_by_class[0] == 1][:, 0],
        #             y=self.red_2d_activations_by_class[0][self.clusters_by_class[0] == 1][:, 1],
        #             mode='markers',
        #             marker=dict(
        #                 color='darkred',  # 'rgb(127, 127, 127)',
        #                 size=12,
        #                 symbol='circle',
        #                 line=dict(
        #                     color='rgb(204, 204, 204)',
        #                     width=.3
        #                 ),
        #                 opacity=.8
        #             )
        #         )
        #         data = [trace1, trace2]
        #         layout = go.Layout(
        #             margin=dict(
        #                 l=0,
        #                 r=0,
        #                 b=0,
        #                 t=0
        #             )
        #         )
        #         fig = go.Figure(data=data, layout=layout)
        #         py.iplot(fig, filename='simple-2d-scatter')
        #         if(save):
        #             # to save .html file (offline)
        #             plotly.offline.plot(fig, filename='2d_plot.html')
        #         iplot(fig)
        #
        #     if (is_plot_3d):
        #         trace1 = go.Scatter3d(
        #             x=self.red_3d_activations_by_class[0][self.clusters_by_class[0]==0][:,0],
        #             y=self.red_3d_activations_by_class[0][self.clusters_by_class[0]==0][:,1],
        #             z=self.red_3d_activations_by_class[0][self.clusters_by_class[0]==0][:,2],
        #             mode='markers',
        #             marker=dict(
        #                 color='aqua',
        #                 size=12,
        #                 line=dict(
        #                     color='rgba(217, 217, 217, 0.14)',
        #                     width=0.2
        #                 ),
        #                 opacity=.8
        #             )
        #         )
        #         trace2 = go.Scatter3d(
        #             x=self.red_3d_activations_by_class[0][self.clusters_by_class[0]==1][:,0],
        #             y=self.red_3d_activations_by_class[0][self.clusters_by_class[0]==1][:,1],
        #             z=self.red_3d_activations_by_class[0][self.clusters_by_class[0]==1][:,2],
        #             mode='markers',
        #             marker=dict(
        #                 color='darkred',#'rgb(127, 127, 127)',
        #                 size=12,
        #                 symbol='circle',
        #                 line=dict(
        #                     color='rgb(204, 204, 204)',
        #                     width=.3
        #                 ),
        #                 opacity=.8
        #             )
        #         )
        #         data = [trace1, trace2]
        #         layout = go.Layout(
        #             margin=dict(
        #                 l=0,
        #                 r=0,
        #                 b=0,
        #                 t=0
        #             )
        #         )
        #         fig = go.Figure(data=data, layout=layout)
        #         py.iplot(fig, filename='simple-3d-scatter')
        #         if(save):
        #             # to save .html file (offline)
        #             plotly.offline.plot(fig, filename='3d_plot.html')
        #         iplot(fig)

    def plotly_clusters(self, is_plot_3d=False, interactive=False, save=True, folder='.', class_num=0, **kwargs):
        """
        Create plots with errors.
        Creates 2D-plot and 3D-plot to visualize each cluster. Each cluster with 4 different type of datapoints
        is assigned a different color in the plot.
        When save=True, it also stores the 3D-plot per cluster in DATA_PATH.

        :param is_plot_3d: Boolean specifying if image is 3d or 2d
        :param interactive: Boolean, indicating whether the plot will be shown in jupyter notebook
        :param save: Boolean specifying if image should be saved
        :type  save: `bool`
        :param folder: Directory where the sprites will be saved inside DATA_PATH folder
        :type folder: `str`
        :param kwargs: a dictionary of cluster-analysis-specific parameters
        :type kwargs: `dict`
        :return: None
        """
        self.set_params(**kwargs)

        if not self.clusters_by_class:
            self.cluster_activations()
        # if not self.red_2d_activations_by_class==[] or not self.red_3d_activations_by_class==[]:
        #     self.clusters_by_class, self.red_activations_by_class = self._cluster_activations()
        if not interactive:
            # Get activations reduced to 2-components:
            if not (is_plot_3d):
                # For each class generate a plot:
                for class_id, (labels, coordinates) in enumerate(zip(self.errors_by_class, self.red_2d_activations_by_class)):
                    if class_id==class_num:
                        f_name = ''
                        if save:
                            f_name = os.path.join(folder, '2D_plot_class_' + str(class_id) + '.png')
                        plot_2d(coordinates, labels, save=save, f_name=f_name)

            if (is_plot_3d):
                # For each class generate a plot:
                for class_id, (labels, coordinates) in enumerate(zip(self.errors_by_class, self.red_3d_activations_by_class)):
                    if class_id==class_num:
                        f_name = ''
                        if save:
                            f_name = os.path.join(folder, '3D_plot_class_' + str(class_id) + '.png')
                        plot_3d(coordinates, labels, save=save, f_name=f_name)
        if interactive: # for interactive with jupyter
            import plotly
            import plotly.graph_objs as go
            from plotly.offline import init_notebook_mode
            import numpy as np

            init_notebook_mode(connected=True)

            if (not(is_plot_3d)) and class_num==0:
                t0 = self.red_2d_activations_by_class[0][self.errors_by_class[0] == 0]
                t1 = self.red_2d_activations_by_class[0][self.errors_by_class[0] == 1]
                t2 = self.red_2d_activations_by_class[0][self.errors_by_class[0] == 2]
                t3 = self.red_2d_activations_by_class[0][self.errors_by_class[0] == 3]
                trace0 = go.Scatter(
                    x=t0[:, 0],
                    y=t0[:, 1],
                    legendgroup='poison_group',
                    name='%0.04d poison, marked poison' %t0.shape[0],
                    mode='markers',
                    marker=dict(
                        color='aqua',
                        size=6,
                        line=dict(
                            color='rgba(217, 217, 217, 0.14)',
                            width=0.2
                        ),
                        opacity=.8
                    )
                )
                trace1 = go.Scatter(
                    x=t1[:, 0],
                    y=t1[:, 1],
                    legendgroup='clean_group',
                    name='%0.04d clean, marked clean' %t1.shape[0],
                    mode='markers',
                    marker=dict(
                        color='darkred',  # 'rgb(127, 127, 127)',
                        size=6,
                        symbol='circle',
                        line=dict(
                            color='rgb(204, 204, 204)',
                            width=.3
                        ),
                        opacity=.8
                    )
                )
                trace2 = go.Scatter(
                    x=t2[:, 0],
                    y=t2[:, 1],
                    legendgroup='clean_group',
                    name='%0.04d clean, marked poison' %t2.shape[0],
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=6,
                        line=dict(
                            color='rgba(217, 217, 217, 0.14)',
                            width=0.2
                        ),
                        opacity=.8
                    )
                )
                trace3 = go.Scatter(
                    x=t3[:, 0],
                    y=t3[:, 1],
                    legendgroup='poison_group',
                    name='%0.04d poison, marked clean' %t3.shape[0],
                    mode='markers',
                    marker=dict(
                        color='yellow',  # 'rgb(127, 127, 127)',
                        size=6,
                        symbol='circle',
                        line=dict(
                            color='rgb(204, 204, 204)',
                            width=.3
                        ),
                        opacity=.8
                    )
                )
                data = [trace0, trace1, trace2, trace3]
                layout = go.Layout(
                    margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0
                    )
                )
                fig = go.Figure(data=data, layout=layout)
                # plotly.offline.iplot(fig, filename='simple-2d-scatter')
                if(save):
                    # to save .html file (offline)
                    f_name = os.path.join(folder, '2d_plot_class_' + str(class_num) + '.html')
                    plotly.offline.plot(fig, filename=f_name)
                plotly.offline.iplot(fig)

            if (is_plot_3d) and class_num==0:
                t0=self.red_3d_activations_by_class[0][self.errors_by_class[0]==0]
                t1=self.red_3d_activations_by_class[0][self.errors_by_class[0]==1]
                t2=self.red_3d_activations_by_class[0][self.errors_by_class[0]==2]
                t3=self.red_3d_activations_by_class[0][self.errors_by_class[0]==3]

                trace0 = go.Scatter3d(
                    x=t0[:,0],
                    y=t0[:,1],
                    z=t0[:,2],
                    # legendgroup='poison_group',
                    name='%0.04d poison, marked poison' %t0.shape[0],
                    mode='markers',
                    marker=dict(
                        color='aqua',
                        size=6,
                        line=dict(
                            color='rgba(217, 217, 217, 0.14)',
                            width=0.2
                        ),
                        opacity=.8
                    )
                )
                trace1 = go.Scatter3d(
                    x=t1[:,0],
                    y=t1[:,1],
                    z=t1[:,2],
                    # legendgroup='clean_group',
                    name='%0.04d clean, marked clean' %t1.shape[0],
                    mode='markers',
                    marker=dict(
                        color='darkred',#'rgb(127, 127, 127)',
                        size=6,
                        symbol='circle',
                        line=dict(
                            color='rgb(204, 204, 204)',
                            width=.3
                        ),
                        opacity=.8
                    )
                )
                trace2 = go.Scatter3d(
                    x=t2[:,0],
                    y=t2[:,1],
                    z=t2[:,2],
                    # legendgroup='clean_group',
                    name='%0.04d clean, marked poison' %t2.shape[0],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=6,
                        line=dict(
                            color='rgba(217, 217, 217, 0.14)',
                            width=0.2
                        ),
                        opacity=.8
                    )
                )
                trace3 = go.Scatter3d(
                    x=t3[:,0],
                    y=t3[:,1],
                    z=t3[:,2],
                    # legendgroup='poison_group',
                    name='%0.04d poison, marked clean' %t3.shape[0],
                    mode='markers',
                    marker=dict(
                        color='yellow',#'rgb(127, 127, 127)',
                        size=6,
                        symbol='circle',
                        line=dict(
                            color='rgb(204, 204, 204)',
                            width=.3
                        ),
                        opacity=.8
                    )
                )
                data = [trace0, trace1, trace2, trace3]
                layout = go.Layout(
                    margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0
                    )
                )
                fig = go.Figure(data=data, layout=layout)
                # plotly.offline.iplot(fig, filename='simple-3d-scatter')
                if(save):
                    # to save .html file (offline)
                    f_name = os.path.join(folder, '3d_plot_class_' + str(class_num) + '.html')
                    plotly.offline.plot(fig, filename=f_name)
                plotly.offline.iplot(fig)

    def plot_tsne(self, activations, Targets):
        from plotly.offline import iplot
        import plotly.graph_objs as go
        # Invoking the t-SNE method
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(activations)
        traceTSNE = go.Scatter(
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            #     name = 'Target',
            #      hoveron = 'Target',
            mode='markers',
            #     text = 'Target',
            showlegend=True,
            marker=dict(
                size=8,
                color=Targets,
                colorscale='Jet',
                showscale=False,
                line=dict(
                    width=2,
                    color='rgb(255, 255, 255)'
                ),
                opacity=0.8
            )
        )
        data = [traceTSNE]

        layout = dict(title='TSNE (T-Distributed Stochastic Neighbour Embedding)',
                      hovermode='closest',
                      yaxis=dict(zeroline=False),
                      xaxis=dict(zeroline=False),
                      showlegend=False,

                      )

        fig = dict(data=data, layout=layout)
        # py.iplot(fig, filename='styled-scatter2.html')
        iplot(fig)

    def component_analysis(self, is_scalar=False, **kwargs):
        """
        analysis and plot diagram for dimensionality reduction algorithms

        :param kwargs: A dictionary of detection-specific parameters.
        :type kwargs: `dict`
        :return:
        """
        self.set_params(**kwargs)

        if not self.activations_by_class:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)

        # scaling activations between 0 and 1
        if (is_scalar):
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=[0, 1])
            activations = scaler.fit_transform(activations)
        # n_components == 'mle'
        n_components = 30
        for reduce in self.valid_reduce:
            if reduce == 'PCA':
                from sklearn.decomposition import PCA
                for svd_solver in ['auto']:#, 'full', 'arpack', 'randomized']:
                    # Fitting the PCA algorithm with our Data
                    pca = PCA(n_components=n_components,
                              svd_solver=svd_solver,
                              random_state=22)
                    pca.fit(activations)
                    # Plotting the Cumulative Summation of the Explained Variance
                    plt.figure()
                    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
                    plt.step(range(1, 1+n_components), np.cumsum(pca.explained_variance_ratio_), where='mid',
                             label='cumulative explained variance')
                    plt.xlabel('Number of Components')
                    plt.ylabel('Variance (%)')  # for each component
                    plt.title('Spacenet Dataset Explained Variance with svd_solver: '+svd_solver)
                    plt.show()

                    plt.bar(range(1, 1+n_components), pca.explained_variance_ratio_, alpha=0.5, align='center',
                        label='individual explained variance')

                    plt.ylabel('Explained variance ratio')
                    plt.xlabel('Principal components')
                    plt.legend(loc='best')
                    plt.show()
                    a = np.cumsum(pca.explained_variance_ratio_)
                    print('Number of components for 98% variance: ', a[a <= 0.98].shape[0])
                    print('Number of components for 99% variance: ', a[a <= 0.99].shape[0])
            if reduce == 'FastICA ???':
                from sklearn.decomposition import FastICA
                for algorithm in ['parallel', 'deflation']:
                    for max_iter in [200, 1000]:
                        for tol in [0.002, 0.005, 0.01]:
                            fast_ica = FastICA(n_components=n_components,
                                               algorithm=algorithm,
                                               max_iter=max_iter,
                                               tol=tol,
                                               random_state=22)
                            fast_ica.fit(activations)
                pass
            if reduce == 'TSNE':
                pass

    def measure_misclassification(self, classifier, x_test, y_test):
        """
        Computes 1-accuracy given x_test and y_test

        :param classifier: art.classifier to be used for predictions
        :param x_test: test set
        :type x_test: `np.darray`
        :param y_test: labels test set
        :type y_test: `np.darray`
        :return: 1-accuracy
        :rtype `float`
        """
        predictions = np.argmax(classifier.predict(x_test), axis=1)
        return 1 - np.sum(predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]

    def train_remove_backdoor(self, classifier, x_train, y_train, x_test, y_test, tolerable_backdoor,
                              max_epochs, batch_epochs):
        """
        Trains the provider classifier until the tolerance or number of maximum epochs are reached.

        :param classifier: art.classifier to be used for predictions
        :type classifier: `art.classifier`
        :param x_train: training set
        :type x_train: `np.darray`
        :param y_train: labels used for training
        :type y_train: `np.darray`
        :param x_test: samples in test set
        :type x_test: `np.darray`
        :param y_test: labels in test set
        :type y_train: `np.darray`
        :param tolerable_backdoor: Parameter that determines how many missclassifications are acceptable.
        :type tolerable_backdoor: `float`
        :param max_epochs: maximum number of epochs to be run
        :type max_epochs: `int`
        :param batch_epochs: groups of epochs that will be run together before checking for termination
        :type batch_epochs: `int`
        :return: (improve_factor, classifier)
        :rtype `tuple`
        """
        # Measure poison success in current model:
        initial_missed = self.measure_misclassification(classifier, x_test, y_test)

        curr_epochs = 0
        curr_missed = 1
        while curr_epochs < max_epochs and curr_missed > tolerable_backdoor:
            classifier.fit(x_train, y_train, nb_epochs=batch_epochs)
            curr_epochs += batch_epochs
            curr_missed = self.measure_misclassification(classifier, x_test, y_test)
            logger.info('Current epoch: ' + str(curr_epochs))
            logger.info('Misclassifications: ' + str(curr_missed))

        improve_factor = initial_missed - curr_missed
        return improve_factor, classifier
    def _cluster_activations(self, **kwargs):
        """
        Clusters activations and returns cluster_by_class and red_activations_by_class,
        where cluster_by_class[i][j] is the cluster to which the j-th datapoint in the
        ith class belongs and the correspondent activations reduced by class
        red_activations_by_class[i][j].

        :param kwargs: A dictionary of cluster-specific parameters.
        :type kwargs: `dict`
        :return: Clusters per class and activations by class.
        :rtype: `tuple`
        """
        self.set_params(**kwargs)
        if not self.activations_by_class:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)

        [self.clusters_by_class, self.red_activations_by_class] = self.cluster_activations(
            self.activations_by_class,
            self.clustering_hyparam,
            nb_clusters=self.nb_clusters,
            nb_dims=self.nb_dims,
            reduce=self.reduce,
            clustering_method=self.clustering_method)

        self.compute_2d_dimred()
        self.compute_2d_centroids()
        self.compute_3d_dimred()
        self.compute_3d_centroids()
        return self.clusters_by_class, self.red_activations_by_class

    def compute_2d_dimred(self):
        # compute 2-D and 3-D dim red for plotting in future
        self.red_2d_activations_by_class = []
        self.red_3d_activations_by_class = []
        for ac in self.activations_by_class:
            # 2-D
            reduced_activations = self.reduce_dimensionality(ac, nb_dims=2, reduce=self.reduce)
            self.red_2d_activations_by_class.append(reduced_activations)

    def compute_2d_centroids(self):
        # centroids for class 0
        self.centers_2d.append(self.compute_centroids(reduced_X=self.red_2d_activations_by_class[0],
                                                      y_pred=self.clusters_by_class[0]))

    def compute_3d_dimred(self):
        # compute 2-D and 3-D dim red for plotting in future
        for ac in self.activations_by_class:
            # 3-D
            reduced_activations = self.reduce_dimensionality(ac, nb_dims=3, reduce=self.reduce)
            self.red_3d_activations_by_class.append(reduced_activations)

    def compute_3d_centroids(self):
        # centroids for class 0
        self.centers_3d.append(self.compute_centroids(reduced_X=self.red_3d_activations_by_class[0],
                                                      y_pred=self.clusters_by_class[0]))

    def cluster_activations(self, activations_by_class,
                            clustering_hyparam,
                            nb_clusters=2,
                            nb_dims=10, reduce='FastICA',
                            clustering_method='KMeans'):
        """
        Clusters activations and returns two arrays.
        1) separated_clusters: where separated_clusters[i] is a 1D array indicating which cluster each datapoint
        in the class has been assigned
        2) separated_reduced_activations: activations with dimensionality reduced using the specified reduce method

        :param activations_by_class: list where separated_activations[i] is a np matrix for the ith class where
        each row corresponds to activations for a given data point
        :type activations_by_class: `list`
        :param nb_clusters: number of clusters (defaults to 2 for poison/clean)
        :type nb_clusters: `int`
        :param nb_dims: number of dimensions to reduce activation to via PCA
        :type nb_dims: `int`
        :param reduce: Method to perform dimensionality reduction, default is FastICA
        :type reduce: `str`
        :param clustering_method: Clustering method to use, default is KMeans
        :type clustering_method: `str`
        :return: separated_clusters, separated_reduced_activations
        :rtype: `tuple`
        """
        from sklearn.cluster import KMeans
        from sklearn.cluster import DBSCAN
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.cluster import AffinityPropagation
        from sklearn.cluster import SpectralClustering

        separated_clusters = []
        separated_reduced_activations = []
        h=clustering_hyparam

        if clustering_method == 'KMeans':
            clusterer = KMeans(n_clusters=h['n_clusters'],
                               init=h['init'], n_init=h['n_init'],
                               max_iter=h['max_iter'], tol=h['tol'],
                               precompute_distances=h['precompute_distances'],
                               random_state=h['random_state'], algorithm=h['algorithm'])
        elif clustering_method == 'DBSCAN':
            clusterer = DBSCAN(eps=h['eps'], min_samples=h['min_samples'],
                               leaf_size=h['leaf_size'], algorithm=h['algorithm'])
        elif clustering_method == 'AgglomerativeClustering':
            clusterer = AgglomerativeClustering(n_clusters=h['n_clusters'],
                                                affinity=h['affinity'],
                                                linkage=h['linkage'])
        elif clustering_method == 'AffinityPropagation':
            clusterer = AffinityPropagation()
        elif clustering_method == 'SpectralClustering':
            clusterer = SpectralClustering()
        else:
            raise ValueError(clustering_method + " clustering method not supported.")

        # if not hasattr(clusterer, 'n_clusters'):
        #     # algorithm.n_clusters = lambda: None
        #     if type(clusterer.cluster_centers_) != 'numpy.ndarray':
        #         clusterer.cluster_centers_ = np.array([clusterer.cluster_centers_])
        #     if clusterer.cluster_centers_.shape.__len__() == 3:
        #         clusterer.cluster_centers_ = clusterer.cluster_centers_[0, :, :]
        #     setattr(clusterer, 'n_clusters', clusterer.cluster_centers_.shape[0])

        for ac in activations_by_class:
            if ac.shape == (0,):
                continue
            # Apply dimensionality reduction
            nb_activations = np.shape(ac)[1]
            if nb_activations > nb_dims:
                reduced_activations = self.reduce_dimensionality(ac, nb_dims=nb_dims, reduce=reduce)
            else:
                logger.info("Dimensionality of activations = %i less than nb_dims = %i. Not applying dimensionality "
                            "reduction.", nb_activations, nb_dims)
                reduced_activations = ac
            separated_reduced_activations.append(reduced_activations)

            # Get cluster assignments
            clusters = clusterer.fit_predict(reduced_activations)
            try:
                self.centers.append(clusterer.cluster_centers_)
            except:
                self.centers.append(self.compute_centroids(reduced_X=reduced_activations,y_pred=clusters))

            separated_clusters.append(clusters)


        # post processing
        detected_nb_clusters = np.unique(list(separated_clusters[0])).shape[0]
        # print('Detected number of clusters for %s is: %d'
        #       % (clustering_method, detected_nb_clusters))
        # replace -1 with 1 if there is not exisiting any 1
        for cluster in separated_clusters:
            if detected_nb_clusters == 2:
                if cluster[cluster == -1].shape[0] != 0:
                    cluster[cluster == -1] = 1
            else:
                raise ValueError('Detected number of clusters are: %d' % detected_nb_clusters)
        return separated_clusters, separated_reduced_activations

    def cluster_activations_2(self, separated_activations,
                            clustering_hyparam,
                            nb_clusters=2,
                            nb_dims=10, reduce='FastICA',
                            clustering_method='KMeans'):
        """
        Clusters activations and returns two arrays.
        1) separated_clusters: where separated_clusters[i] is a 1D array indicating which cluster each datapoint
        in the class has been assigned
        2) separated_reduced_activations: activations with dimensionality reduced using the specified reduce method

        :param separated_activations: list where separated_activations[i] is a np matrix for the ith class where
        each row corresponds to activations for a given data point
        :type separated_activations: `list`
        :param nb_clusters: number of clusters (defaults to 2 for poison/clean)
        :type nb_clusters: `int`
        :param nb_dims: number of dimensions to reduce activation to via PCA
        :type nb_dims: `int`
        :param reduce: Method to perform dimensionality reduction, default is FastICA
        :type reduce: `str`
        :param clustering_method: Clustering method to use, default is KMeans
        :type clustering_method: `str`
        :return: separated_clusters, separated_reduced_activations
        :rtype: `tuple`
        """
        from sklearn.cluster import KMeans
        from sklearn.cluster import DBSCAN
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.cluster import AffinityPropagation
        from sklearn.cluster import SpectralClustering

        separated_clusters = []
        separated_reduced_activations = []

        # clustering_hyparam = {
        #     'KMeans': {
        #         'n_clusters': 2,  # int, optional, default: 8
        #         'init': 'k-means++',  # {‘k-means++’, ‘random’ or an ndarray} defaults to ‘k-means++’
        #         'n_init': 30,  # number of running algo with rand centroids, int, default: 10
        #         'max_iter': 300,  # int default: 300
        #         'tol': 0.0001,  # tolerance, float, default: 1e-4
        #         'precompute_distances': 'auto',  # {‘auto’, True, False}
        #         'random_state': 22,  # int, RandomState instance or None(default)
        #         'algorithm': 'auto'  # “auto”, “full” or “elkan”, default =”auto”
        #
        #     },
        #     'DBSCAN': {
        #         'eps': 0.5,  # default=0.5
        #         'algorithm': 'auto',  # algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        #         'min_samples': 100,
        #         'leaf_size': 30  # default = 30
        #     },
        #     'AgglomerativeClustering': {
        #         'n_clusters': 2,  # int or None, optional (default=2)
        #         'affinity': 'euclidean',
        #         # “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”, default: “euclidean”
        #         'linkage': 'ward',  # {“ward”, “complete”, “average”, “single”}, optional (default=”ward”)
        #     }
        # }
        c_conf = configparser.ConfigParser()
        c_conf.read('config_clustering.ini')
        # c_conf.set('clustering', 'algorithm', clustering_method)
        h = dict()
        if clustering_method == 'KMeans':
            h['n_clusters'] = c_conf.getint(clustering_method, 'n_clusters')
            h['init'] = c_conf.get(clustering_method, 'init')
            h['n_init'] = c_conf.getint(clustering_method, 'n_init')
            h['max_iter'] = c_conf.getint(clustering_method, 'max_iter')
            h['tol'] = c_conf.getfloat(clustering_method, 'tol')
            h['precompute_distances'] = c_conf.get(clustering_method, 'precompute_distances')
            h['random_state'] = c_conf.getint(clustering_method, 'random_state')
            h['algorithm'] = c_conf.get(clustering_method, 'algorithm')
            clusterer = KMeans(n_clusters=h['n_clusters'],
                               init=h['init'], n_init=h['n_init'],
                               max_iter=h['max_iter'], tol=h['tol'],
                               precompute_distances=h['precompute_distances'],
                               random_state=h['random_state'], algorithm=h['algorithm'])
        elif clustering_method == 'DBSCAN':
            h['eps'] = c_conf.getfloat(clustering_method, 'eps')
            h['min_samples'] = c_conf.getint(clustering_method, 'min_samples')
            h['leaf_size'] = c_conf.getint(clustering_method, 'leaf_size')
            h['algorithm'] = c_conf.get(clustering_method, 'algorithm')
            # leaf_size: int, optional(default=30)
            clusterer = DBSCAN(eps=h['eps'], min_samples=h['min_samples'],
                               leaf_size=h['leaf_size'], algorithm=h['algorithm'])
        elif clustering_method == 'AgglomerativeClustering':
            h['n_clusters'] = c_conf.getint(clustering_method, 'n_clusters')
            h['affinity'] = c_conf.get(clustering_method, 'affinity')
            h['linkage'] = c_conf.get(clustering_method, 'linkage')
            clusterer = AgglomerativeClustering(n_clusters=h['n_clusters'],
                                                affinity=h['affinity'],
                                                linkage=h['linkage'])
        elif clustering_method == 'AffinityPropagation':
            clusterer = AffinityPropagation()
        elif clustering_method == 'SpectralClustering':
            clusterer = SpectralClustering()
        else:
            raise ValueError(clustering_method + " clustering method not supported.")

        for ac in separated_activations:
            if ac.shape == (0,):
                continue
            # Apply dimensionality reduction
            nb_activations = np.shape(ac)[1]
            if nb_activations > nb_dims:
                reduced_activations = self.reduce_dimensionality(ac, nb_dims=nb_dims, reduce=reduce)
            else:
                logger.info("Dimensionality of activations = %i less than nb_dims = %i. Not applying dimensionality "
                            "reduction.", nb_activations, nb_dims)
                reduced_activations = ac
            separated_reduced_activations.append(reduced_activations)

            # Get cluster assignments
            clusters = clusterer.fit_predict(reduced_activations)
            separated_clusters.append(clusters)

        # post processing
        detected_nb_clusters = np.unique(list(separated_clusters[0])).shape[0]
        print('Detected number of clusters for %s is: %d'
              % (clustering_method, detected_nb_clusters))
        # replace -1 with 1 if there is not exisiting any 1
        for cluster in separated_clusters:
            if detected_nb_clusters == 2:
                if cluster[cluster == -1].shape[0] != 0:
                    cluster[cluster == -1] = 1
            else:
                raise ValueError('Detected number of clusters are: %d' % detected_nb_clusters)
        return separated_clusters, separated_reduced_activations

    def reduce_dimensionality(self, activations, nb_dims=10, reduce='FastICA', is_scalar=False):
        """
        Reduces dimensionality of the activations provided using the specified number of dimensions and reduction technique.

        :param activations: Activations to be reduced
        :type activations: `numpy.ndarray`
        :param nb_dims: number of dimensions to reduce activation to via PCA
        :type nb_dims: `int`
        :param reduce: Method to perform dimensionality reduction, default is FastICA
        :type reduce: `str`
        :return: array with the activations reduced
        :rtype: `numpy.ndarray`
        """

        from sklearn.decomposition import FastICA, PCA
        from sklearn.preprocessing import MinMaxScaler

        if reduce == 'FastICA':
            projector = FastICA(n_components=nb_dims, max_iter=1000, tol=0.005)
        elif reduce == 'PCA':
            projector = PCA(n_components=nb_dims)

        else:
            raise ValueError(reduce + " dimensionality reduction method not supported.")

        # scaling activations between 0 and 1
        if is_scalar:
            scaler = MinMaxScaler(feature_range=[0, 1])
            activations = scaler.fit_transform(activations)

        reduced_activations = projector.fit_transform(activations)

        return reduced_activations
    def compute_centroids(self, reduced_X ,y_pred):
        from sklearn.neighbors.nearest_centroid import NearestCentroid
        clf = NearestCentroid()
        if y_pred.shape[0] != y_pred.tolist().count(0):  # check if there is more than one cluster
            clf.fit(reduced_X, y_pred)
        else:
            from scipy import ndimage
            clf.centroids_ = ndimage.measurements.center_of_mass(reduced_X)
        # print(clf.centroids_)
        return clf.centroids_

    def remove_poisons(self):
        # p, detected as c
        cp = self.x_train[self.y_train[:, 1] == 0][self.errors_by_class[0] == 3]
        # c, detected as c
        cc = self.x_train[self.y_train[:, 1] == 0][self.errors_by_class[0] == 1]
        # has building class
        hb = self.x_train[self.y_train[:, 1] == 1]
        cp_cc = np.append(cp, cc, axis=0)
        x_train_new = np.append(cp_cc, hb, axis=0)

        # p, detected as c
        cp = self.y_train[self.y_train[:, 1] == 0][self.errors_by_class[0] == 3]
        # c, detected as c
        cc = self.y_train[self.y_train[:, 1] == 0][self.errors_by_class[0] == 1]
        # has building class
        hb = self.y_train[self.y_train[:, 1] == 1]
        cp_cc = np.append(cp, cc, axis=0)
        y_train_new = np.append(cp_cc, hb, axis=0)

        return x_train_new, y_train_new

