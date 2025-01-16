#standard libraries
import pandas as pd, numpy as np

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

#data transformation
from sklearn import preprocessing

#clusterization
from sklearn import cluster
from sklearn import mixture
#from sklearn import decomposition
#from sklearn import manifold

#quality control
from sklearn import metrics



def get_quantity_cancelled(data: pd.DataFrame):
    
    """
        > create a flag for the number of cancelled orders

    Parameters:
        data (DataFrame): table with transactions

    Output:
        column (Series): returns a column that indicates the number of subsequently cancelled items for each transaction,
                         i.e. if a transaction with a negative quantity of items does not have a counterparty, the flag is marked as NaN
    """

    #initialize the Series with zeros of the same dimensions as the main table
    quantity_cancelled = pd.Series(np.zeros(data.shape[0]),
                                  index=data.index)

    negative_quantity = data[(data['Quantity'] < 0)].copy()

    for index, col in negative_quantity.iterrows():
        #create the DataFrame with all counterparties
        df_test = data[(data['CustomerID'] == col['CustomerID']) &
                       (data['StockCode']  == col['StockCode']) & 
                       (data['InvoiceDate'] < col['InvoiceDate']) & 
                       (data['Quantity'] > 0)].copy()
        
        #return transactions have no counterparty
        if (df_test.shape[0] == 0):
            #set the column as a blank
            quantity_cancelled.loc[index] = np.nan

        # return transaction has exactly one counterparty
        #add the quantity of the cancelled product to the QuantityCanceled column
        elif (df_test.shape[0] == 1):
            index_order = df_test.index[0]
            quantity_cancelled.loc[index_order] = -col['Quantity']

        # return transaction has several counterparties
        #set the quantity of the cancelled product in the QuantityCanceled column for the purchase transaction,
        #in which the quantity of the product is greater than the quantity of the product in the return transaction.
        elif (df_test.shape[0] > 1):
            df_test.sort_index(axis=0,
                               ascending=False,
                               inplace=True)

            for ind, val in df_test.iterrows():
                if val['Quantity'] < -col['Quantity']:
                    continue
                quantity_cancelled.loc[ind] = -col['Quantity']
                break
    return quantity_cancelled



def plot_cluster_profile(grouped_data: pd.DataFrame,
                         n_clusters: int):
    
    """
        > visualize cluster profile as scatter polar chart

    Parameters:
        grouped_data (DataFrame): table grouped by cluster numbers with aggregated object characteristics
        n_clusters (int): number of clusters
        
    Output:
    
    """
    
    #normalize the grouped data by bringing it to a scale of 0-1
    scaler = preprocessing.MinMaxScaler()
    
    grouped_data = pd.DataFrame(scaler.fit_transform(grouped_data),
                                columns=grouped_data.columns)
    
    #create the list of features
    features = grouped_data.columns
    
    #create an empty figure
    fig = go.Figure()
    
    #visualize the scatter polar chart for each cluster
    for i in range(n_clusters):
        #create the scatter polar chart and add it to the general graph
        fig.add_trace(go.Scatterpolar(
            r=grouped_data.iloc[i].values,
            theta=features,
            fill='toself',
            name=f'Cluster {i}',
        ))
    
    #set the chart characteristics
    fig.update_layout(
        showlegend=True,
        autosize=False,
        width=800,
        height=800,
    )
    
    fig.show()
    
    

def clusters_by_silhouette(model_type: str,
                           data_scaled: np.ndarray,
                           start_number: int,
                           end_number: int):
    
    """
        > to display the silhouette score of each specified cluster and to visualize the results on a line graph

    Parameters:
        model_type (string): model type, i.e. k-means, em-algorithm, agglomerative clustering
        data_scaled (np.ndarray): standardized data set
        start_number: starting number of clusters
        end_number: ending number of clusters to loop through (+1)

    Output:
        text (string): cluster number with silhouette score
        line graph (matplotlib pyplot): rise or fall in score by cluster

    """

    clusters = []
    silhouette = []

    clusters_ward, clusters_complete, clusters_average, clusters_single = [], [], [], []
    silhouette_ward, silhouette_complete, silhouette_average, silhouette_single = [], [], [], []


    if model_type == 'k-means':
        for i in range(start_number, end_number):
            kmeans_model = cluster.KMeans(n_clusters=i,
                                        random_state=42,
                                        init='k-means++').fit(data_scaled)

            y_pred = kmeans_model.labels_

            #add results
            clusters.append(i)
            silhouette.append(round(metrics.silhouette_score(data_scaled, y_pred), 3))
            print(f'Clusters: {i}, Silhouette Score: {round(metrics.silhouette_score(data_scaled, y_pred), 3)}')

    elif model_type == 'em-algorithm':
         for i in range(start_number, end_number):
            em_model = mixture.GaussianMixture(n_components=i,
                                               random_state=42)

            predictions = em_model.fit_predict(data_scaled)

            #add results
            clusters.append(i)
            silhouette.append(round(metrics.silhouette_score(data_scaled, predictions), 3))
            print(f'Clusters: {i}, Silhouette Score: {round(metrics.silhouette_score(data_scaled, predictions), 3)}')

    elif model_type == 'agglomerative_clustering':
        for i in range(start_number, end_number):
            for l in ['ward', 'complete', 'average', 'single']:
                    aggl_model = cluster.AgglomerativeClustering(n_clusters=i,
                                                                linkage=l)

                    y_pred = aggl_model.fit_predict(data_scaled)

                    #add results
                    if l == 'ward':
                        clusters_ward.append(i)
                        silhouette_ward.append(round(metrics.silhouette_score(data_scaled, y_pred), 3))
                        print(f'Clusters: {i}, Linkage: {l}, Silhouette Score: {round(metrics.silhouette_score(data_scaled, y_pred), 3)}')
                    elif l == 'complete':
                        clusters_complete.append(i)
                        silhouette_complete.append(round(metrics.silhouette_score(data_scaled, y_pred), 3))
                        print(f'Clusters: {i}, Linkage: {l}, Silhouette Score: {round(metrics.silhouette_score(data_scaled, y_pred), 3)}')
                    elif l == 'average':
                        clusters_average.append(i)
                        silhouette_average.append(round(metrics.silhouette_score(data_scaled, y_pred), 3))
                        print(f'Clusters: {i}, Linkage: {l}, Silhouette Score: {round(metrics.silhouette_score(data_scaled, y_pred), 3)}')
                    elif l == 'single':
                        clusters_single.append(i)
                        silhouette_single.append(round(metrics.silhouette_score(data_scaled, y_pred), 3))
                        print(f'Clusters: {i}, Linkage: {l}, Silhouette Score: {round(metrics.silhouette_score(data_scaled, y_pred), 3)}')
                    else:
                        print('errored')

    else:
         print('incorrect model type, i.e. only use k-means, em-algorithm, or agglomerative_clustering')

    if model_type == 'k-means' or model_type == 'em-algorithm':
        def create_the_graph(clusters_lst: list,
                             silhouette_lst: list,
                             model_type: str,
                             start_number: int,
                             end_number: int):
                
                #create the line graph
                plt.xlabel('Cluster', fontsize=11)
                plt.ylabel('Silhouette Score', fontsize=11)
                plt.plot(clusters_lst,
                         silhouette_lst,
                         'xb-')
                plt.title(f'{model_type.capitalize()}: Silhouette Score for Clusters {start_number} to {end_number-1}')
        
        return(create_the_graph(clusters, silhouette, model_type, start_number, end_number))

    elif model_type == 'agglomerative_clustering':
        def create_the_graph_for_aggl_clust(clusters_lst_wrd: list,
                                            clusters_lst_cmpl: list,
                                            clusters_lst_av: list,
                                            clusters_lst_sngl: list,
                                            silhouette_lst_wrd: list,
                                            silhouette_lst_cmpl: list,
                                            silhouette_lst_av: list,
                                            silhouette_lst_sngl: list,
                                            model_type: str,
                                            start_number: int,
                                            end_number: int
                                            ):
            #create the line graph
            plt.xlabel('Cluster', fontsize=11)
            plt.ylabel('Silhouette Score', fontsize=11)
              
            #blue: ward
            plt.plot(clusters_lst_wrd,
                     silhouette_lst_wrd,
                     'xb-')
            
            #red: complete
            plt.plot(clusters_lst_cmpl,
                     silhouette_lst_cmpl,
                     'xr-')
            #yellow: average
            plt.plot(clusters_lst_av,
                     silhouette_lst_av,
                     'xy-')
            #black: single
            plt.plot(clusters_lst_sngl,
                     silhouette_lst_sngl,
                     'xk-')
            
            plt.legend(('linkage = ward', 'linkage = complete', 'linkage = average', 'linkage = single'))
            plt.title(f'{model_type.capitalize()}: Silhouette Score for Clusters {start_number} to {end_number-1}');
        
        return(create_the_graph_for_aggl_clust(clusters_ward, clusters_complete, clusters_average, clusters_single,
                                               silhouette_ward, silhouette_complete, silhouette_average, silhouette_single,
                                               model_type, start_number, end_number))



def clusters_by_calinski_harabasz(model_type: str,
                                  data_scaled: np.ndarray,
                                  start_number: int,
                                  end_number: int):
    
    """
        > to display the calinski_harabasz score of each specified cluster and to visualize the results on a line graph

    Parameters:
        model_type (string): model type, i.e. k-means, em-algorithm, agglomerative clustering
        data_scaled (np.ndarray): standardized data set
        start_number: starting number of clusters
        end_number: ending number of clusters to loop through (+1)

    Output:
        text (string): cluster number with silhouette score
        line graph (matplotlib pyplot): rise or fall in score by cluster

    """

    clusters = []
    calinski_harabasz = []

    clusters_ward, clusters_complete, clusters_average, clusters_single = [], [], [], []
    ch_ward, ch_complete, ch_average, ch_single = [], [], [], []


    if model_type == 'k-means':
        for i in range(start_number, end_number):
            kmeans_model = cluster.KMeans(n_clusters=i,
                                          random_state=42,
                                          init='k-means++').fit(data_scaled)

            y_pred = kmeans_model.labels_

            #add results
            clusters.append(i)
            calinski_harabasz.append(round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3))
            print(f'Clusters: {i}, Calinski-Harabasz Score: {round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3)}')

    elif model_type == 'em-algorithm':
         for i in range(start_number, end_number):
            em_model = mixture.GaussianMixture(n_components=i,
                                               random_state=42)

            predictions = em_model.fit_predict(data_scaled)

            #add results
            clusters.append(i)
            calinski_harabasz.append(round(metrics.calinski_harabasz_score(data_scaled, predictions), 3))
            print(f'Clusters: {i}, Calinski-Harabasz Score: {round(metrics.calinski_harabasz_score(data_scaled, predictions), 3)}')

    elif model_type == 'agglomerative_clustering':
        for i in range(start_number, end_number):
            for l in ['ward', 'complete', 'average', 'single']:
                    aggl_model = cluster.AgglomerativeClustering(n_clusters=i,
                                                                linkage=l)

                    y_pred = aggl_model.fit_predict(data_scaled)

                    #add results
                    if l == 'ward':
                        clusters_ward.append(i)
                        ch_ward.append(round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3))
                        print(f'Clusters: {i}, Linkage: {l}, Calinski-Harabasz Score: {round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3)}')
                    elif l == 'complete':
                        clusters_complete.append(i)
                        ch_complete.append(round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3))
                        print(f'Clusters: {i}, Linkage: {l}, Calinski-Harabasz Score: {round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3)}')
                    elif l == 'average':
                        clusters_average.append(i)
                        ch_average.append(round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3))
                        print(f'Clusters: {i}, Linkage: {l}, Calinski-Harabasz Score: {round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3)}')
                    elif l == 'single':
                        clusters_single.append(i)
                        ch_single.append(round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3))
                        print(f'Clusters: {i}, Linkage: {l}, Calinski-Harabasz Score: {round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3)}')
                    else:
                        print('errored')

    else:
         print('incorrect model type, i.e. only use k-means, em-algorithm, or agglomerative_clustering')

    if model_type == 'k-means' or model_type == 'em-algorithm':
        def create_the_graph(clusters_lst: list,
                             ch_lst: list,
                             model_type: str,
                             start_number: int,
                             end_number: int):
                
                #create the line graph
                plt.xlabel('Cluster', fontsize=11)
                plt.ylabel('Calinski-Harabasz Score', fontsize=11)
                plt.plot(clusters_lst,
                         ch_lst,
                         'xb-')
                plt.title(f'{model_type.capitalize()}: Calinski-Harabasz Score for Clusters {start_number} to {end_number-1}')
        
        return(create_the_graph(clusters, calinski_harabasz, model_type, start_number, end_number))

    elif model_type == 'agglomerative_clustering':
        def create_the_graph_for_aggl_clust(clusters_lst_wrd: list,
                                            clusters_lst_cmpl: list,
                                            clusters_lst_av: list,
                                            clusters_lst_sngl: list,
                                            ch_lst_wrd: list,
                                            ch_lst_cmpl: list,
                                            ch_lst_av: list,
                                            ch_lst_sngl: list,
                                            model_type: str,
                                            start_number: int,
                                            end_number: int
                                            ):
            #create the line graph
            plt.xlabel('Cluster', fontsize=11)
            plt.ylabel('Calinski-Harabasz Score', fontsize=11)
              
            #blue: ward
            plt.plot(clusters_lst_wrd,
                     ch_lst_wrd,
                     'xb-')
            
            #red: complete
            plt.plot(clusters_lst_cmpl,
                     ch_lst_cmpl,
                     'xr-')
            #yellow: average
            plt.plot(clusters_lst_av,
                     ch_lst_av,
                     'xy-')
            #black: single
            plt.plot(clusters_lst_sngl,
                     ch_lst_sngl,
                     'xk-')
            
            plt.legend(('linkage = ward', 'linkage = complete', 'linkage = average', 'linkage = single'))
            plt.title(f'{model_type.capitalize()}: Calinski-Harabasz Score for Clusters {start_number} to {end_number-1}');
        
        return(create_the_graph_for_aggl_clust(clusters_ward, clusters_complete, clusters_average, clusters_single,
                                               ch_ward, ch_complete, ch_average, ch_single,
                                               model_type, start_number, end_number))



def clusters_by_davies_bouldin(model_type: str,
                               data_scaled: np.ndarray,
                               start_number: int,
                               end_number: int):
    
    """
        > to display the davies-bouldin score of each specified cluster and to visualize the results on a line graph

    Parameters:
        model_type (string): model type, i.e. k-means, em-algorithm, agglomerative clustering
        data_scaled (np.ndarray): standardized data set
        start_number: starting number of clusters
        end_number: ending number of clusters to loop through (+1)

    Output:
        text (string): cluster number with silhouette score
        line graph (matplotlib pyplot): rise or fall in score by cluster

    """

    clusters = []
    davies_bouldin = []

    clusters_ward, clusters_complete, clusters_average, clusters_single = [], [], [], []
    db_ward, db_complete, db_average, db_single = [], [], [], []


    if model_type == 'k-means':
        for i in range(start_number, end_number):
            kmeans_model = cluster.KMeans(n_clusters=i,
                                          random_state=42,
                                          init='k-means++').fit(data_scaled)

            y_pred = kmeans_model.labels_

            #add results
            clusters.append(i)
            davies_bouldin.append(round(metrics.davies_bouldin_score(data_scaled, y_pred), 3))
            print(f'Clusters: {i}, Davies-Bouldin Score: {round(metrics.davies_bouldin_score(data_scaled, y_pred), 3)}')

    elif model_type == 'em-algorithm':
         for i in range(start_number, end_number):
            em_model = mixture.GaussianMixture(n_components=i,
                                               random_state=42)

            predictions = em_model.fit_predict(data_scaled)

            #add results
            clusters.append(i)
            davies_bouldin.append(round(metrics.davies_bouldin_score(data_scaled, predictions), 3))
            print(f'Clusters: {i}, Davies-Bouldin Score: {round(metrics.davies_bouldin_score(data_scaled, predictions), 3)}')

    elif model_type == 'agglomerative_clustering':
        for i in range(start_number, end_number):
            for l in ['ward', 'complete', 'average', 'single']:
                    aggl_model = cluster.AgglomerativeClustering(n_clusters=i,
                                                                linkage=l)

                    y_pred = aggl_model.fit_predict(data_scaled)

                    #add results
                    if l == 'ward':
                        clusters_ward.append(i)
                        db_ward.append(round(metrics.davies_bouldin_score(data_scaled, y_pred), 3))
                        print(f'Clusters: {i}, Linkage: {l}, Davies-Bouldin Score: {round(metrics.davies_bouldin_score(data_scaled, y_pred), 3)}')
                    elif l == 'complete':
                        clusters_complete.append(i)
                        db_complete.append(round(metrics.davies_bouldin_score(data_scaled, y_pred), 3))
                        print(f'Clusters: {i}, Linkage: {l}, Davies-Bouldin Score: {round(metrics.davies_bouldin_score(data_scaled, y_pred), 3)}')
                    elif l == 'average':
                        clusters_average.append(i)
                        db_average.append(round(metrics.davies_bouldin_score(data_scaled, y_pred), 3))
                        print(f'Clusters: {i}, Linkage: {l}, Davies-Bouldin Score: {round(metrics.davies_bouldin_score(data_scaled, y_pred), 3)}')
                    elif l == 'single':
                        clusters_single.append(i)
                        db_single.append(round(metrics.davies_bouldin_score(data_scaled, y_pred), 3))
                        print(f'Clusters: {i}, Linkage: {l}, Davies-Bouldin Score: {round(metrics.davies_bouldin_score(data_scaled, y_pred), 3)}')
                    else:
                        print('errored')

    else:
         print('incorrect model type, i.e. only use k-means, em-algorithm, or agglomerative_clustering')

    if model_type == 'k-means' or model_type == 'em-algorithm':
        def create_the_graph(clusters_lst: list,
                             db_lst: list,
                             model_type: str,
                             start_number: int,
                             end_number: int):
                
                #create the line graph
                plt.xlabel('Cluster', fontsize=11)
                plt.ylabel('Davies-Bouldin Score', fontsize=11)
                plt.plot(clusters_lst,
                         db_lst,
                         'xb-')
                plt.title(f'{model_type.capitalize()}: Davies-Bouldin Score for Clusters {start_number} to {end_number-1}')
        
        return(create_the_graph(clusters, davies_bouldin, model_type, start_number, end_number))

    elif model_type == 'agglomerative_clustering':
        def create_the_graph_for_aggl_clust(clusters_lst_wrd: list,
                                            clusters_lst_cmpl: list,
                                            clusters_lst_av: list,
                                            clusters_lst_sngl: list,
                                            db_lst_wrd: list,
                                            db_lst_cmpl: list,
                                            db_lst_av: list,
                                            db_lst_sngl: list,
                                            model_type: str,
                                            start_number: int,
                                            end_number: int
                                            ):
            #create the line graph
            plt.xlabel('Cluster', fontsize=11)
            plt.ylabel('Davies-Bouldin Score', fontsize=11)
              
            #blue: ward
            plt.plot(clusters_lst_wrd,
                     db_lst_wrd,
                     'xb-')
            
            #red: complete
            plt.plot(clusters_lst_cmpl,
                     db_lst_cmpl,
                     'xr-')
            #yellow: average
            plt.plot(clusters_lst_av,
                     db_lst_av,
                     'xy-')
            #black: single
            plt.plot(clusters_lst_sngl,
                     db_lst_sngl,
                     'xk-')
            
            plt.legend(('linkage = ward', 'linkage = complete', 'linkage = average', 'linkage = single'))
            plt.title(f'{model_type.capitalize()}: Davies-Bouldin Score for Clusters {start_number} to {end_number-1}');
        
        return(create_the_graph_for_aggl_clust(clusters_ward, clusters_complete, clusters_average, clusters_single,
                                               db_ward, db_complete, db_average, db_single,
                                               model_type, start_number, end_number))



def best_result_by_silhouette(model_type: str,
                              data_scaled: np.ndarray,
                              start_number: int,
                              end_number: int):
    
    """
        > to find the highest (best) silhouette score based on the range of clusters specified
        > the starting score value = 0

    Parameters:
        model_type (string): model type, i.e. k-means, em-algorithm, agglomerative clustering
        data_scaled (np.ndarray): standardized data set
        start_number: starting number of clusters
        end_number: ending number of clusters to loop through (+1)

    Output:
        text (string): cluster number with (best) silhouette score

    """

    clusters = []
    silhouette = []

    clusters_ward, clusters_complete, clusters_average, clusters_single = [], [], [], []
    silhouette_ward, silhouette_complete, silhouette_average, silhouette_single = [], [], [], []

    if model_type == 'k-means' or model_type == 'em-algorithm':

        if model_type == 'k-means':
            for i in range(start_number, end_number):
                kmeans_model = cluster.KMeans(n_clusters=i,
                                              random_state=42,
                                              init='k-means++').fit(data_scaled)

                y_pred = kmeans_model.labels_

                #add results
                clusters.append(i)
                silhouette.append(round(metrics.silhouette_score(data_scaled, y_pred), 3))

        elif model_type == 'em-algorithm':
            for i in range(start_number, end_number):
                em_model = mixture.GaussianMixture(n_components=i,
                                                   random_state=42)

                predictions = em_model.fit_predict(data_scaled)

                #add results
                clusters.append(i)
                silhouette.append(round(metrics.silhouette_score(data_scaled, predictions), 3))

        #find the highest score
        highest_score = np.amax(silhouette,
                                axis=0)

        #find the index (position) of the highest score in the unpacked list
        temp = max(silhouette)
        highest_score_idx = [i for i, j in enumerate(silhouette) if j == temp]

        #find the cluster and linkage corresponding to the smallest score
        highest_cluster_by_idx = [clusters[i] for i in (highest_score_idx)]

        return(f'Clusters: {highest_cluster_by_idx[0]}, Silhouette Score: {round(highest_score, 3)}')
    
    
    elif model_type == 'agglomerative_clustering':
        for i in range(start_number, end_number):
            for l in ['ward', 'complete', 'average', 'single']:
                    aggl_model = cluster.AgglomerativeClustering(n_clusters=i,
                                                                 linkage=l)

                    y_pred = aggl_model.fit_predict(data_scaled)

                    #add results
                    if l == 'ward':
                        clusters_ward.append(i)
                        silhouette_ward.append(round(metrics.silhouette_score(data_scaled, y_pred), 3))
                    elif l == 'complete':
                        clusters_complete.append(i)
                        silhouette_complete.append(round(metrics.silhouette_score(data_scaled, y_pred), 3))
                    elif l == 'average':
                        clusters_average.append(i)
                        silhouette_average.append(round(metrics.silhouette_score(data_scaled, y_pred), 3))
                    elif l == 'single':
                        clusters_single.append(i)
                        silhouette_single.append(round(metrics.silhouette_score(data_scaled, y_pred), 3))
    
        #create the duplicate elements for linkage
        n_dupl = len(range(start_number, end_number))
        w_link, c_link, a_link, s_link = ('ward,' * n_dupl).split(','), ('complete,' * n_dupl).split(','), ('average,' * n_dupl).split(','), \
            ('single,' * n_dupl).split(',')
        w_link, c_link, a_link, s_link = w_link[:-1], c_link[:-1], a_link[:-1], s_link[:-1]

        #combine multiple lists into a list of lists
        clusters_all = clusters_ward, clusters_complete, clusters_average, clusters_single
        scores_all = silhouette_ward, silhouette_complete, silhouette_average, silhouette_single
        linkage_all = w_link, c_link, a_link, s_link

        def flatten(lst_of_lsts: list):
            '''
                > unpack a list within a list, thus convert multiple lists into one
            '''
            return [x for xs in lst_of_lsts for x in xs]

        #unpack the lists
        clusters_all = flatten(clusters_all)
        scores_all = flatten(scores_all)
        linkage_all = flatten(linkage_all)


        #find the highest score
        highest_score = np.amax(scores_all,
                                axis=0)

        #find the index (position) of the highest score in the unpacked list
        temp = max(scores_all)
        highest_score_idx = [i for i, j in enumerate(scores_all) if j == temp]

        #find the cluster and linkage corresponding to the smallest score
        highest_cluster_by_idx = [clusters_all[i] for i in (highest_score_idx)]
        highest_linkage_by_idx = [linkage_all[i] for i in (highest_score_idx)]
    
        return(f'Clusters: {highest_cluster_by_idx[0]}, Linkage: {highest_linkage_by_idx[0].capitalize()}, \
               Silhouette Score: {round(highest_score, 3)}')

    else:
        return(f'incorrect model type, i.e. only use k-means, em-algorithm, or agglomerative_clustering')



def best_result_by_calinski_harabasz(model_type: str,
                                     data_scaled: np.ndarray,
                                     start_number: int,
                                     end_number: int):
    
    """
        > to find the highest (best) calinski-harabasz score based on the range of clusters specified
        > the starting score value = 0

    Parameters:
        model_type (string): model type, i.e. k-means, em-algorithm, agglomerative clustering
        data_scaled (np.ndarray): standardized data set
        start_number: starting number of clusters
        end_number: ending number of clusters to loop through (+1)

    Output:
        text (string): cluster number with (best) calinski-harabasz score
        
    """

    clusters = []
    calinski_harabasz = []

    clusters_ward, clusters_complete, clusters_average, clusters_single = [], [], [], []
    calinski_harabasz_ward, calinski_harabasz_complete, calinski_harabasz_average, calinski_harabasz_single = [], [], [], []

    if model_type == 'k-means' or model_type == 'em-algorithm':

        if model_type == 'k-means':
            for i in range(start_number, end_number):
                kmeans_model = cluster.KMeans(n_clusters=i,
                                              random_state=42,
                                              init='k-means++').fit(data_scaled)

                y_pred = kmeans_model.labels_

                #add results
                clusters.append(i)
                calinski_harabasz.append(round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3))

        elif model_type == 'em-algorithm':
            for i in range(start_number, end_number):
                em_model = mixture.GaussianMixture(n_components=i,
                                                   random_state=42)

                predictions = em_model.fit_predict(data_scaled)

                #add results
                clusters.append(i)
                calinski_harabasz.append(round(metrics.calinski_harabasz_score(data_scaled, predictions), 3))

        #find the highest score
        highest_score = np.amax(calinski_harabasz,
                                axis=0)

        #find the index (position) of the highest score in the unpacked list
        temp = max(calinski_harabasz)
        highest_score_idx = [i for i, j in enumerate(calinski_harabasz) if j == temp]

        #find the cluster and linkage corresponding to the smallest score
        highest_cluster_by_idx = [clusters[i] for i in (highest_score_idx)]

        return(f'Clusters: {highest_cluster_by_idx[0]}, Calinski-Harabasz Score: {round(highest_score, 3)}')
    
    
    elif model_type == 'agglomerative_clustering':
        for i in range(start_number, end_number):
            for l in ['ward', 'complete', 'average', 'single']:
                    aggl_model = cluster.AgglomerativeClustering(n_clusters=i,
                                                                 linkage=l)

                    y_pred = aggl_model.fit_predict(data_scaled)

                    #add results
                    if l == 'ward':
                        clusters_ward.append(i)
                        calinski_harabasz_ward.append(round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3))
                    elif l == 'complete':
                        clusters_complete.append(i)
                        calinski_harabasz_complete.append(round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3))
                    elif l == 'average':
                        clusters_average.append(i)
                        calinski_harabasz_average.append(round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3))
                    elif l == 'single':
                        clusters_single.append(i)
                        calinski_harabasz_single.append(round(metrics.calinski_harabasz_score(data_scaled, y_pred), 3))
    
        #create the duplicate elements for linkage
        n_dupl = len(range(start_number, end_number))
        w_link, c_link, a_link, s_link = ('ward,' * n_dupl).split(','), ('complete,' * n_dupl).split(','), ('average,' * n_dupl).split(','), \
            ('single,' * n_dupl).split(',')
        w_link, c_link, a_link, s_link = w_link[:-1], c_link[:-1], a_link[:-1], s_link[:-1]

        #combine multiple lists into a list of lists
        clusters_all = clusters_ward, clusters_complete, clusters_average, clusters_single
        scores_all = calinski_harabasz_ward, calinski_harabasz_complete, calinski_harabasz_average, calinski_harabasz_single
        linkage_all = w_link, c_link, a_link, s_link

        def flatten(lst_of_lsts: list):
            '''
                > unpack a list within a list, thus convert multiple lists into one
            '''
            return [x for xs in lst_of_lsts for x in xs]

        #unpack the lists
        clusters_all = flatten(clusters_all)
        scores_all = flatten(scores_all)
        linkage_all = flatten(linkage_all)


        #find the highest score
        highest_score = np.amax(scores_all,
                                axis=0)

        #find the index (position) of the highest score in the unpacked list
        temp = max(scores_all)
        highest_score_idx = [i for i, j in enumerate(scores_all) if j == temp]

        #find the cluster and linkage corresponding to the smallest score
        highest_cluster_by_idx = [clusters_all[i] for i in (highest_score_idx)]
        highest_linkage_by_idx = [linkage_all[i] for i in (highest_score_idx)]
    
        return(f'Clusters: {highest_cluster_by_idx[0]}, Linkage: {highest_linkage_by_idx[0].capitalize()}, \
               Calinski-Harabasz Score: {round(highest_score, 3)}')

    else:
        return(f'incorrect model type, i.e. only use k-means, em-algorithm, or agglomerative_clustering')
    


def best_result_by_davies_bouldin(model_type: str,
                                  data_scaled: np.ndarray,
                                  start_number: int,
                                  end_number: int):
    
    """
        > to find the lowest (best) davies-bouldin score based on the range of clusters specified

    Parameters:
        model_type (string): model type, i.e. k-means, em-algorithm, agglomerative clustering
        data_scaled (np.ndarray): standardized data set
        start_number: starting number of clusters
        end_number: ending number of clusters to loop through (+1)

    Output:
        text (string): cluster number with (best) davies-bouldin score

    """

    clusters = []
    davies_bouldin = []

    clusters_ward, clusters_complete, clusters_average, clusters_single = [], [], [], []
    davies_bouldin_ward, davies_bouldin_complete, davies_bouldin_average, davies_bouldin_single = [], [], [], []

    if model_type == 'k-means' or model_type == 'em-algorithm':

        if model_type == 'k-means':
            for i in range(start_number, end_number):
                kmeans_model = cluster.KMeans(n_clusters=i,
                                              random_state=42,
                                              init='k-means++').fit(data_scaled)

                y_pred = kmeans_model.labels_

                #add results
                clusters.append(i)
                davies_bouldin.append(round(metrics.davies_bouldin_score(data_scaled, y_pred), 3))

        elif model_type == 'em-algorithm':
            for i in range(start_number, end_number):
                em_model = mixture.GaussianMixture(n_components=i,
                                                   random_state=42)

                predictions = em_model.fit_predict(data_scaled)

                #add results
                clusters.append(i)
                davies_bouldin.append(round(metrics.davies_bouldin_score(data_scaled, predictions), 3))

        #find the smallest score
        smallest_score = np.amin(davies_bouldin,
                                 axis=0)

        #find the index (position) of the smallest score in the unpacked list
        temp = min(davies_bouldin)
        smallest_score_idx = [i for i, j in enumerate(davies_bouldin) if j == temp]

        #find the cluster and linkage corresponding to the smallest score
        smallest_cluster_by_idx = [clusters[i] for i in (smallest_score_idx)]

        return(f'Clusters: {smallest_cluster_by_idx[0]}, Davies-Bouldin Score: {round(smallest_score, 3)}')
    
    
    elif model_type == 'agglomerative_clustering':
        for i in range(start_number, end_number):
            for l in ['ward', 'complete', 'average', 'single']:
                    aggl_model = cluster.AgglomerativeClustering(n_clusters=i,
                                                                 linkage=l)

                    y_pred = aggl_model.fit_predict(data_scaled)

                    #add results
                    if l == 'ward':
                        clusters_ward.append(i)
                        davies_bouldin_ward.append(round(metrics.davies_bouldin_score(data_scaled, y_pred), 3))
                    elif l == 'complete':
                        clusters_complete.append(i)
                        davies_bouldin_complete.append(round(metrics.davies_bouldin_score(data_scaled, y_pred), 3))
                    elif l == 'average':
                        clusters_average.append(i)
                        davies_bouldin_average.append(round(metrics.davies_bouldin_score(data_scaled, y_pred), 3))
                    elif l == 'single':
                        clusters_single.append(i)
                        davies_bouldin_single.append(round(metrics.davies_bouldin_score(data_scaled, y_pred), 3))
    
        #create the duplicate elements for linkage
        n_dupl = len(range(start_number, end_number))
        w_link, c_link, a_link, s_link = ('ward,' * n_dupl).split(','), ('complete,' * n_dupl).split(','), ('average,' * n_dupl).split(','), \
            ('single,' * n_dupl).split(',')
        w_link, c_link, a_link, s_link = w_link[:-1], c_link[:-1], a_link[:-1], s_link[:-1]

        #combine multiple lists into a list of lists
        clusters_all = clusters_ward, clusters_complete, clusters_average, clusters_single
        scores_all = davies_bouldin_ward, davies_bouldin_complete, davies_bouldin_average, davies_bouldin_single
        linkage_all = w_link, c_link, a_link, s_link

        def flatten(lst_of_lsts: list):
            '''
                > unpack a list within a list, thus convert multiple lists into one
            '''
            return [x for xs in lst_of_lsts for x in xs]

        #unpack the lists
        clusters_all = flatten(clusters_all)
        scores_all = flatten(scores_all)
        linkage_all = flatten(linkage_all)


        #find the smallest score
        smallest_score = np.amin(scores_all,
                                 axis=0)

        #find the index (position) of the smallest score in the unpacked list
        temp = min(scores_all)
        smallest_score_idx = [i for i, j in enumerate(scores_all) if j == temp]

        #find the cluster and linkage corresponding to the smallest score
        smallest_cluster_by_idx = [clusters_all[i] for i in (smallest_score_idx)]
        smallest_linkage_by_idx = [linkage_all[i] for i in (smallest_score_idx)]
    
        return(f'Clusters: {smallest_cluster_by_idx[0]}, Linkage: {smallest_linkage_by_idx[0].capitalize()}, \
               Davies-Bouldin Score: {round(smallest_score, 3)}')

    else:
        return(f'incorrect model type, i.e. only use k-means, em-algorithm, or agglomerative_clustering')