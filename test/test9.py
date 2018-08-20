# -*- coding: utf-8 -*-
import random

from sklearn.decomposition import PCA
import csv
import pandas as pd

import numpy as np
import plotly
from plotly.graph_objs import *

import re

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics.pairwise import euclidean_distances

#使用pca 降維度至2維,並且視覺化

def euli_dist(x,y):
    np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
def main():
    print("good")
    #df = pd.read_csv("D:\\\document_vector.csv", delimiter=',')
    #print(len(df))
    '''
    my_randoms = []

    for i in range(6797):
        my_randoms.append(random.randrange(1, 101, 1))

    print(my_randoms)
    '''
    #取出content
    title_list=[]
    content_list=[]
    show_list=[]
    content="D:\\NTUST\\人工智慧\\final\\csv\\headfile.csv"
    content_file = open(content, 'r',encoding='utf-8')
    content_filecsvCursor = csv.reader(content_file)
    next(content_filecsvCursor, None)  # skip the headers
    for row in content_filecsvCursor:
        #print(row[0]+"~~"+row[1])
        #title_list.append(row[3])
        #content_list.append(row[5])
        context = re.findall("(.{1,25})", row[5])
        rebuildcontext=""
        for w in context :
            rebuildcontext+=(w+"<br>")

        show_list.append(row[0]+"<br>"+row[3]+"<br>"+rebuildcontext)
        #print(rebuildcontext)
        #exit()

    #讀取document_vector
    fileName = "D:\\NTUST\\人工智慧\\final\\csv\\document_vector.csv"
    file = open(fileName, 'r', encoding='utf-8')
    filecsvCursor = csv.reader(file)
    high_dim_data=[]
    for row in filecsvCursor:
        rowlist = list(row)
        #print(rowlist)
        temp=[]
        count=0
        for item in rowlist:
            if(count>=2):
                temp.append(float(item))
                #print(float(item))
            count+=1
        high_dim_data.append(temp)

    #PCA降至2維
    pca = PCA(n_components=2)
    newData = pca.fit_transform(high_dim_data)#降至2維
    print(newData)
    lx = [x for x, y in newData]
    print(lx)
    ly = [y for x, y in newData]
    print(ly)

    output = open('D:\\NTUST\\人工智慧\\final\\csv\\pca_onlyCoordinate.txt', 'w', encoding = 'utf8')

    rr=0
    for now in newData:
        output.write(str(now[0])+","+str(now[1]))#+","+show_list[rr])
        output.write("\n")
        rr+=1
    output.close()


    # Create a trace
    trace = Scatter(
        x=lx,
        y=ly,
        mode='markers',
        marker = dict(
            size = 10,
            color = 'rgba(255, 182, 193, .9)',
            line = dict(
                width = 2,
            )
        ),
        text="good"
    )

    data = [trace]
    #plotly.offline.plot(data)



    #k-means
    k = 5
    kmeans = KMeans(n_clusters=k)

    kmeans.fit(newData)

    print(newData)
    print(newData[0])
    print(type(newData))
    print(type(newData[0]))
    labels = kmeans.labels_
    print("labels:")
    print(type(labels))
    print(labels)
    centroids = kmeans.cluster_centers_
    print("中心點座標")
    print(centroids)
    print(type(centroids))
    print(type(centroids[0]))
    '''
    for i in range(k):
        # select only data observations with cluster label == i
        ds = newData[np.where(labels == i)]
        # plot the data observations
        plt.plot(ds[:, 0], ds[:, 1], 'o')
        # plot the centroids
        lines = plt.plot(centroids[i, 0], centroids[i, 1], 'kx')
        # make the centroid x's bigger
        plt.setp(lines, ms=15.0)
        plt.setp(lines, mew=2.0)
    plt.show()
    '''
    print("~~~~~~~~~~~~~~~~")
    trace_set=[]
    #c=['rgba(255, 182, 193, .9)','rgba(0, 182, 193, .9)','rgba(0, 0, 0, .9)','rgba(25, 34, 66, .9)','rgba(122, 0, 3, .9)']
    colornow=[]
    for i in range(k):
        #colornow.append(random.random(0,255))
        r=random.randrange(0,255)
        g=random.randrange(0,255)
        b=random.randrange(0,255)
        colornow.append('rgba('+str(r)+', '+str(g)+', '+str(b)+', .9)')
    for i in range(k):
        print("cluster",end="")
        print(i,end="")
        print("=",end="")
        print(centroids[i])
        ds = newData[np.where(labels == i)]
        d = kmeans.transform(ds)[:, i]
        ind = np.argsort(d)[::1][:4]
        print(ds[ind])

        #centroids_array=np.array(centroids[i])
        #print(type(ds))
        #distance=[]
        #for now in ds:
        #   distance.append(euli_dist(now,centroids[i]))

        #print(centroids_array)
        # count distance
        #distance=euclidean_distances(ds,centroids_array)
        #exit()
        #distance.sort()
        #print(distance[:3])

        trace_now = Scatter(
            x=ds[:, 0],
            y=ds[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=colornow[i],
                line=dict(
                    width=2,
                )
            ),
            text=show_list
        )
        trace_set.append(trace_now)

    print("~~~~~~~~~~~~~~~~")


    centerx = [x for x, y in centroids]
    #print(lx)
    centery = [y for x, y in centroids]
    center_trace=Scatter(
        x=centerx,
        y=centery,
        mode='markers',
        marker=dict(
            size=10,
            color="rgba(0,0,0)",
            line=dict(
                width=30,
            )
        ),
    )
    trace_set.append(center_trace)
    plotly.offline.plot(trace_set)

    #centroids
    #newData
    #labels
    '''
    countid=0;
    centroids_id=[]
    print("中心點id")
    for coordinate in newData:
        for centerNow in centroids:
            if(coordinate[0]==centerNow[0] and coordinate[1]==centerNow[1]):
                centroids_id.append(countid)
        countid +=1;
    print(countid)
    print(centroids_id)
    '''



    '''
    #silhouette
    range_n_clusters = [2, 3, 4, 5, 6,7,8,9,10]
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(newData) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(newData)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(newData, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(newData, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(newData[:, 0], newData[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        # Labeling the clusters
        centers = clusterer.cluster_centers_

        print("center:",end='')
        #print(centers)


        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=200)

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()
         '''
if __name__ == "__main__":
    main()