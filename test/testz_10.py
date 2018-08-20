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

from nltk import cluster
from nltk.cluster import euclidean_distance

from nltk.cluster import cosine_distance
from numpy import array



#使用pca 降維度至2維,並且視覺化
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

    vectors = [array(f) for f in newData]

    # test k-means using the euclidean distance metric, 2 means and repeat
    # clustering 10 times with random seeds
    k=5
    clusterer = cluster.KMeansClusterer(k, cosine_distance, repeats=10)
    clusters = clusterer.cluster(vectors, True)

    print('Clustered:',end="")
    print(vectors)
    print('As:',end="")
    print(clusters)
    print('Means:',end="")
    print(clusterer.means())

    totaldata=array(vectors)
    print(totaldata)
    print(type(totaldata))
    print(type(totaldata[0]))
    label=array(clusters)
    print(label)
    center=[f.tolist() for f in clusterer.means()]

    trace_set=[]
    colornow=[]
    ds = totaldata[np.where(label == 0)]

    for i in range(k):
        #colornow.append(random.random(0,255))
        r=random.randrange(0,255)
        g=random.randrange(0,255)
        b=random.randrange(0,255)
        colornow.append('rgba('+str(r)+', '+str(g)+', '+str(b)+', .9)')
    for i in range(k):
        ds = totaldata[np.where(label == i)]

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



    centerx = [x for x, y in center]
    print(lx)
    centery = [y for x, y in center]
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

if __name__ == "__main__":
    main()