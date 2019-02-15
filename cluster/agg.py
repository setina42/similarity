import os
import numpy as np
import glob
from sklearn.cluster import KMeans,AgglomerativeClustering

def read_list(path,rows):
   ranked_genes = list()
   i = 0
   with open(path) as f:
      for row in f:
         name = (row.split(",")[1]).split("[")[1]
         ranked_genes.append(name)
         i = i+ 1
         if i >= rows:break
   return ranked_genes

def read_names(path,rows):
   ranked_genes = list()
   i = 0
   with open(path) as f:
      for row in f:
         name = (row.split(",")[0])
         ranked_genes.append(name)
         i = i+ 1
         if i >= rows:break
   return ranked_genes


#for abi-corr
def read_list2(path,rows):
   ranked_genes = list()
   i = 0
   with open(path) as f:
      for row in f:
         if i == 0:
            i = 1
            continue
         name = (row.split(",")[1])
         ranked_genes.append(name)
         i = i+ 1
         if i >= rows:break
   return ranked_genes


#for reverse sorted lists
def read_list3(path,rows):
   all_genes = list()
   ranked_genes = list()
   i = 0
   with open(path) as f:
      for row in f:
         name = (row.split(",")[0]).split("_")[0]
         all_genes.append(name)
   for genes in reversed(all_genes):
      ranked_genes.append(genes)
      i = i + 1
      if i >= rows:break
   return ranked_genes

exp = glob.glob("*rm_top")[0]

gene = read_names(exp,25000)
res = read_list(exp,25000)
resi = np.asarray(res)

resi = resi.reshape(-1, 1)

#centroids = [res[0],res[1000],res[2000]]
#centroids = np.asarray(centroids)
#centroids = centroids.reshape(3,1)
agg = AgglomerativeClustering(n_clusters = 2,linkage='average').fit(resi)
lab = agg.labels_


#kmeans = KMeans(n_clusters=3,n_init = 1,init=centroids).fit(resi)
#kmeans = KMeans(n_clusters = 15,n_init=20).fit(resi)
#print(kmeans.labels_)

with open("cluster2_"+ exp,'w') as f:

   for i in range(0,len(lab)):
      f.write("{},{},{}\n".format(gene[i],resi[i],lab[i]))

agg = AgglomerativeClustering(n_clusters = 3,linkage='average').fit(resi)
lab = agg.labels_


#kmeans = KMeans(n_clusters=3,n_init = 1,init=centroids).fit(resi)
#kmeans = KMeans(n_clusters = 15,n_init=20).fit(resi)
#print(kmeans.labels_)

with open("cluster3_"+ exp,'w') as f:

   for i in range(0,len(lab)):
      f.write("{},{},{}\n".format(gene[i],resi[i],lab[i]))


