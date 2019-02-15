import os
import glob
import numpy
import glob
def read_list(path):
   ranked_genes = list()
   i = 0
   with open(path) as f:
      for row in f:
         name = (row.split(",")[1]).split("[")[1]
         ranked_genes.append(name)
         i = i+ 1
   return ranked_genes

def read_names(path):
   ranked_genes = list()
   i = 0
   with open(path) as f:
      for row in f:
         name = (row.split(",")[0])
         ranked_genes.append(name)
   return ranked_genes

def read_cluster(path,no):
   cluster = list()
   with open(path) as f:
      for row in f:
         name = row.split(",")
         if int(name[2]) == 1 or int(name[2]) == 1 : continue
         cluster.append(name[0])
   return cluster

exp = glob.glob("*_similarity_results_experiment.csv")[0]
#exp = "2_4_similarity_results_experiment.csv"

original_values = read_list(exp)
original_names = read_names(exp)

clus2 = glob.glob("cluster2*")[0]
clus3 = glob.glob("cluster3*")[0]
remaining_cluster = read_cluster(clus2,1)

for name in original_names:
   if any(name in s for s in remaining_cluster):
      print(name)
      break



