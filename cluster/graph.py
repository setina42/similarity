import os 
from similarity import ants_measure_similarity,create_mask
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

def read_list(path,count):
    paths = list()
    names = list()
    names.append("added")
    scores_to_add = list()
    i = 0
    with open(path) as f:
        for row in f:
            all = row.split(',')
            paths.append(all[2].split("'")[1])
            names.append(all[0])
            scores_to_add.append(all[1].split("[")[1])
            i = 1 + i
            if i > count: break
    return paths,names,scores_to_add


def add_single(exp,path_sinlge,paths,names,scores_to_add,co):
    with open(exp) as f:
        for row in f:
            if path_sinlge in row:
                all = row.split(",")
                paths.append(all[2].split("'")[1])
                names.append(all[0])
                scores_to_add.append(all[1].split("[")[1])
                co = co + 1
                return paths,names,scores_to_add,co


co = 7
#exp = "10a_8_10_8a_similarity_results_experiment.csv"
exp = glob.glob("*similarity_results_experiment.csv")[0]


paths,names,scores_to_add = read_list(exp,co)

#paths,names,scores_to_add,co = add_single(exp,"Golga7b_P56_coronal_76097699",paths,names,scores_to_add,co)
#paths,names,scores_to_add,co = add_single(exp,"Nudc_P56_coronal_2442_200um",paths,names,scores_to_add,co)


sim_res = np.ones((len(paths)+1,len(paths)+1))
sim_res = np.multiply(sim_res,-1)
np.fill_diagonal(sim_res,0)

scores_to_add.insert(0,0)
print(sim_res[:][0])

print(scores_to_add)
sim_res[0][:] = np.asarray(scores_to_add)
col  = np.asarray(scores_to_add)
col.shape=(co + 2)
sim_res[:,0] = col



for i in range(0,len(paths)):
    for j in range(i + 1,len(paths)):
            a = paths[i]
            b = paths[j]
            a_mask = create_mask(a,-1)
            b_mask = create_mask(b,-1)
            sim = ants_measure_similarity(a,b,mask_gene = b_mask,mask_map = a_mask,metric = "GC")
            #sim = ants_measure_similarity(a,b,metric = "MI")
            sim_res[i+1][j+1] = sim
            sim_res[j+1][i+1] = sim

sim_res = np.multiply(sim_res,-1)
print(np.sum(sim_res,axis=0))
print(np.sum(sim_res,axis=1))
print(sim_res)

sim = pd.DataFrame(data = sim_res,index=names,columns=names)
print(sim)


adj = np.ones((len(paths),len(paths)))
np.fill_diagonal(adj,0)




a = nx.from_numpy_matrix(sim_res)
nx.drawing.nx_agraph.write_dot(a,"here.dot")


print(a.adj)
b = nx.from_numpy_matrix(sim_res)

nx.draw(a)
nx.draw_networkx_edge_labels(a)

plt.draw()




plt.savefig("a.png")
nx.draw(b)
plt.draw()
plt.savefig("b.png")

