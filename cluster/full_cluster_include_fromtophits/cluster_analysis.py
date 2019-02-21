import numpy as np
import os
import random
from similarity import measure_similarity_expression,output_results
import glob
from sklearn.cluster import AgglomerativeClustering
from graph import calc_sim
import pandas
from scipy.stats import ttest_1samp
from mne.stats import permutation_t_test


#TODO: Can speed the whole thing up a lot if I dont measure the similarity of rejected genes. Parameter in measure_similarity_expressoin should take names that I want to run against, instead of full dataset.
#Should not be too hard.

def get_image_path(name,id):
   path = "/usr/share/ABI-expression-data/"
#	/usr/share/ABI-expression-data/Ermap/Ermap_P56_sagittal_68844875_200um/Ermap_P56_sagittal_68844875_200um_2dsurqec_mirrored.nii.gz
   path_s = glob.glob("/usr/share/ABI-expression-data/{}/{}_P56_sagittal_{}_200um/{}_P56_sagittal_{}_200um_2dsurqec_mirrored.nii.gz".format(name,name,id,name,id))
   path_c = glob.glob("/usr/share/ABI-expression-data/{}/{}_P56_coronal_{}_200um/{}_P56_coronal_{}_200um_2dsurqec.nii.gz".format(name,name,id,name,id))
   if len(path_s) == len(path_c) or len(path_s) > 1 or len(path_c)> 1 or (len(path_s) == 1 and len(path_c) == 1) or (len(path_s) != 1 and len(path_c) != 1): 
      raise ValueError("Idiot child!!!!!")
   if len(path_s) == 1: return path_s[0]
   if len(path_c) == 1: return path_c[0]


#get top hit of an add_result file, take file as an argument

def get_top_hit(path):
   with open(path) as f:
      for row in f:
         name = row.split("_")[0]
         id = row.split("_")[1].split(",")[0]
         path = row.split("'")[1]
         return name,id,path


#if not already present, run similarity all against it
def run_sim_single(name,id,img_path,exclude = None):
   print(name)
   print(id)
   base_path = "/home/gentoo/src/similarity/cluster/full_cluster/single_gene_results/"
   file_name = os.path.join(base_path,"{}_{}_similarity_results_experiment.csv".format(name,str(id)))
   if not os.path.exists(file_name):
      results = measure_similarity_expression(img_path,comparison = "experiment",metric = "GC",exclude = exclude)
      sorted_results = sorted(results.items(),key=lambda x: x[1][0])
      name_out = "{}_{}_similarity_results_experiment".format(name,str(id))
      output_path = base_path
      path_ = os.path.join(output_path,name_out)
      output_results(sorted_results,output_name= path_)

   return os.path.join(base_path,"{}_{}_similarity_results_experiment.csv".format(name,str(id)))


#rm top score, return path
def rm_top(path,name,id):
   with open(path, 'r') as fin:
      data = fin.read().splitlines(True)
   out = "{}_{}_rm_top.csv".format(name,str(id))
   out_path = os.path.join(os.path.dirname(path),out)
   with open(out_path, 'w') as fout:
      fout.writelines(data[1:])
   return out_path

#Here, remove the top git from the list, and run clustering against it


#now you have both lists, run the cluster analysis

#count number of iterations and output list of results after every step

def read_list(path):
   ranked_genes = list()
   with open(path) as f:
      for row in f:
         name = (row.split(",")[1]).split("[")[1]
         ranked_genes.append(name)
   return ranked_genes

def read_names(path):
   ranked_genes = list()
   with open(path) as f:
      for row in f:
         name = (row.split(",")[0])
         ranked_genes.append(name)
   return ranked_genes


def run_cluster(path,name,id):
   #from the single against all experiment
   exp = path
   out = "{}_{}_cluster.csv".format(name,str(id))
   out_path = os.path.join(os.path.dirname(exp),out)
   gene = read_names(exp)
   res = read_list(exp)
   resi = np.asarray(res)
   resi = resi.reshape(-1, 1)
   agg = AgglomerativeClustering(n_clusters = 2,linkage='average').fit(resi)
   lab = agg.labels_
   top_label = lab[0]

   with open(out_path,'w') as f:

      for i in range(0,len(lab)):
         f.write("{},{},{}\n".format(gene[i],resi[i],lab[i]))
   return out_path,top_label

def read_cluster(path,no,rejected_genes):
   cluster = list()
   with open(path) as f:
      for row in f:
         name = row.split(",")
         if int(name[2]) == int(no) :
            rejected_genes.append(name[0])
         else:
            cluster.append(name[0])
   return cluster,rejected_genes


def get_res(cluster,original,top_label,rejected_genes,done):

   exp = original

   original_values = read_list(exp)
   original_names = read_names(exp)
   remaining_cluster,rejected_genes = read_cluster(cluster,top_label,rejected_genes)
   for name in original_names:
      if any(name in s for s in remaining_cluster):
         if not any(name in s for s in rejected_genes):
            names = name.split("_")
            return names[0],names[1],rejected_genes,done
            break
      continue

      done = True
      return None,None,rejected_genes,done


def write_res(all_result_genes,input_additions,out_prefix = "1"):
   out = os.path.basename(input_additions).split("similarity_results_experiment")[0]
   out = out + "clustering_results" + "_" + out_prefix
   out_name = os.path.join(os.path.dirname(input_additions),out)
   with open(out_name,'w') as f:
      for gene in all_result_genes:
         f.write("{},{},{}\n".format(gene[0],gene[1],gene[2]))



input_additions = "2b_2a_2_4a_4b_4_similarity_results_experiment.csv"
rejected_genes = list()
done = False
exclude = list()
all_result_genes = list()
#get top hit, name, id, path
top_hit = get_top_hit(input_additions)
all_result_genes.append(top_hit)
rejected_genes.append(top_hit[0] + "_" + str(top_hit[1]))
exclude.append(str(top_hit[1]))


#make this optional, but append if sim is significantly lower than others
res_sim,sim = calc_sim(input_additions)

#numpy,pandas

print(sim)


col = sim.loc[[top_hit[0] + "_" + str(top_hit[1])]]
c = col.iloc[:,2:12]
h = c.to_dict(orient='list')
print(h)


col_values = col.iloc[:,2:12].values[0]
names = list(col)
names = names[2:12]

h = dict()

i = 0
for val in col_values:

   t,p = ttest_1samp(col_values,val)
   if p < 0.05 and val< np.mean(col_values):
      print(names[i])
      print(p)
   i = i + 1

col_values.shape = (np.shape(col_values)[0],1)

i = 0
for val in col_values:
   print("perm")
   values = np.subtract(col_values,val)
   T,p,HO = permutation_t_test(values,n_permutations = 1000)
   if p < 0.05 and val< np.mean(col_values):
      print(names[i])
      print(p)
   i = 1 + i



##choose column with the top hit, and then identify any row name with significant diff by t-tse


print(ldskjf)

#create or get path single expression value vs all,excluding itself
res_top_hit_path = run_sim_single(top_hit[0],top_hit[1],top_hit[2],exclude = exclude)

#rm identity (top hit) for clustering
#res_top_hit_rm_top_path = rm_top(res_top_hit_path,top_hit[0],top_hit[1])

#write the cluster file
#cluster,top_label = run_cluster(res_top_hit_rm_top_path,top_hit[0],top_hit[1])
cluster,top_label = run_cluster(res_top_hit_path,top_hit[0],top_hit[1])

#get the first hit that was not removed from cluster that appears in the original results
name,id,rejected_genes,done = get_res(cluster,input_additions,top_label,rejected_genes,done)
#print(rejected_genes)

#continue this cycle with this gene: find all similarity and so on thill the cluster only consists of one gene
path = get_image_path(name,id) 
all_result_genes.append([name,id,path])
rejected_genes.append(name + "_" + str(id))



i = 0
while not done:
   i = i + 1
   res_top_hit_path = run_sim_single(name,id,path)
   #res_top_hit_rm_top_path = rm_top(res_top_hit_path,top_hit[0],top_hit[1])
   #cluster,top_label = run_cluster(res_top_hit_rm_top_path,top_hit[0],top_hit[1])
   cluster,top_label = run_cluster(res_top_hit_path,top_hit[0],top_hit[1])
   name,id,rejected_genes,done = get_res(cluster,input_additions,top_label,rejected_genes,done)
   for gene in rejected_genes:
      exclude.append(str(gene.split("_")[1]))
   path = get_image_path(name,id)
   all_result_genes.append([name,id,path])
   #Just for debugging reasons, save all_result_genes at every iteration
   write_res(all_result_genes,input_additions,out_prefix = str(i))
   print((exclude))


