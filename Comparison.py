from rbo import rbo
import os
import pandas
from scipy.stats import kendalltau, spearmanr,weightedtau, spearmanr,wilcoxon
import scipy
from math import sqrt


#for expression lsits
def read_list(path,rows):
   ranked_genes = list()
   i = 0
   with open(path) as f:
      for row in f:
         name = (row.split(",")[0]).split("_")[0]
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


#generalized kendall tau:

def generalized_kendall_tau(list_one,list_two,count):
   all = list_one + list_two
   all = list(set(all))

   generalized_kendall_score = 0
   pairs=0
   for a in range(0,len(all)):

      for b in range((a+1),len(all)):
         pairs += 1
         i = all[a]
         j = all[b]

         try:
            ind = list_one.index(i)
            i_is_in_list_one=True
            rank_list_one_i = ind
         except ValueError:
            i_is_in_list_one = False

         try:
            ind = list_two.index(i)
            i_is_in_list_two=True
            rank_list_two_i = ind
         except ValueError:
            i_is_in_list_two = False

         try:
            ind = list_one.index(j)
            j_is_in_list_one=True
            rank_list_one_j = ind
         except ValueError:
            j_is_in_list_one = False

         try:
            ind = list_two.index(j)
            j_is_in_list_two=True
            rank_list_two_j = ind
         except ValueError:
            j_is_in_list_two = False



      #case 1 : i and j are in both rankings

         if i_is_in_list_one and i_is_in_list_two and j_is_in_list_one and j_is_in_list_two:
            if (rank_list_one_i > rank_list_one_j) and (rank_list_two_i > rank_list_two_j) or (rank_list_one_i < rank_list_one_j) and (rank_list_two_i < rank_list_two_j):
               generalized_kendall_score += 0
            else:
               generalized_kendall_score += 1

      #case 2: i and j are in one list and either j or i is in second ranking
         if i_is_in_list_one and j_is_in_list_one and i_is_in_list_two and not j_is_in_list_two:
            if(rank_list_one_i > rank_list_one_j):
               generalized_kendall_score += 0
            else:
               generalized_kendall_score +=1
         if i_is_in_list_one and j_is_in_list_one and not i_is_in_list_two and j_is_in_list_two:
            if(rank_list_one_j > rank_list_one_i):
               generalized_kendall_score += 0
            else:
               generalized_kendall_score +=1
         if i_is_in_list_two and j_is_in_list_two and i_is_in_list_one and not j_is_in_list_one:
            if(rank_list_two_i > rank_list_two_j):
               generalized_kendall_score += 0
            else:
               generalized_kendall_score +=1
         if i_is_in_list_two and j_is_in_list_two and not i_is_in_list_one and j_is_in_list_one:
            if(rank_list_two_j > rank_list_two_i):
               generalized_kendall_score += 0
            else:
               generalized_kendall_score +=1

      #case 3: i (but not j) appears in one ranking and j (but not i) appears in second ranking)
         if (i_is_in_list_one and not i_is_in_list_two and not j_is_in_list_one and j_is_in_list_two) or (i_is_in_list_two and not i_is_in_list_one and not j_is_in_list_two and j_is_in_list_one):
            generalized_kendall_score += 1

      #case 4: i and j are in one ranking, but none is in the other
         if (i_is_in_list_one and j_is_in_list_one and not i_is_in_list_two and not j_is_in_list_two) or (i_is_in_list_two and j_is_in_list_two and not i_is_in_list_one and not j_is_in_list_one):
            generalized_kendall_score += 0.5



 #  tau = (pairs - 2*generalized_kendall_score) / pairs
   tau = (pairs - generalized_kendall_score) / pairs

   alpha = 0.05

   Z = tau /  sqrt((2 * (2* count + 5)) / (9*count*(count-1)))

   p_value = scipy.stats.norm.sf(abs(Z)) * 2

   return tau,Z,p_value



#ranking for wilcoxon score

def wilcoxon(list_one,list_two):

   list_one_rank = list()
   i = len(list_one)
   for elem in enumerate(list_one):
      list_one_rank.append([i,elem[1]])
      i = i-1

   list_two_rank = list()

   for elem in list_one:
      try:
         ind = list_two.index(elem)
         list_two_rank.append([list_one_rank[ind][0],elem])
      except ValueError:
         ind = -1
         list_two_rank.append([1,elem])


   score_list_two = [x[0] for x in list_two_rank]
   score_list_one = [x[0] for x in list_one_rank]


   s,p = wilcoxon(score_list_one,score_list_two)
   return s,p


def calc_rbo(a,b,p,count):
   scores = rbo(a,b,p)
   min = scores['min']
   res = scores['res']
   ext = scores['ext']


   Z = ext /  sqrt((2 * (2* count + 5)) / (9*count*(count-1)))

   p_value = scipy.stats.norm.sf(abs(Z)) * 2



   return ext,p_value


def overlap(a,b):
   common = list(set(a).intersection(b))
   common_no = len(common)
   size = len(a)
   val = float(common_no)/(size)
   return val

#cor,p = weightedtau(score_max,score_mean)
#print("weighted tau")
#print(cor)
count = 2000
#print(experiment,mean,max, abi)
#experiment = read_list("/home/gentoo/src/similarity/test_strategies/ex_vs_ex_experiment/expression.csv",count)
mean = read_list("/home/gentoo/src/similarity/test_strategies/ex_vs_ex_gene_mean/expression.csv",count)
max = read_list("/home/gentoo/src/similarity/test_strategies/ex_vs_ex_gene_max/expression.csv",count)
abi = read_list2("ABI-correlation-Znfx1-69524632.csv",count + 1)

MI_exp = read_list("/home/gentoo/src/thesis/data/MI_Znfx1_69524632_similarity_results_experiment.csv.csv",count)
GC_exp = read_list("/home/gentoo/src/thesis/data/GC_Znfx1_69524632_similarity_results_experiment.csv.csv",count)
#MS_exp = read_list("/home/gentoo/src/thesis/data/MI_Znfx1_69524632_similarity_results_experiment.csv.csv",count)
   
MI_exp_raw_no_mask = read_list("/home/gentoo/src/similarity/metrics_raw/expression_raw_nomask_MI_Znfx1.csv",count)
#MI_exp_raw_with_mask = read_list(""),count
   
pears = read_list3("/home/gentoo/src/similarity/pearson_corr/expressionZnfx1_P56_sagittal_69524632_200um_2dsurqec_mirrored.nii.gz.csv",count)
pears_raw = read_list3("/home/gentoo/src/similarity/pearson_raw/expressionZnfx1_P56_sagittal_69524632_200um_2dsurqec.nii.gz.csv",count)
print("MI_exp")
print(MI_exp)
print("max")
print(max)
print("mean")
print(mean)
print("GC_exp")
print(GC_exp)
print("abi")
print(abi)
print("pears")
print(pears)
print("pears_raw")
print(pears_raw)
print("MI_exp_raw_no_mask")
print(MI_exp_raw_no_mask )


all = [mean,max,abi,MI_exp,GC_exp,pears,pears_raw,MI_exp_raw_no_mask]
names = ["mean","max","abi","MI_exp","GC_exp","pears","pears_raw","MI_exp_raw_no_mask"]


print("Count = " + str(count))
print("-----------------------")
for i in range(0,len(all)):
   for j in range(i+1,len(all)):
       a = all[i]
       b = all[j]

       print("comparing: " + names[i] + " and "  + names[j])
       print("----------------------------------------------")



       print("generallized kendall tau")
       tau,Z,p_value = generalized_kendall_tau(a,b,count)
       print(tau,p_value)
       print("                         ")
       print("                          ")

print("---------------------------------------")
for i in range(0,len(all)):
   for j in range(i+1,len(all)):
       a = all[i]
       b = all[j]

       print("comparing: " + names[i] + " and "  + names[j])
       print("----------------------------------------------")



       print("kendalltau")
       cor,p = kendalltau(a,b)
       print(cor,p)
       print("                         ")
       print("                          ")


print("---------------------------------------")
for i in range(0,len(all)):
   for j in range(i+1,len(all)):
       a = all[i]
       b = all[j]

       print("comparing: " + names[i] + " and "  + names[j])
       print("----------------------------------------------")




       print("rbo")
       ext,p = calc_rbo(a,b,0.9,count)
       print(ext,p)
       print("                         ")
       print("                          ")


print("---------------------------------------")
for i in range(0,len(all)):
   for j in range(i+1,len(all)):
       a = all[i]
       b = all[j]

       print("comparing: " + names[i] + " and "  + names[j])
       print("----------------------------------------------")


       print("% list overlap")
       val = overlap(a,b)
       print(val)
       print("                         ")
       print("                          ")














