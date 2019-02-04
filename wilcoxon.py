from rbo import rbo
import os
import pandas
from scipy.stats import kendalltau, spearmanr,weightedtau, spearmanr,wilcoxon
import scipy
from math import sqrt

a = [1,2,3,4,5]
b = [2,1,5,4,3]
c = [1,2,3,4,5]

cor,p = kendalltau(a,b)
print(cor,p)

cor,p = kendalltau(a,c)

print(cor,p)

a = [1,2,3,4,5,20,20,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
b = [1,2,5,20,20,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]

s,p = wilcoxon(a,b)


print(s,p)


a = [1,2,3,4,5,6,7,8,9,10]
b = [1,2,5,6,3,4,7,8,9,10]

cor,p = weightedtau(a,b)

print(cor,p)

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

def read_list2(path,rows):
   ranked_genes = list()
   i = 0
   with open(path) as f:
      for row in f:
         name = (row.split(",")[1]).split("_")[0]
         ranked_genes.append(name)
         i = i+ 1
         if i >= rows:break
   return ranked_genes



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




count = 15

experiment = read_list("/home/gentoo/src/similarity/test_strategies/ex_vs_ex_experiment/expression.csv",count)
mean = read_list("/home/gentoo/src/similarity/test_strategies/ex_vs_ex_gene_mean/expression.csv",count)
max = read_list("/home/gentoo/src/similarity/test_strategies/ex_vs_ex_gene_max/expression.csv",count)
abi = read_list2("ABI-correlation-Znfx1-69524632.csv",count + 1)





pears = read_list3("/home/gentoo/src/similarity/pearson_corr/expressionZnfx1_P56_sagittal_69524632_200um_2dsurqec_mirrored.nii.gz.csv",count)
print("pears")
print(pears)



print(abi)
print(mean)



g_mean = list()
g_max = list()
i = 1
for elem in enumerate(mean):
   g_mean.append([i,elem[1]])
   i = i+1

for elem in enumerate(max):
   g_max.append([i,elem[1]])
   i = i+1

all = mean + max
all = list(set(all))

generalized_kendall_score = 0


rand_1 = [1,2,3,4,5,6,7,8,10,11,12,13,14,15]
rand_2 = [1,2,3,116,17,18,19,20,21,22,23,23]
#all = rand_1
z=0
for a in range(0,len(all)):

   for b in range((a+1),len(all)):
      z += 1
      i = all[a]
      j = all[b]

      try:
         ind = mean.index(i)
         i_is_in_list_one=True
         rank_list_one_i = ind
      except ValueError:
         i_is_in_list_one = False

      try:
         ind = max.index(i)
         i_is_in_list_two=True
         rank_list_two_i = ind
      except ValueError:
         i_is_in_list_two = False

      try:
         ind = mean.index(j)
         j_is_in_list_one=True
         rank_list_one_j = ind
      except ValueError:
         j_is_in_list_one = False

      try:
         ind = max.index(j)
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


print(z)
print("generalized_kendall_score")

print(generalized_kendall_score)


tau = (z - 2*generalized_kendall_score) / z
print("tau")
print(tau)



alpha = 0.05

Z = tau /  sqrt((2 * (2* count + 5)) / (9*count*(count-1)))

print("z")
print(Z)


p_value = scipy.stats.norm.sf(abs(Z)) * 2

print("p_value")
print(p_value)


mean_rank = list()
i = len(mean)
for elem in enumerate(mean):
   mean_rank.append([i,elem[1]])
   i = i-1
print(mean_rank)


print("kendalltau max mean")
print(kendalltau(max,mean))




max_rank = list()

for elem in mean:
   try:
      ind = max.index(elem)
      max_rank.append([mean_rank[ind][0],elem])
   except ValueError:
      ind = -1
      max_rank.append([1,elem])


score_max = [x[0] for x in max_rank]
score_mean = [x[0] for x in mean_rank]


print(score_max)
print(score_mean)

s,p = wilcoxon(score_max,score_mean,zero_method = "pratt")

spear

print("wilcoxon")
print(p)


cor,p = weightedtau(score_max,score_mean)
print("weighted tau")
print(cor)

#print(experiment,mean,max, abi)


print("mean","max")
re = rbo(mean,max,0.9)
print(re)
cor,p = spearmanr(mean,max)
print(cor,p)





