from rbo import rbo
import os
import pandas
from scipy.stats import kendalltau, spearmanr,weightedtau, spearmanr,wilcoxon
import scipy
from math import sqrt
import matplotlib.pyplot as plt

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
count = 15
#print(experiment,mean,max, abi)
#experiment = read_list("/home/gentoo/src/similarity/test_strategies/ex_vs_ex_experiment/expression.csv",count)
mean = read_list("/home/gentoo/src/similarity/test_strategies/ex_vs_ex_gene_mean/expression.csv",count)
max = read_list("/home/gentoo/src/similarity/test_strategies/ex_vs_ex_gene_max/expression.csv",count)
abi = read_list2("ABI-correlation-Znfx1-69524632.csv",count + 1)

MI_exp = read_list("/home/gentoo/src/thesis/data/MI_Znfx1_69524632_similarity_results_experiment.csv.csv",count)
GC_exp = read_list("/home/gentoo/src/thesis/data/GC_Znfx1_69524632_similarity_results_experiment.csv.csv",count)
MS_exp = read_list("/home/gentoo/src/thesis/data/MeanSquares_Znfx1_69524632_similarity_results_experiment.csv.csv",count)

MI_exp_raw_no_mask = read_list("/home/gentoo/src/similarity/metrics_raw/expression_raw_nomask_MI_Znfx1.csv",count)
MI_exp_raw_mask = read_list("/home/gentoo/src/similarity/metrics_raw/expression_raw_withmask_MI_Znfx1.csv",count)

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


all = [mean,max,abi,MI_exp,GC_exp,pears,pears_raw,MI_exp_raw_no_mask,MI_exp_raw_mask,MS_exp ]
names = ["mean","max","abi","MI_exp","GC_exp","pears","pears_raw","MI_exp_raw_no_mask","MI_exp_raw_mask ", "MS_exp"]

kendall_scores = list()
generalized_kendall_scores = list()
overlap_scores = list()
rbo_scores = list()


print("Count = " + str(count))
print("-----------------------")
for i in range(0,len(all)):
   for j in range(i+1,len(all)):
       a = all[i]
       b = all[j]
      
       name = names[i] + " vs " + names[j]
       print("comparing: " + names[i] + " and "  + names[j])
       print("----------------------------------------------")



       print("generallized kendall tau")
       tau,Z,p_value = generalized_kendall_tau(a,b,count)
       print(tau,p_value)
       print("                         ")
       print("                          ")
       generalized_kendall_scores.append([name,tau,p_value])
print("---------------------------------------")
for i in range(0,len(all)):
   for j in range(i+1,len(all)):
       a = all[i]
       b = all[j]

       name = names[i] + " vs " + names[j]
       print("comparing: " + names[i] + " and "  + names[j])
       print("----------------------------------------------")



       print("kendalltau")
       cor,p = kendalltau(a,b)
       print(cor,p)
       print("                         ")
       print("                          ")

       kendall_scores.append([name,cor,p])
print("---------------------------------------")
for i in range(0,len(all)):
   for j in range(i+1,len(all)):
       a = all[i]
       b = all[j]

       name = names[i] + " vs " + names[j]
       print("comparing: " + names[i] + " and "  + names[j])
       print("----------------------------------------------")




       print("rbo")
       ext,p = calc_rbo(a,b,0.9,count)
       print(ext,p)
       print("                         ")
       print("                          ")

       rbo_scores.append([name,ext,p])
print("---------------------------------------")
for i in range(0,len(all)):
   for j in range(i+1,len(all)):
       a = all[i]
       b = all[j]

       name = names[i] + " vs " + names[j]
       print("comparing: " + names[i] + " and "  + names[j])
       print("----------------------------------------------")


       print("% list overlap")
       val = overlap(a,b)
       print(val)
       print("                         ")
       print("                          ")
       overlap_scores.append([name,val])

#sort after hightest score
gk = sorted(generalized_kendall_scores,key=lambda x: x[1])
k = sorted(kendall_scores,key=lambda x: x[1])
r = sorted(rbo_scores,key=lambda x: x[1])
o = sorted(overlap_scores,key=lambda x: x[1])

print(gk)



plt.rcParams.update({'font.size': 6})

x_axis = range(1, len(gk) + 1)

y_gk = [item[1] for item in gk]
y_k = [item[1] for item in k]
y_o = [item[1] for item in o]
y_r = [item[1] for item in r]


print(y_gk)

print(len(y_gk))
names_gk = [item[0] for item in gk]
names_k = [item[0] for item in k]
names_o = [item[0] for item in o]
names_r = [item[0] for item in r]

fig1,ax1 = plt.subplots()
fig1.subplots_adjust(bottom=0.3)
ax1.plot(x_axis,y_gk)
ax1.set_title("Generalized Kendall Tau")
plt.vlines(x_axis,0,1,linestyles='dashed',lw=0.5,colors='lightgrey')
plt.xticks(x_axis,names_gk,rotation=90)

plt.savefig("/home/gentoo/Sync/Send/a.png")


fig2,ax2 = plt.subplots()
fig2.subplots_adjust(bottom=0.3)
ax2.plot(x_axis,y_k)
ax2.set_title("Kendall Tau")
plt.vlines(x_axis,0,1,linestyles='dashed',lw=0.5,colors='lightgrey')
plt.xticks(x_axis,names_k,rotation=90)
plt.savefig("/home/gentoo/Sync/Send/b.png")

fig3,ax3 = plt.subplots()
fig3.subplots_adjust(bottom=0.3)
ax3.plot(x_axis,y_r)
ax3.set_title("RBO")
plt.vlines(x_axis,0,1,linestyles='dashed',lw=0.5,colors='lightgrey')
plt.xticks(x_axis,names_r,rotation=90)
plt.savefig("/home/gentoo/Sync/Send/c.png")

fig4,ax4 = plt.subplots()
fig4.subplots_adjust(bottom=0.3)
ax4.plot(x_axis,y_o)
ax4.set_title("Overlap")
plt.vlines(x_axis,0,1,linestyles='dashed',lw=0.5,colors='lightgrey')
plt.xticks(x_axis,names_o,rotation=90)
plt.savefig("/home/gentoo/Sync/Send/d.png")





#sort after name
gk = sorted(generalized_kendall_scores,key=lambda x: x[0])
k = sorted(kendall_scores,key=lambda x: x[0])
r = sorted(rbo_scores,key=lambda x: x[0])
o = sorted(overlap_scores,key=lambda x: x[0])

y_gk = [item[1] for item in gk]
y_k = [item[1] for item in k]
y_o = [item[1] for item in o]
y_r = [item[1] for item in r]


names_gk = [item[0] for item in gk]
names_k = [item[0] for item in k]
names_o = [item[0] for item in o]
names_r = [item[0] for item in r]


fig5,ax5 = plt.subplots()
fig5.subplots_adjust(bottom=0.3)
ax5.plot(x_axis,y_o,label="overlap",lw=0.8)
ax5.plot(x_axis,y_r,label = "rbo",lw=0.8)
ax5.plot(x_axis,y_gk,label = "generalized kendalltau",lw=0.8)
ax5.plot(x_axis,y_k,label="kendall tau",lw=0.8)
plt.vlines(x_axis,0,1,linestyles='dashed',lw=0.5,colors='lightgrey')
plt.xticks(x_axis,names_o,rotation=90)
plt.legend()
plt.savefig("/home/gentoo/Sync/Send/e.png")






