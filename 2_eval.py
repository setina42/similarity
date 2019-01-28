from rbo import rbo
import os
import pandas


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
         if i == 0:
            i = 1
            continue
         name = (row.split(",")[1]).split("_")[0]
         ranked_genes.append(name)
         i = i+ 1
         if i >= rows:break
   return ranked_genes


count = 20

experiment = read_list("/home/gentoo/src/similarity/test_strategies/ex_vs_ex_experiment/expression.csv",count)
mean = read_list("/home/gentoo/src/similarity/test_strategies/ex_vs_ex_gene_mean/expression.csv",count)
max = read_list("/home/gentoo/src/similarity/test_strategies/ex_vs_ex_gene_max/expression.csv",count)
abi = read_list2("ABI-correlation-Znfx1-69524632.csv",count)


print(experiment,mean,max, abi)

print("exp","mean")
re = rbo(experiment,mean,0.9)
print(re)

print("exp","max")
re = rbo(experiment,max,0.9)
print(re)

print("mean","max")
re = rbo(mean,max,0.9)
print(re)


print("ABI","exp")
re = rbo(abi,experiment,0.9)
print(re)
print("ABI","mean")
re = rbo(abi,mean,0.9)
print(re)
print("ABI","max")
re = rbo(abi,max,0.9)
print(re)

