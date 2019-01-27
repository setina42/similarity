from rbo import rbo
import os
import pandas


def read_list(path,rows):
   ranked_genes = list()
   i = 0
   with open(path) as f:
      for row in f:
         if i == 0:
            i = 1
            continue
         ranked_genes.append(row.split(",")[0])
         i = i+ 1
         if i >= rows:break
   return ranked_genes


Mef2c_667 = read_list("ABI-correlation-Mef2c-667.csv",200)
Znfx1 = read_list("ABI-correlation-Znfx1-69524632.csv",200)

#top 1 hit of Mef2c queried, should be really close to Mef2c as a way to compare 
Nlk = read_list("ABI-correlation-Nlk-76085742.csv",200)
Mef2c_668 = read_list("ABI-correlation-Mef2c-668.csv",200)

Stx1a = read_list("ABI-correlation-Stx1a-2645.csv",200)

Mef2c_79567505 = read_list("ABI-correlation-Mef2c-79567505.csv",200)


print("Mef2c_667","Mef2c_667")
re = rbo(Mef2c_667,Mef2c_667,0.9)
print(re)

print("Mef2c_667","Nlk")
re = rbo(Mef2c_667,Nlk,0.9)
print(re)

print("Mef2c_667","Mef2c_668")
re = rbo(Mef2c_667,Mef2c_668,0.9)
print(re)

print("Mef2c_667","Mef2c_79567505")
re = rbo(Mef2c_667,Mef2c_79567505,0.9)
print(re)


print("Mef2c_667","Stx1a")
re = rbo(Mef2c_667,Stx1a,0.9)
print(re)




#tp 10 or so of Mef2c
Dusp3 =("ABI-correlation-Dusp3-70795853.csv")

