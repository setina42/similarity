import os
import glob
path_to_genes = "/usr/share/ABI-expression-data"
res = "Mef2c"
g = glob.glob(os.path.join(path_to_genes,res,'*','*_2dsurqec.nii.gz'))
print(g)
