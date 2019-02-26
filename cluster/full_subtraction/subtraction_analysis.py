import numpy as np
import os
import random
from similarity import measure_similarity_expression,output_results
import glob
from sklearn.cluster import AgglomerativeClustering


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

def get_image_path_from_id(id):
   path = "/usr/share/ABI-expression-data/"
#	/usr/share/ABI-expression-data/Ermap/Ermap_P56_sagittal_68844875_200um/Ermap_P56_sagittal_68844875_200um_2dsurqec_mirrored.nii.gz
   path_s = glob.glob("/usr/share/ABI-expression-data/*/*_P56_sagittal_{}_200um/*_P56_sagittal_{}_200um_2dsurqec_mirrored.nii.gz".format(id,id))
   path_c = glob.glob("/usr/share/ABI-expression-data/*/*_P56_coronal_{}_200um/*_P56_coronal_{}_200um_2dsurqec.nii.gz".format(id,id))
   if len(path_s) == len(path_c) or len(path_s) > 1 or len(path_c)> 1 or (len(path_s) == 1 and len(path_c) == 1) or (len(path_s) != 1 and len(path_c) != 1): 
      raise ValueError("Idiot child!!!!!")
   if len(path_s) == 1: return path_s[0]
   if len(path_c) == 1: return path_c[0]


def get_ids(exp):
   exp = os.path.basename(exp)
   no_additions = exp.split("_")[0]
   gene_ids = list()
   for part in exp.split("_"):
      if len(part) < 3:continue
      if "additions" in part: continue
      if "similarity" in part: break
      if "-" in part:
          gene_ids.append(part.split("-")[1])
      else:
          gene_ids.append(part)
   return gene_ids, no_additions

#get top hit of an add_result file, take file as an argument

def get_top_hit(path):
   try:
      with open(path) as f:
         for row in f:
            name = row.split("_")[0]
            id = row.split("_")[1].split(",")[0]
            path = row.split("'")[1]
            return name,id,path
   except FileNotFoundError:
       with open(path) as f:
         for row in f:
            name = row.split("_")[0]
            id = row.split("_")[1].split(",")[0]
            path = row.split("'")[1]
            return name,id,path


#if not already present, run similarity all against it
def run_sim_single(name,id,img_path,iteration,exclude = None):
   #print(name)
   #print(id)
   base_path = "/home/gentoo/src/similarity/cluster/full_subtraction/single_gene_results/"
   file_name = os.path.join(base_path,"subtract_{}_{}_{}_similarity_results_experiment.csv".format(name,str(id),str(iteration)))
   if not os.path.exists(file_name):
      results = measure_similarity_expression(img_path,comparison = "experiment",metric = "GC",exclude = exclude)
      sorted_results = sorted(results.items(),key=lambda x: x[1][0])
      name_out = "subtract_{}_{}_{}_similarity_results_experiment".format(name,str(id),str(iteration))
      output_path = base_path
      path_ = os.path.join(output_path,name_out)
      output_results(sorted_results,output_name= path_)
   else:
      path_ = file_name

   #return os.path.join(base_path,"{}_{}_similarity_results_experiment.csv".format(name,str(id)))
   return path_


def read_list(path):
   ranked_genes = list()
   try:
      with open(path) as f:
         for row in f:
            name = (row.split(",")[1]).split("[")[1]
            ranked_genes.append(name)
      return ranked_genes
   except FileNotFoundError:
      path = path + ".csv"
      with open(path) as f:
         for row in f:
            name = (row.split(",")[1]).split("[")[1]
            ranked_genes.append(name)
      return ranked_genes

def read_names(path):
   ranked_genes = list()
   try:
      with open(path) as f:
         for row in f:
            name = (row.split(",")[0])
            ranked_genes.append(name)
   except FileNotFoundError:
      path = path + ".csv"
      with open(path) as f:
         for row in f:
            name = (row.split(",")[0])
            ranked_genes.append(name)
   return ranked_genes




def write_res(all_result_genes,input_additions,out_prefix = "1"):
   out = os.path.basename(input_additions).split("similarity_results_experiment")[0]
   out = out + "subtraction_results" + "_" + out_prefix
   out_name = os.path.join(os.path.dirname(input_additions),out)
   with open(out_name,'w') as f:
      for gene in all_result_genes:
         f.write("{},{},{}\n".format(gene[0],gene[1],gene[2]))


def all_in_exclude(exclude,input_components):
   for id in input_components:
      if str(id) not in exclude: return False
   return True

def add_images(imgs):
	ii = imgs[0]
	if "sagittal" in ii:ii = mirror_sagittal(ii)
	image1 = nibabel.load(ii)
	added_data = image1.get_fdata()
	name = os.path.basename(ii.split("_200um")[0])
	imgs.pop(0)
	added_names = name
	for image in imgs:
		if "sagittal" in image: image = mirror_sagittal(image)
		img = nibabel.load(image)
		img_data = img.get_fdata()
		added_data = np.add(added_data,img_data)
		name = os.path.basename(image).split("_200um")[0]
		added_names = name + "+" + name
	img_added = nibabel.Nifti1Image(added_data,image1.affine)
	filename = added_names + ".nii.gz"
	path = os.path.join("/home/gentoo/src/similarity/cluster/full_subtraction",filename)
	nibabel.save(img_added,path)
	return path

def subtract(image1,image2,name):
   done = False
   img1 = nibabel.load(image1)
   img1_data = img1.get_fdata()
   img2 = nibabel.load(image2)
   img2_data = img2.get_fdata()
   sub = np.copy(img1_data)
   sub[img2_data>=0] = np.subtract(img1_data[img2_data[>=0],img2_data[img2_data >= 0])
   if np.max(sub) <= 0:done = True
   img_sub = nibabel.Nifti1Image(sub,img1.affine)
   filename = name + "_sub.nii.gz"
   path = os.path.join("/home/gentoo/src/similarity/cluster/full_subtraction",filename)
   nibabel.save(img_sub,path)
   return path,done





input_additions = "6_additions_69262234_70919382_79587720_70634395_69529059_74562284__similarity_results_experiment.csv" 
input_components,no_additions = get_ids(input_additions)
input_paths = list()

for id_comp in input_components:
   input_paths.append(get_image_path_from_id(id_comp))

added_img = add_images(input_paths)

top_hit_name,top_hit_id,top_hit_path = get_top_hit(input_additions)

print("top hit")
print(top_hit_name,top_hit_id,top_hit_path)

print("hoews into subtraction")
print(added_img,top_hit_path)
subtract_top_hit_path,done = subtract(added_img,top_hit_path,"First")



results_first_sub_path = run_sim_single(top_hit_name,top_hit_id,subtract_top_hit_path,"1")
all_result_genes.append([top_hit_name,top_hit_id,top_hit_path])
write_res(all_result_genes,input_additions,out_prefix = str(i))

i = 1

done = False
while not done:
   i = i+1
   print("i")
   print(i)


   top_hit_name,top_hit_id,top_hit_path = get_top_hit(input_additions)
   
   print("next top hit")
   print(top_hit_name,top_hit_id,top_hit_path)
   all_result_genes.append([top_hit_name,top_hit_id,top_hit_path])
   write_res(all_result_genes,input_additions,out_prefix = str(i))
   print("goes into next subtraction")
   print(added_img,top_hit_path)
   subtract_top_hit_path,done = subtract(subtract_top_hit_path,top_hit_path,str(i))
   if done=True:
      print("end reached after {} iterations".format(str(i)))
      break
   results_first_sub_path = run_sim_single(top_hit_name,top_hit_id,subtract_top_hit_path,str(i))
   if i > 10: done = True



