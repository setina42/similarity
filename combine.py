import numpy as np
import os
import random
from sklearn.cluster import AgglomerativeClustering
from similarity import measure_similarity_expression,output_results,mirror_sagittal
import glob
from sklearn.cluster import AgglomerativeClustering
import nibabel
import sys

def get_image_path_from_id(id,path_to_data="/usr/share/ABI-expression-data/"):

#	/usr/share/ABI-expression_data-data/Ermap/Ermap_P56_sagittal_68844875_200um/Ermap_P56_sagittal_68844875_200um_2dsurqec_mirrored.nii.gz
   path_s = glob.glob(os.path.join(path_to_data,"*/*_P56_sagittal_{}_200um/*_P56_sagittal_{}_200um_2dsurqec_mirrored.nii.gz".format(id,id)))
   path_c = glob.glob(os.path.join(path_to_data,"*/*_P56_coronal_{}_200um/*_P56_coronal_{}_200um_2dsurqec.nii.gz".format(id,id)))
   if len(path_s) == len(path_c) or len(path_s) > 1 or len(path_c)> 1 or (len(path_s) == 1 and len(path_c) == 1) or (len(path_s) != 1 and len(path_c) != 1):
      raise ValueError("No image or multiple images found with given name, id or path to data (id:{} and path {}.".format(str(id),path_to_data))
   if len(path_s) == 1: return path_s[0]
   if len(path_c) == 1: return path_c[0]


def get_image_path(name,id,path_to_data="/usr/share/ABI-expression-data/"):

#	/usr/share/ABI-expression_data-data/Ermap/Ermap_P56_sagittal_68844875_200um/Ermap_P56_sagittal_68844875_200um_2dsurqec_mirrored.nii.gz
   path_s = glob.glob(os.path.join(path_to_data,"{}/{}_P56_sagittal_{}_200um/{}_P56_sagittal_{}_200um_2dsurqec_mirrored.nii.gz".format(name,name,id,name,id)))
   path_c = glob.glob(os.path.join(path_to_data,"{}/{}_P56_coronal_{}_200um/{}_P56_coronal_{}_200um_2dsurqec.nii.gz".format(name,name,id,name,id)))
   if len(path_s) == len(path_c) or len(path_s) > 1 or len(path_c)> 1 or (len(path_s) == 1 and len(path_c) == 1) or (len(path_s) != 1 and len(path_c) != 1):
      raise ValueError("No image or nmultiple images found with given name, id or path to data (name: {}, id:{} and path {}.".format(name,str(id),path_to_data))
   if len(path_s) == 1: return path_s[0]
   if len(path_c) == 1: return path_c[0]

def get_top_hit(path):
   try:
      with open(path) as f:
         for row in f:
            name,id,score,path = row.split(",")
            #TODO: Nope, nope and nope!!!!!
            path = path.split("\n")[0]
            return name,id,path
   except FileNotFoundError:
      path = path + ".csv"
      with open(path) as f:
         for row in f:
            name,id,score,path = row.split(",")
            path = path.split("\n")[0]
            return name,id,path

def get_ids(exp):
   """only for evaluation/debugging"""
   exp = os.path.basename(exp)
   no_additions = exp.split("_")[0]
   gene_ids = list()
   for part in exp.split("_"):
      if len(part) < 3:continue
      if "added" in part or "max" in part or "cap" in part: continue
      if "similarity" in part: break
      if "-" in part:
          gene_ids.append(part.split("-")[1])
      else:
          gene_ids.append(part)
   return gene_ids, no_additions


# cluster
def read_list(path):
   scores = list()
   try:
      with open(path) as f:
         for row in f:
            score = row.split(',')[2]
            scores.append(score)
      return scores
   except FileNotFoundError:
      path = path + ".csv"
      with open(path) as f:
         for row in f:
            score = row.split(',')[2]
            scores.append(score)
      return scores

def read_names(path):
   ranked_genes = list()
   try:
      with open(path) as f:
         for row in f:
            name,id,score,path = row.split(",")
            name = name +"_" + id
            ranked_genes.append(name)
   except FileNotFoundError:
      path = path + ".csv"
      with open(path) as f:
         for row in f:
            name,id,score,path = row.split(",")
            name = name +"_" + id
            ranked_genes.append(name)
   return ranked_genes

#subtraction
def sub_read_list(path):
   ranked_genes = list()
   try:
      with open(path) as f:
         for row in f:
            name = row.split(",")[0]
            ranked_genes.append(name)
      return ranked_genes
   except FileNotFoundError:
      path = path + ".csv"
      with open(path) as f:
         for row in f:
            name = row.split(",")[0]
            ranked_genes.append(name)
      return ranked_genes

def sub_read_names(path):
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


def write_res(all_result_genes,results,mode,out_prefix = "1"):
   out = os.path.basename(results).split("similarity_results_experiment")[0]
   out = "{}_{}_results_{}".format(out,mode,out_prefix)
   out_name = os.path.join(os.path.dirname(results),out)
   with open(out_name,'w') as f:
      for gene in all_result_genes:
         f.write("{},{},{}\n".format(gene[0],gene[1],gene[2]))


#if not already present, run similarity all against it
def run_sim_single(base_path,name,id,img_path,iteration,exclude = None,path_to_genes="/usr/share/ABI-expression_data",metric = "GC",mode="cluster",comparison="experiment",radius_or_number_of_bins = 64,strategy="mean"):

   name_out = "{}_{}_{}_{}_similarity_results_experiment".format(mode,name,str(id),str(iteration))
   file_name = os.path.join(base_path,name_out)
   if not os.path.exists(file_name):
      results = measure_similarity_expression(img_path,comparison = comparison,plot=False,path_to_genes=path_to_genes,metric = metric,exclude = exclude,strategy=strategy,radius_or_number_of_bins=radius_or_number_of_bins,save_results=False)
      sorted_results = sorted(results.items(),key=lambda x: x[1][0])
      output_path = base_path
      path_ = os.path.join(output_path,name_out)
      output_results(sorted_results,output_name= path_)
   else:
      path_ = file_name
   #return os.path.join(base_path,"{}_{}_similarity_results_experiment.csv".format(name,str(id)))
   return path_


def all_in_exclude(exclude,input_components):
   """debugging/evaluation only"""
   for id in input_components:
      if str(id) not in exclude: return False
   return True

def add_images(imgs):
   """debugging/evaluationonly"""
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
      added_names = added_names + "+" + name
   img_added = nibabel.Nifti1Image(added_data,image1.affine)
   filename = added_names + ".nii.gz"
   path = os.path.join("/home/gentoo/src/similarity/cluster/final_test_set_add/subtraction/2_added_cap_max_half_68445185_385779",filename)
   nibabel.save(img_added,path)
   return path

def subtract(image1,image2,name,base_path):
   done = False
   if "sagittal" in image2:image2 = mirror_sagittal(image2)
   img1 = nibabel.load(image1)
   img1_data = img1.get_fdata()
   img2 = nibabel.load(image2)
   img2_data = img2.get_fdata()
   sub = np.copy(img1_data)
   sub[img2_data >=0] = np.subtract(img1_data[img2_data >=0],img2_data[img2_data >= 0])
   print(np.max(sub))
   if np.max(sub) <= 0:done = True
   img_sub = nibabel.Nifti1Image(sub,img1.affine)
   filename = name + "_sub.nii.gz"
   path = os.path.join(base_path,filename)
   nibabel.save(img_sub,path)
   return path,done


def run_cluster(path,name,id,iteration="0"):
   #from the single against all experiment
   exp = path
   out = "{}_{}_cluster.csv".format(name,str(id))
   out_path = os.path.join(os.path.dirname(exp),out)
   gene = read_names(exp)
   res = read_list(exp)
   if len(res) < 2: 
      print("Finished after {} iterations.".format(iteration))
      sys.exit(0)

   resi = np.asarray(res)
   resi[resi.astype(np.float) >= 1e300] = 0
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
   """ 
   Compares ???

   """ 
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


def run_cluster_analysis(results,out=None,max_iterations=None,path_to_genes="/usr/share/ABI-expression_data",metric = "GC",comparison="experiment",radius_or_number_of_bins = 64,strategy="mean"):

   if out is None: out = os.getcwd()
   try:
      os.mkdir(os.path.join(out,"clustering_results"))
      base_path = os.path.join(out,"clustering_results")
   except FileExistsError:
      base_path = os.path.join(out,"clustering_results")


   rejected_genes = list()
   done = False
   exclude = list()
   all_result_genes = list()

   #get top hit, name, id, path
   top_hit = get_top_hit(results)
   all_result_genes.append(top_hit)
   rejected_genes.append(top_hit[0] + "_" + str(top_hit[1]))
   exclude.append(str(top_hit[1]))

   #evalutation/Debugging
   #reject_all = all_in_exclude(exclude,input_components)
   #if reject_all:
   #   done=True

   res_top_hit_path = run_sim_single(base_path,top_hit[0],top_hit[1],top_hit[2],"0",exclude = exclude,path_to_genes=path_to_genes,metric = metric,comparison=comparison,radius_or_number_of_bins = radius_or_number_of_bins,strategy=strategy)
   cluster,top_label = run_cluster(res_top_hit_path,top_hit[0],top_hit[1])


   #get the first hit that was not removed from cluster that appears in the original results
   name,id,rejected_genes,done = get_res(cluster,results,top_label,rejected_genes,done)

   #continue this cycle with this gene: find all similarity and so on till the cluster only consists of one gene
   path = get_image_path(name,id)
   all_result_genes.append([name,id,path])
   rejected_genes.append(name + "_" + str(id))
   for gene in rejected_genes:
      exclude.append(str(gene.split("_")[1]))

   write_res(all_result_genes,results,"cluster",out_prefix = "0")

   #Evaluation/Deugging
   #reject_all = all_in_exclude(exclude,input_components)
   #if reject_all == True:
   #   done=True
   #   print("All rejected after one iteration")

   i = 0
   while not done:
      i = i + 1
      res_top_hit_path = run_sim_single(base_path,name,id,path,i,exclude=exclude,path_to_genes=path_to_genes,metric = metric,comparison=comparison,radius_or_number_of_bins = radius_or_number_of_bins,strategy=strategy)
      cluster,top_label = run_cluster(res_top_hit_path,name,id,iteration=str(i))
      name,id,rejected_genes,done = get_res(cluster,results,top_label,rejected_genes,done)
      rejected_genes.append(name + "_" + str(id))
      for gene in rejected_genes:
         exclude.append(str(gene.split("_")[1]))
         #not the prettiest solution
         exclude = list(set(exclude))
      path = get_image_path(name,id)
      all_result_genes.append([name,id,path])

      #Just for debugging reasons, save all_result_genes at every iteration
      write_res(all_result_genes,results,"cluster",out_prefix = str(i))
      #print((exclude))

      #Evaluation/Debugging
      #reject_all = all_in_exclude(exclude,input_components)
      #if reject_all==True:
      #   done=True
      #   print("all genes rejected after {} iteration.".format(str(i+1))) 

      if max_iterations:
         if max_iterations < i:
            print("{} iterations reached without finishing".format(str(max_iterations)))
            done=True
      if done: print("Finished after {} iterations".format(str(i)))


def subtraction_analysis(stat_map,results,out=None,max_iterations=None,path_to_genes="/usr/share/ABI-expression_data",metric = "GC",comparison="experiment",radius_or_number_of_bins = 64,strategy="mean"):

   if out is None: out = os.getcwd()
   path = os.path.join(out,"subtraction_results")
   try:
      os.mkdir(path)
      base_path = path
   except FileExistsError:
      base_path = path

   all_result_genes = list()
   
   #Evaluation/Debugging
   #input_paths = list()
   #input_components,no_additions = get_ids(results)
   #print(input_components)

   #for id_comp in input_components:
   #   input_paths.append(get_image_path_from_id(id_comp))

   #added_img = add_images(input_paths)

   #top_hit_name,top_hit_id,top_hit_path = get_top_hit(results)
   #subtract_top_hit_path,done = subtract(added_img,top_hit_path,"First",base_path)


   top_hit_name,top_hit_id,top_hit_path = get_top_hit(results)
   subtract_top_hit_path,done = subtract(stat_map,top_hit_path,"First",base_path)

   results_first_sub_path = run_sim_single(base_path,top_hit_name,top_hit_id,subtract_top_hit_path,"1",path_to_genes=path_to_genes,metric = metric,comparison=comparison,radius_or_number_of_bins = radius_or_number_of_bins,strategy=strategy)
   all_result_genes.append([top_hit_name,top_hit_id,top_hit_path])
   write_res(all_result_genes,results,"subtraction",out_prefix = "0")

   i = 1

   done = False
   while not done:
      i = i+1

      top_hit_name,top_hit_id,top_hit_path = get_top_hit(results_first_sub_path)
      all_result_genes.append([top_hit_name,top_hit_id,top_hit_path])
      write_res(all_result_genes,results,"subtraction",out_prefix = str(i))
      print(top_hit_path)
      subtract_top_hit_path,done = subtract(subtract_top_hit_path,top_hit_path,str(i),base_path)
      if done==True:
         print("end reached after {} iterations".format(str(i)))
         break
      results_first_sub_path = run_sim_single(base_path,top_hit_name,top_hit_id,subtract_top_hit_path,str(i),path_to_genes = path_to_genes,metric = metric,comparison=comparison,radius_or_number_of_bins = radius_or_number_of_bins,strategy=strategy)
      if max_iterations:
         if i > max_iterations:
            print("max iterations reached without finishing")
            done = True



