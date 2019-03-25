import numpy as np
import os
import random
from sklearn.cluster import AgglomerativeClustering
from similarity import measure_similarity_expression,output_results,mirror_sagittal
import glob
from sklearn.cluster import AgglomerativeClustering
import nibabel

def get_image_path_from_id(id,data_path):

#	/usr/share/ABI-expression-data/Ermap/Ermap_P56_sagittal_68844875_200um/Ermap_P56_sagittal_68844875_200um_2dsurqec_mirrored.nii.gz
   path_s = glob.glob(os.path.join(data_path,"*/*_P56_sagittal_{}_200um/*_P56_sagittal_{}_200um_2dsurqec_mirrored.nii.gz".format(id,id)))
   path_c = glob.glob(os.path.join(data_path,"*/*_P56_coronal_{}_200um/*_P56_coronal_{}_200um_2dsurqec.nii.gz".format(id,id)))
   if len(path_s) == len(path_c) or len(path_s) > 1 or len(path_c)> 1 or (len(path_s) == 1 and len(path_c) == 1) or (len(path_s) != 1 and len(path_c) != 1):
      raise ValueError("No image or nmultiple images found with given name, id or path to data (name: {}, id:{} and path {}.".format(name,str(id),path_to_data))
   if len(path_s) == 1: return path_s[0]
   if len(path_c) == 1: return path_c[0]


def get_image_path(name,id,path_to_data"/usr/share/ABI-expression-data/"):

#	/usr/share/ABI-expression-data/Ermap/Ermap_P56_sagittal_68844875_200um/Ermap_P56_sagittal_68844875_200um_2dsurqec_mirrored.nii.gz
   path_s = glob.glob(os.path.join(path_to_data,"/{}/{}_P56_sagittal_{}_200um/{}_P56_sagittal_{}_200um_2dsurqec_mirrored.nii.gz".format(name,name,id,name,id)))
   path_c = glob.glob(os.path.join(path_to_data,"/{}/{}_P56_coronal_{}_200um/{}_P56_coronal_{}_200um_2dsurqec.nii.gz".format(name,name,id,name,id)))
   if len(path_s) == len(path_c) or len(path_s) > 1 or len(path_c)> 1 or (len(path_s) == 1 and len(path_c) == 1) or (len(path_s) != 1 and len(path_c) != 1):
      raise ValueError("No image or nmultiple images found with given name, id or path to data (name: {}, id:{} and path {}.".format(name,str(id),path_to_data))
   if len(path_s) == 1: return path_s[0]
   if len(path_c) == 1: return path_c[0]

def get_top_hit(path):
   try:
      with open(path) as f:
         for row in f:
            name,id,score,path = row.split(",")
            return name,id,path
   except FileNotFoundError:
      path = path + ".csv"
       with open(path) as f:
         for row in f:
            name,id,score,path = row.split(",")
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

#if not already present, run similarity all against it
def run_sim_single(base_path,name,id,img_path,iteration,exclude = None,metric = "GC",mode="cluster",comparison="experiment",radius_or_number_of_bins = 64,strategy="mean"):

   name_out = "{}_{}_{}_{}_similarity_results_experiment".format(mode,name,str(id),str(iteration))
   file_name = os.path.join(base_path,name_out)
   if not os.path.exists(file_name):
      results = measure_similarity_expression(img_path,comparison = comparison,metric = metric,exclude = exclude,strategy=strategy,radius_or_number_of_bins=strategy_or_number_of_bins)
      sorted_results = sorted(results.items(),key=lambda x: x[1][0])
      output_path = base_path
      path_ = os.path.join(output_path,name_out)
      output_results(sorted_results,output_name= path_)
   else:
      path_ = file_name

   #return os.path.join(base_path,"{}_{}_similarity_results_experiment.csv".format(name,str(id)))
   return path_

def run_cluster_analysis(results,out=None,max_iterations=None):

   if out is None: out = os.getcwd()
   try:
      os.mkdir(os.path.join(out,"clustering_results"))
      base_path = os.path.join(out,"clustering_results")
   except FileExistsError
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

   reject_all = all_in_exclude(exclude,input_components)
   if reject_all:
      done=True

   res_top_hit_path = run_sim_single(base_path,top_hit[0],top_hit[1],top_hit[2],"0",exclude = exclude)
   cluster,top_label = run_cluster(res_top_hit_path,top_hit[0],top_hit[1])


   #get the first hit that was not removed from cluster that appears in the original results
   name,id,rejected_genes,done = get_res(cluster,results,top_label,rejected_genes,done)

   #continue this cycle with this gene: find all similarity and so on till the cluster only consists of one gene
   path = get_image_path(name,id)
   all_result_genes.append([name,id,path])
   rejected_genes.append(name + "_" + str(id))
   for gene in rejected_genes:
      exclude.append(str(gene.split("_")[1]))

   write_res(all_result_genes,results,out_prefix = "0")

   reject_all = all_in_exclude(exclude,input_components)
   if reject_all == True:
      done=True
      print("All rejected after one iteration")

   i = 0
   while not done:
      i = i + 1
      res_top_hit_path = run_sim_single(base_path,name,id,path,i,exclude=exclude)
      cluster,top_label = run_cluster(res_top_hit_path,name,id)
      name,id,rejected_genes,done = get_res(cluster,results,top_label,rejected_genes,done)
      rejected_genes.append(name + "_" + str(id))
      for gene in rejected_genes:
         exclude.append(str(gene.split("_")[1]))
         #not the prettiest solution
         exclude = list(set(exclude))
      path = get_image_path(name,id)
      all_result_genes.append([name,id,path])

      #Just for debugging reasons, save all_result_genes at every iteration
      write_res(all_result_genes,results,out_prefix = str(i))
      #print((exclude))
      reject_all = all_in_exclude(exclude,input_components)
      if reject_all==True:
         done=True
         print("all genes rejected after {} iteration.".format(str(i+1))) 

      if max_iterations:
         if max_iterations < i:
            print("{} iterations reached without finishing".format(str(max_iterations))
            done=True


def subtraction_analysis(results,out=None,max_iterations=None):

   if out is None: out = os.getcwd()
   path = os.path.join(out,"subtraction_results")
   try:
      os.mkdir(path))
      base_path = path
   except FileExistsError:
      base_path = path



   all_result_genes = list()

   input_components,no_additions = get_ids(results)
   print(input_components)

   input_paths = list()
   all_results_genes = list()
   for id_comp in input_components:
      input_paths.append(get_image_path_from_id(id_comp))

   added_img = add_images(input_paths)

   top_hit_name,top_hit_id,top_hit_path = get_top_hit(results)

   print("top hit")
   print(top_hit_name,top_hit_id,top_hit_path)

   print("hoews into subtraction")
   print(added_img,top_hit_path)
   subtract_top_hit_path,done = subtract(added_img,top_hit_path,"First")



   results_first_sub_path = run_sim_single(base_path,top_hit_name,top_hit_id,subtract_top_hit_path,"1")
   all_result_genes.append([top_hit_name,top_hit_id,top_hit_path])
   write_res(all_result_genes,results,out_prefix = "0")

   i = 1

   done = False
   while not done:
      i = i+1
      print("i")
      print(i)


      top_hit_name,top_hit_id,top_hit_path = get_top_hit(results_first_sub_path)
   
      print("next top hit")
      print(top_hit_name,top_hit_id,top_hit_path)
      all_result_genes.append([top_hit_name,top_hit_id,top_hit_path])
      write_res(all_result_genes,results,out_prefix = str(i))
      print("goes into next subtraction")
      print(added_img,top_hit_path)
      subtract_top_hit_path,done = subtract(subtract_top_hit_path,top_hit_path,str(i))
      if done==True:
         print("end reached after {} iterations".format(str(i)))
         break
      results_first_sub_path = run_sim_single(base_path,top_hit_name,top_hit_id,subtract_top_hit_path,str(i))
      if max_iterations:
         if i > max_iterations:
            print("max iterations reached without finishing")
            done = True



