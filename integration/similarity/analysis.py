import os
from utils import get_top_hit, run_sim_single, run_cluster, get_res, get_image_path, write_res, subtract

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

      write_res(all_result_genes,results,"cluster",out_prefix = str(i))
      if max_iterations:
         if max_iterations < i:
            print("{} iterations reached without finishing".format(str(max_iterations)))
            break

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
   exclude = list()
   top_hit_name,top_hit_id,top_hit_path = get_top_hit(results)
   exclude.append(top_hit_id)
   subtract_top_hit_path,done = subtract(stat_map,top_hit_path,"First",base_path)

   results_first_sub_path = run_sim_single(base_path,top_hit_name,top_hit_id,subtract_top_hit_path,"1",path_to_genes=path_to_genes,metric = metric,comparison=comparison,radius_or_number_of_bins = radius_or_number_of_bins,strategy=strategy,exclude=exclude)
   all_result_genes.append([top_hit_name,top_hit_id,top_hit_path])
   write_res(all_result_genes,results,"subtraction",out_prefix = "0")

   i = 1

   done = False
   while not done:
      i = i+1

      top_hit_name,top_hit_id,top_hit_path = get_top_hit(results_first_sub_path)
      exclude.append(top_hit_id)
      all_result_genes.append([top_hit_name,top_hit_id,top_hit_path])
      write_res(all_result_genes,results,"subtraction",out_prefix = str(i))
      print(top_hit_path)
      subtract_top_hit_path,done = subtract(subtract_top_hit_path,top_hit_path,str(i),base_path)
      if done==True:
         print("End reached after {} iterations".format(str(i)))
         break
      results_first_sub_path = run_sim_single(base_path,top_hit_name,top_hit_id,subtract_top_hit_path,str(i),path_to_genes = path_to_genes,metric = metric,comparison=comparison,radius_or_number_of_bins = radius_or_number_of_bins,strategy=strategy,exclude=exclude)
      if max_iterations:
         if i > max_iterations:
            print("Max iterations reached without finishing")
            done = True


