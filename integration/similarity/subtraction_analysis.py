import os
from utils import get_top_hit, subtract, run_sim_single, write_res

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
         print("End reached after {} iterations".format(str(i)))
         break
      results_first_sub_path = run_sim_single(base_path,top_hit_name,top_hit_id,subtract_top_hit_path,str(i),path_to_genes = path_to_genes,metric = metric,comparison=comparison,radius_or_number_of_bins = radius_or_number_of_bins,strategy=strategy)
      if max_iterations:
         if i > max_iterations:
            print("Max iterations reached without finishing")
            done = True

