from similarity import plot_results
import os 

img = "/usr/share/ABI-expression-data/Cr2/Cr2_P56_sagittal_74427072_200um/Cr2_P56_sagittal_74427072_200um_2dsurqec.nii.gz"
n = "out/results_expression_Cr2_P56_sagittal_74427072_200um_2dsurqec_mirrored.nii.gz.csv"
n_clus = "test/clustering_results/cluster_Plod1_77792637_3_similarity_results_experiment"
n_sub = "out/results_expression_Cr2_P56_sagittal_74427072_200um_2dsurqec_mirrored.nii.gz.csv_subtraction_results_6"
n_small = "out/results_expression_Cr2_P56_sagittal_74427072_200um_2dsurqec_mirrored.nii.gz.csv_subtraction_results_1"


for r in [n,n_clus,n_sub,n_small]:
   plot_results(img,r)
   
