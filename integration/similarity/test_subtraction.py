from analysis import subtraction_analysis

stat_map = "/usr/share/ABI-expression-data/Neo1/Neo1_P56_sagittal_68843817_200um/Neo1_P56_sagittal_68843817_200um_2dsurqec.nii.gz"
#stat_map = "/home/gentoo/src/similarity/small_dataset_exp/Unc13a/Unc13a_P56_sagittal_70785218_200um/Unc13a_P56_sagittal_70785218_200um_2dsurqec_mirrored.nii.gz"
res = "out/results_expression_Cr2_P56_sagittal_74427072_200um_2dsurqec_mirrored.nii.gz.csv"
subtraction_analysis(stat_map,res,out="test",max_iterations=5,path_to_genes="/home/gentoo/src/similarity/small_dataset_exp")
