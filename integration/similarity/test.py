from similarity import measure_similarity_expression
img = "/usr/share/ABI-expression-data/Cr2/Cr2_P56_sagittal_74427072_200um/Cr2_P56_sagittal_74427072_200um_2dsurqec.nii.gz"


#measure_similarity_expression(img,path_to_genes="/home/gentoo/src/similarity/small_dataset_exp")
measure_similarity_expression(img,path_to_genes="/home/gentoo/src/similarity/small_dataset_exp",comparison = "gene",out="out")
