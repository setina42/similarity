import random
import numpy as np
import nibabel
import glob
import os
from similarity import measure_similarity_expression


#take 100 random genes from each category

#get image path with acronym and ID
def get_image_path(name,id):
	path = "/usr/share/ABI-expression-data/"
#	/usr/share/ABI-expression-data/Ermap/Ermap_P56_sagittal_68844875_200um/Ermap_P56_sagittal_68844875_200um_2dsurqec_mirrored.nii.gz
	path_s = glob.glob("/usr/share/ABI-expression-data/{}/{}_P56_sagittal_{}_200um/{}_P56_sagittal_{}_200um_2dsurqec_mirrored.nii.gz".format(name,name,id,name,id))
	path_c = glob.glob("/usr/share/ABI-expression-data/{}/{}_P56_coronal_{}_200um/{}_P56_coronal_{}_200um_2dsurqec.nii.gz".format(name,name,id,name,id))
	if len(path_s) == len(path_c) or len(path_s) > 1 or len(path_c)> 1 or (len(path_s) == 1 and len(path_c) == 1) or (len(path_s) != 1 and len(path_c) != 1): 
		raise ValueError("Idiot child!!!!!")
	if len(path_s) == 1: return path_s[0]
	if len(path_c) == 1: return path_c[0]



#randomly take between 2 and 15 with a higher chance for 2??? and rand_genes = list()


def sample_rand_genes(energy,count):

	rand_genes = list()

	rand_genes.append(random.sample([elem for elem in  energy if float(elem[3]) >2 and float(elem[3]) < 4],count))
	rand_genes.append(random.sample([elem for elem in  energy if float(elem[3]) >4 and float(elem[3]) < 6],count))
	rand_genes.append(random.sample([elem for elem in  energy if float(elem[3]) >6 and float(elem[3]) < 8],count))
	rand_genes.append(random.sample([elem for elem in  energy if float(elem[3]) >8 and float(elem[3]) < 10],count))
	rand_genes.append(random.sample([elem for elem in  energy if float(elem[3]) >10 and float(elem[3]) < 12],count))
	rand_genes.append(random.sample([elem for elem in  energy if float(elem[3]) >12 and float(elem[3]) < 14],count))
	rand_genes.append(random.sample([elem for elem in  energy if float(elem[3]) >14 and float(elem[3]) < 16],count))
	rand_genes.append(random.sample([elem for elem in  energy if float(elem[3]) >16 and float(elem[3]) < 18],count))
	rand_genes.append(random.sample([elem for elem in  energy if float(elem[3]) >18 and float(elem[3]) < 20],count))
	rand_genes.append(random.sample([elem for elem in  energy if float(elem[3]) >20 and float(elem[3]) < 22],count))
	rand_genes.append(random.sample([elem for elem in  energy if float(elem[3]) >22 and float(elem[3]) < 26],count))
	rand_genes.append(random.sample([elem for elem in  energy if float(elem[3]) >26 ],count))

	rand_genes_paths = list()
	for genes in rand_genes:
		for gene in genes:
			try:
				p = get_image_path(gene[0],gene[1])
				rand_genes_paths.append(p)
			except ValueError:
				b = 5

	return rand_genes_paths


#Read expression density all and sort
def read_energy(path="/usr/share/ABI-expression-data/density_energy.csv"):
	energy = list()
	i = 0
	with open(path) as f:
		for row in f:
			if "energy" in row: continue
			all = row.split(",")
			all[3] = all[3].split("\n")[0]
			energy.append(all)
	return energy

#['Tuba1a', '68844164', '0.188283', '35.7502']


#add all images in list
def add_images(imgs):
	image1 = nibabel.load(imgs[0])
	added_data = image1.get_fdata()
	name = os.path.basename(imgs[0]).split("_200um")[0]
	imgs.pop(0)
	added_names = name
	for image in imgs:
		img = nibabel.load(image)
		img_data = img.get_fdata()
		added_data = np.add(added_data,img_data)
		name = os.path.basename(image).split("_200um")[0]
		added_names = name + "+" + name
	img_added = nibabel.Nifti1Image(added_data,image1.affine)
	filename = added_names + ".nii.gz"
	path = os.path.join("/home/gentoo/src/similarity/cluster/new_test_dataset",filename)
	nibabel.save(img_added,path)
	return path


#receives a list of images, adds them and generates similarity output
def gen(images,out):
	img = add_images(images)
	results = measure_similarity_expression(img,comparison = "experiment",metric = "GC")
	sorted_results = sorted(results.items(),key=lambda x: x[1][0])
	name_out = out + "_similarity_results_experiment"
	output_path = "/home/gentoo/src/similarity/cluster/new_test_dataset/"
	path_ = os.path.join(output_path,name_out)
	output_results(sorted_results,output_name= path_)


energy = read_energy()
energy_sorted = sorted(energy,key=lambda x: float(x[3]),reverse=True)

counts = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]

a = True
while(a):
	out = ""
	rand_paths = sample_rand_genes(energy_sorted,50)
	no = random.sample(counts,1)[0]
	out = str(no) + "_additions_"
	genes = random.sample(rand_paths,no)
	for gene in genes:
		name = os.path.basename(gene).split("_")[0]
		id = os.path.basename(gene).split("_")[3]
		out = out + name + "-" + str(id) + "_"
	print(out)
	gen(genes,out)


#






#for the addition, run the similarity metric


#Output the result with name of the addition
