import numpy as np
import nibabel
import os
import glob
from nipype.interfaces import ants as ants
import nipype.interfaces.fsl.maths as fsl
import csv
import argparse
import nilearn
import matplotlib.pyplot as plt
import samri.plotting.maps as maps
from collections import defaultdict
from nilearn._utils.extmath import fast_abs_percentile
import pandas as pd
from scipy.stats import ttest_1samp
from mne.stats import permutation_t_test
from nipype.interfaces.base import CommandLine
import time
from sklearn.cluster import AgglomerativeClustering
import sys

def transform(x,y,z,affine):
	"""
	Returns affine transformed coordinates (x,y,z) -> (i,j,k)

	Parameters
	----------
	x,y,z: int
		Integer coordinates.
	affine: array
		4x4 matrix specifying image affine.
	"""
	M = affine[:4, :4]
	A = np.linalg.inv(M)
	return np.round(A.dot([x,y,z,1]),decimals=0)

def create_base_path(image,mode="expression"):
	"""
	Creates base path structure for temporary files like mirrored data files, experiment averages and mask files.
	"""
	#/usr/share/ABI-expression-data/Mef2c/Mef2c_P56_sagittal_669_200um/Mef2c_P56_sagittal_669_200um_2dsurqec.nii.gz
	#/var/tmp/similarity/ABI-expression-data/...
	if not os.path.exists("/var/tmp/similarity"): os.makedirs("/var/tmp/similarity")
	if mode == "expression" and not os.path.exists("/var/tmp/similarity/ABI-expression-data"):
		os.makedirs("/var/tmp/similarity/ABI-expression-data")
	if mode == "connectivity" and not os.path.exists("/var/tmp/similarity/ABI-connectivity-data"):
		os.makedirs("/var/tmp/similarity/ABI-connectivity-data")
	base = "/var/tmp/similarity"
	path = os.path.dirname(image)  #/usr/share/ABI-expression-data/Mef2c/Mef2c_P56_sagittal_669_200um/
	path = os.path.normpath(path)
	folders = path.split(os.sep)
	exp = folders.pop()
	gene = folders.pop()
	mode = folders.pop()

	if not os.path.exists(os.path.join(base,mode,gene)): os.makedirs(os.path.join(base,mode,gene))
	return os.path.join(base,mode,gene)


def mirror_sagittal(image):
	"""
	Sagittal datasets form Allen mouse brain are only collected for one hemisphere.
	Function to mirror feature map at midline (midline determined at the origin by affine)
	and saving image as NIfTI-file.

	Parameters
	----------
	image: str
		Path to NIfTI file.

	Returns
	---------
	path_to_mirrored: str
		Path to the mirrored NIfTI file.
	"""

	img = nibabel.load(image)
	img_data = img.get_fdata()

	#find coordinates of origin (Bregma?) in data matrix to determine the left-right midline
	origin = transform(0,0,0,img.affine)
	mid = int(origin[0])

	#TODO:midline point at 31.34, how to mirror properly

	left_side = np.copy(img_data[(mid + 1):,:,: ])
	left_side = np.flip(left_side,0)
	right_side = np.copy(img_data[0:mid,:,:])
	right_side = np.flip(right_side,0)

	#TODO: checkec for case 3, test for other cases as well
	#replace
	if np.shape(left_side)[0] > np.shape(img_data[0:mid,:,:])[0]:
		#case 1: origin slightly to the left (or right??), need to trim left_side to the size of the right side
		replace_value = np.shape(left_side)[0] - np.shape(img_data[0:mid,:,:])[0]
		img_data[0:mid,:,:][img_data[0:mid,:,:]  == -1] = left_side[(replace_value-1):,:,:][img_data[0:mid,:,:]  == -1]
		img_data[mid:np.shape(right_side[0]),:,:][img_data[mid:,:,:]  == -1] = right_side[:,:,:][img_data[mid:,:,:]  == -1]

	elif np.shape(left_side)[0] < np.shape(img_data[0:mid,:,:])[0]:
		#case 2 : origin slightly to the right (or left??), need to
		replace_value = np.shape(img_data[0:mid,:,:])[0] - np.shape(left_side)[0]
		img_data[replace_value:mid,:,:] = left_side
		img_data[mid:,:,:][img_data[mid:,:,:]  == -1] = right_side[0:np.shape(img_data[mid:,:,:]),:,:][img_data[mid:,:,:]  == -1]
	else:
		#case 3: same size
		#TODO: There's got to be a better way to write this....  -> a_slice notation:: a[10:16] is a reference, not a copy!
		img_data[0:mid,:,:][img_data[0:mid,:,:]  == -1] = left_side[img_data[0:mid,:,:]  == -1]
		img_data[mid+1:,:,:][img_data[mid+1:,:,:]  == -1] = right_side[[img_data[mid+1:,:,:]  == -1]]

	img_average = nibabel.Nifti1Image(img_data,img.affine)
	filename = str.split(os.path.basename(image),'.nii')[0] + '_mirrored.nii.gz'
	base_path = create_base_path(image)
	path_to_mirrored = os.path.join(base_path,filename)
	nibabel.save(img_average,path_to_mirrored)

	return path_to_mirrored

def create_mask(image,threshold,mask = "/usr/share/mouse-brain-atlases/dsurqec_200micron_mask.nii"):
	"""
	Creates and saves a binary mask file. Image will be masked according to threshold and the file dsurqec_200micron_mask.nii serves as boundary.

	Parameters
	----------
	image: str
		Path to NIfTI image.
	threshold: int or float or None
		threshold used for mask creation. All voxels equal or below threshold will be mapped to a value of zero,
		all voxels above threshold will be mapped to a value of one. If None, dsurqec_200micron_mask.nii file will be used as mask.

	Returns
	---------
	img_out: str
		Path to the mask in NIfTI format.
	"""

	if threshold is None: return mask

	mask_img = nibabel.load(mask)
	atlas_mask = mask_img.get_fdata()

	img = nibabel.load(image)
	img_data = img.get_fdata()
	if np.shape(img_data) != np.shape(atlas_mask):
		raise ValueError("Template mask {} and image file {} are not of the same shape".format(mask,image))
	img_data[np.logical_and(img_data > threshold,atlas_mask == 1)] = 1
	img_data[np.logical_or(img_data <= threshold,atlas_mask==0)] = 0
	base_path = create_base_path(image)
	img_out = str.split(os.path.basename(image),'.nii')[0] + '_mask.nii.gz'
	img_out = os.path.join(base_path,img_out)
	img_mask = nibabel.Nifti1Image(img_data,img.affine)
	nibabel.save(img_mask,img_out)

	return img_out

def nan_if(arr,value):
	""" replace input value with NaN in given array"""
	return np.where(arr == value, np.nan,arr)


def get_energy_density(id):
	"""
	Reads in the energy, density information for a given dataset ID

	Parameters
	----------
	id: int
		Unique DataSetID used by ABI to identify experiment

	Returns
	---------
	energy: float
		Energy level for a given gene. This value denotes the fraction of voxels that show gene expression,
		modulated by signal intensity.
	density: float
		Density level for a given gene. This value denotes the fraction of voxels that show gene expression.
	"""
	path = "/usr/share/ABI-expression-data/density_energy.csv"
	tb = pd.read_csv(path,delimiter = ',')
	col = tb[tb.id == id]
	energy = col.energy.iloc[0]
	density = col.density.iloc[0]

	return energy,density


def check_expression_level_dataset(imgs):
	"""
	For creating an average gene expression map for a gene, this function filters out datasets that show significantly
	less expression than others, and the average is created only for the remaining experiments.

	Parameters
	----------
	imgs: list of str
		paths to all NIfTI files of a given gene.

	Returns
	--------
	surviving_imgs: list of str
		paths to all NIfTI files to be used to create the average expression map.
	"""
	en = dict()
	dens = dict()
	surviving_imgs = list()
	kicked_out = list()
	for img in imgs:
		img_base = os.path.basename(img)
		id = int(img_base.split("_")[3])
		energy,density = get_energy_density(id)
		en[id] = energy
		dens[id] = density
	res = list(en.values())
	if len(imgs) > 2:
		cut_off = np.mean(res) - np.std(res)
		for key_id in en:
			if en[key_id] >= cut_off:
				index = [i for i, s in enumerate(imgs) if "_" + str(key_id) + "_" in s][0]
				sur = imgs[index]
				surviving_imgs.append(sur)
			else:
				index = [i for i, s in enumerate(imgs) if "_" + str(key_id) + "_" in s][0]
				sur = imgs[index]
				kicked_out.append(sur)
	else:
		cut_off = 0.5
		if (abs(res[0] - res[1]) > 1) and (min(res) < cut_off):

			for key_id in en:
				if en[key_id] < cut_off:
					index = [i for i, s in enumerate(imgs) if "_" + str(key_id) + "_" in s][0]
					sur = imgs[index]
					kicked_out.append(sur)

				else:
					index = [i for i, s in enumerate(imgs) if "_" + str(key_id) + "_" in s][0]
					sur = imgs[index]
					surviving_imgs.append(sur)
		else:
			surviving_imgs.append(imgs[0])
			surviving_imgs.append(imgs[1])

	return surviving_imgs

def create_experiment_average(imgs,strategy='max'):
	"""
	In case of several datasets present for one gene, experiment average is calculated, and
	experiment average is saved as NIfTI file.
	Datasets that show significantly less expression are filtered out.

	Parameters
	----------
	imgs: list of str
		paths to NIfTI files to be used to create an average map.
	strategy: str {mean,max}
		strategy to be used to create the average.
		mean: the mean expression level of all files will be used, values of -1 will be ignored
		max: the maximum expression level of all files will be used

	Returns
	--------
	path_to_exp_average: str
		path to the created experiment average NIfTI file
	"""

	imgs = check_expression_level_dataset(imgs)

	#only one surviving dataset
	if len(imgs) == 1 :
		return imgs[0]

	img_data = []
	img = []
	for image in imgs:
		img_2 = nibabel.load(image)
		img_data.append(img_2.get_fdata())
		img.append(img_2)

	if strategy == 'max':
		average_img = np.maximum(img_data[0],img_data[1])
		if len(img_data)>2:
			for i in range(2,(len(img_data)-1)):
				average_img = np.maximum(average_img,img_data[i])

	elif strategy == 'mean':
		for i in range(0,len(img_data)):
			#ignore values of -1 for mean
			img_data[i] = nan_if(img_data[i],-1)

		average_img = np.nanmean([*img_data],axis=0)
		average_img[np.isnan(average_img)] = -1

	filename = str.split(os.path.basename(imgs[0]),"_")[0] + "_experiments_" + strategy + "_average.nii.gz"
	base_path = create_base_path(imgs[0])
	#path_to_exp_average = os.path.join(os.path.dirname(imgs[0]),"..")
	path_to_exp_average = os.path.join(base_path,filename)
	img_average = nibabel.Nifti1Image(average_img,img[0].affine)
	nibabel.save(img_average,path_to_exp_average)
	return path_to_exp_average



def ants_measure_similarity(fixed_image,moving_image,mask_gene = None,mask_map = None,metric = 'MI',metric_weight = 1.0,radius_or_number_of_bins = 64,sampling_strategy='Regular',sampling_percentage=1.0):
	"""
	Nipype interface for using ANTs' MeasureImageSimilarity. Calculates similarity between two images using various metrics

	Parameters
	----------
	fixed_image: str
		path to fixed image
	moving_image: str
		path to moving image
	mask_gene: str, optional
		path to mask to limit voxels considered by metric for moving image
	mask_map: str, optional
		path to mask to limit voxels considered by metric for fixed image
	metric: str, {'MI'.'CC','GC','Mattes',MeanSquares'}
		metric to be used
	metric_weight: int, optional
		metric weight,, see ANTS MeasureImageSimilarity
	radius_or_number_of_bins: int
		radius of number of bins, see ANTS MeasureImageSimilarity 
	sampling_strategy: str {None,Regular,Random}
		sampling strategy, see ANTS MeasureImageSimilarity
	sampling_percentage: float
		sampling percentage, ANTS MeasureImageSimilarity
	"""

	sim = ants.MeasureImageSimilarity()
	sim.inputs.dimension = 3
	sim.inputs.metric = metric
	sim.inputs.fixed_image = fixed_image
	sim.inputs.moving_image = moving_image
	sim.inputs.metric_weight = metric_weight
	sim.inputs.radius_or_number_of_bins = radius_or_number_of_bins
	sim.inputs.sampling_strategy = sampling_strategy
	sim.inputs.sampling_percentage = sampling_percentage
	if not mask_map is None: sim.inputs.fixed_image_mask = mask_map
	if not mask_gene is None: sim.inputs.moving_image_mask = mask_gene
	try:
		sim_res = sim.run()
		res = sim_res.outputs.similarity
	#TODO:Better error handling?
	except:
		print("Not able to calculate similarity value for {}. Similarity is set to 0".format(moving_image))
		res = 0
	return res

def _plot(dis,stat_map,vs):
	"""
	Combines plots from different feature maps and the input feature map into a single plot
	"""
	fig = plt.figure()
	main = fig.add_axes([0,0,0.6,1.0])
	ax_1=fig.add_axes([0.6,0,0.4,0.3])
	ax_2=fig.add_axes([0.6,0.3,0.4,0.3])
	ax_3=fig.add_axes([0.6,0.6,0.4,0.3])

	for ax in [main,ax_1,ax_2,ax_3]:
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.patch.set_alpha(0.1)
		ax.spines["top"].set_visible(False)
		ax.spines["bottom"].set_visible(False)
		ax.spines["right"].set_visible(False)
		ax.spines["left"].set_visible(False)

	fig_name_prefix = os.path.basename(stat_map) + "_vs_" + vs


#TODO:this temp save fig with the same file name is an issue if i let scirpt run parallel
	main.imshow(plt.imread("_stat.png"))
	try:
		ax_1.imshow(plt.imread("0.png"))
		ax_2.imshow(plt.imread("1.png"))
		ax_3.imshow(plt.imread("2.png"))
	except FileNotFoundError:
		print("")
	plt.savefig(fig_name_prefix + "_all.png",dpi=600)

#TODO: parameterize thresh_percentile and absolute threshold (???)

def plot_results(stat_map,results,hits = 3, template = "/usr/share/mouse-brain-atlases/ambmc2dsurqec_15micron_masked.obj",comparison='gene',vs = "expression",path_to_genes="usr/share/ABI-expression-data",percentile_threshold=94):
	"""
	Plots the input feature map as well as the top three scores

	Parameters:
	stat_map: str
		path to the input feature map NIfTI file
	results: list
		sorted results of the similarity analyis
	template: str
		brain template .obj file to be used for 3D visualization
	percentile_threshold: int, optional
		percentile to determine the treshold used for displaying the feature maps
	path_to_genes: str
		path to folder of ABI-expression-library
	

	"""
	dis = dict()
	img_s = nibabel.load(stat_map)
	img_data_s = img_s.get_fdata()
	tresh_s = fast_abs_percentile(img_data_s[img_data_s >0],percentile=percentile_threshold)
	display_stat = maps.stat3D(stat_map,template="/usr/share/mouse-brain-atlases/dsurqec_200micron_masked.nii",save_as= '_stat.png',threshold=tresh_s,positive_only=True,figure_title=os.path.basename(stat_map))
	dis["main"] = display_stat
	if hits > len(results): hits = len(results) 
	for i in range(0,hits):
		#gene_name = results[i][0]
		gene_name = results[i][0].split("_")[0] #TODO:this should work in both cases (also for connectivity??)
		full_path_to_gene = results[i][1][1]  #TODO what ??????
		img = nibabel.load(full_path_to_gene)
		img_data = img.get_fdata()
		tresh = fast_abs_percentile(img_data[img_data > 0],percentile=98)
		display = maps.stat3D(full_path_to_gene,template="/usr/share/mouse-brain-atlases/dsurqec_200micron_masked.nii",save_as=str(i) + '.png',threshold=tresh,positive_only=True,figure_title=gene_name)
		dis[str(i)] = display
	_plot(dis,stat_map,vs)
	#TODO:sep.function?

	return


def output_results(results,hits = 3,output_name=None,output_path=None):
	"""
	saves sorted results of the similarity analysis into a .csv file
	and prints top hits results to terminal

	Parameters
	----------

	results: list
		sorted results from the similarity analysis, containing gene name, experiment ID, similarity score and path
	hits: int
		number of top hits to pring
	output_name: str
		file name for output results
	"""
	if hits > len(results) : hits = len(results) 
	print("Top " + str(hits) + " hits: ")
	for i in range(0,hits):
		try:
			print(str(results[i][1][0]) + " " + str(results[i][1][1]))
		except:
			a = 3
	#TODO: some smart name...
	#save to csv
	if output_name is None:
		output_name =  "output_results.csv"
	else:
		output_name = output_name + ".csv"
	
	if output_path: output_name = os.path.join(output_path,output_name)

	with open(output_name,'w') as f:
		for i in range(0,len(results)):
			try:
				name,id = results[i][0].split("_")
			except:
				name = results[i][0]
				id = "avg"
			score = results[i][1][0]
			path = results[i][1][1]
			f.write("{},{},{},{}\n".format(name,id,score,path))

	return output_name


#TODO: sorted results: same score, sorting?? warning
def measure_similarity_expression(stat_map,path_to_genes="/usr/share/ABI-expression-data",metric = 'MI',radius_or_number_of_bins = 64,comparison = 'experiment',strategy='mean',percentile_threshold=94,include = None,exclude=None,mask_threshold_map=None,mask_threshold_gene=-1,mask = "/usr/share/mouse-brain-atlases/dsurqec_200micron_mask.nii",out=None,plot=True,save_results=True):
	"""
	Run ANTs MeasureImageSimilarity for given input map against all gene expression patterns.

	Parameters
	-----------

	stat_map: str
		path to input feature map in NIfTI format.
	path_to_genes: str
		path to folder for gene expression data.
	metric: str 
	radius_or_number_of_bins: int
	comparions: str {'gene','experiment'}
		Value of 'gene' means to compare the input feature map with every gene, creating an average expression map per gene.
		Value of 'experiment' means compare input feature map against every experiment.
	strategy: str {'mean','max'}
		Strategy used for creating the experiment average per gene. Only used with comparison = 'gene'.
	percentile_threshold: int
	include: list of str or float,optional
		List of Gene IDs. If specified, similarity measure will only be run against those IDs given in list.
	exclude: list of str or float,optional
		List of Gene IDs. If specified, those ID will be excluded from similarity analysis.
	out: str, optional
		path for saving results and plots.
	plot: bool, optional
		Specifies if top hits should be plotted. Default is True.
	"""

	#TODO: if mirrored or mask files are already present, don't make them again
	mask_map = create_mask(stat_map,mask_threshold_map,mask=mask)
	if "sagittal" in stat_map : stat_map = mirror_sagittal(stat_map)
	results = defaultdict(list)

	#loop through all gene folders, either get data form single experiment or get combined data.
	if comparison == 'gene':
		for dir in os.listdir(path_to_genes):
			path = os.path.join(path_to_genes,dir)
			if not os.path.isdir(path):continue

			#multiple experiment data available per gene
			if len(os.listdir(path)) > 1:
					img = []
					imgs = glob.glob(path + '/*/*_2dsurqec.nii.gz')
					for img_gene in imgs:
						if include is not None:
							id = (os.path.basename(img_gene).split("_")[3])
							if (id not in include) and (int(id) not in include):continue
						if exlude is not None:
							id = (os.path.basename(img_gene).split("_")[3])
							if (id in exclude) or (int(id) in exclude):continue
						if "sagittal" in img_gene and mirror == True:
							 img.append(mirror_sagittal(img_gene))
						else:
							img.append(img_gene)
					img_gene = create_experiment_average(img,strategy=strategy)
			elif len(os.listdir(path)) == 1:
					img_gene = glob.glob(path + '/*/*_2dsurqec.nii.gz')[0]
					if "sagittal" in img_gene:
							img_gene = mirror_sagittal(img_gene)
			else:
				#TODO: wrong place. Insert above in globglob. Generally bad idea with len(list.dir) == 1 If user creates a folder inside, program will crash... Used criteria after globglob
				print("Folder empty or no registered niftifile found. Or weird folder name :)")
				print("Skipping " + dir)
				continue
			#TODO: catch unexpected errors as to not interrupt program, print genename
			mask_gene = create_mask(img_gene,-1)
			similarity = ants_measure_similarity(stat_map,img_gene,mask_gene = mask_gene,mask_map=mask_map,metric=metric,radius_or_number_of_bins=radius_or_number_of_bins)
			results[dir].append(similarity)
			results[dir].append(img_gene)

	elif comparison == 'experiment':
		i = 1
		for dir in os.listdir(path_to_genes):
			path = os.path.join(path_to_genes,dir)
			if not os.path.isdir(path):continue

			img = []
			imgs = glob.glob(path + '/*/*_2dsurqec.nii.gz')
			for img_gene in imgs:
				if include is not None:
					id = (os.path.basename(img_gene).split("_")[3])
					#id = int(id)
					if (id not in include) and (int(id) not in include):continue
				if exclude is not None:
					id = (os.path.basename(img_gene).split("_")[3])
					if (id in exclude) or (int(id) in exclude):continue
					#if (id not in include) and (int(id) not in include):continue
				if "sagittal" in img_gene: img_gene = mirror_sagittal(img_gene)
				mask_gene = create_mask(img_gene,-1)
				experiment_id = os.path.basename(img_gene).split("_")[3]
				id = dir + "_" + experiment_id
				similarity = ants_measure_similarity(stat_map,img_gene,mask_gene = mask_gene,mask_map=mask_map,metric=metric,radius_or_number_of_bins=radius_or_number_of_bins)
				i = i + 1
				results[id].append(similarity)
				results[id].append(img_gene)
	#TODO: sort, or use sorted dict form beginning, or only keep top scores anyway?

	#TODO: if metric = MSE, sort other way round

	sorted_results = sorted(results.items(),key=lambda x: x[1][0])
	if save_results == True: output_results(sorted_results,output_name="results_expression_{}".format(os.path.basename(stat_map)), hits = 3,output_path=out)
	if plot==True: plot_results(stat_map,results,percentile_threshold=percentile_threshold)

	return results

#TODO:maybe, no?
def normalize_image(img_path):
	#TODO: I dont want -1 for no data to be considered as the minimum, and probably not zero either?
	img = nibabel.load(imgpath)
	img_data = img.get_fdata()
	img_min = np.min(img_data)
	img_max = np.max(img_data)

	img_normalized = np.divide(np.subtract(img,img_min),(img_max - img_min))

	img_out = str.split(img_path,'.nii')[0] + '_norm.nii.gz'
	img_mask = nibabel.Nifti1Image(img_data,img.affine)
	nibabel.save(img_mask,img_out)

	return img_out

#TODO: for connectivity data, it would be really cool if the injection site could be indicated in the plot. Cou can sort of see it with the highest values beeing the inj s


#TODO:check if I ever have two experiment files!!
def measure_similarity_connectivity(stat_map,path_to_exp="/usr/share/ABI-connectivity-data",metric = 'MI',radius_or_number_of_bins = 64,resolution=200,percentile_threshold=94):
	"""
	Run ANTs MeasureImageSimilarity for given input map against all projection map.

	Parameters
	-----------

	stat_map: str
		path to input feature map in NIfTI format.
	path_to_exp: str
		path to folder for projecton data.
	metric: str 
	radius_or_number_of_bins: int
	comparions: str {'gene','experiment'}
		Value of 'gene' means to compare the input feature map with every gene, creating an average expression map per gene.
		Value of 'experiment' means compare input feature map against every experiment.
	strategy: str {'mean','max'}
		Strategy used for creating the experiment average per gene. Only used with comparison = 'gene'.
	percentile_threshold: int
	resolution: int {200,40}
	"""
	
	#TODO: mirror sagittal for connectivity?? If stat_map is a sagittal gene, mirror it (maybe do so before)
	mask_map = create_mask(stat_map,-1)
	results = defaultdict(list)
	for dir in os.listdir(path_to_exp):
		path = os.path.join(path_to_exp,dir)
		img = glob.glob(path + '/*' + str(resolution) + '*_2dsurqec.nii*')[0]
		mask_gene = create_mask(img,-1)
		similarity = ants_measure_similarity(stat_map,img,mask_gene = mask_gene,mask_map =mask_map,metric=metric,radius_or_number_of_bins=radius_or_number_of_bins)
		results[dir].append(similarity)
		results[dir].append(img)  #path for plotting
	sorted_results = sorted(results.items(),key=lambda x: x[1][0])
	output_results(sorted_results,hits = 3,output_name=(os.path.basename(stat_map)+ "_connectivity"))
	plot_results(stat_map,sorted_results,hits=3,vs="connectivity")

def get_image_path_from_id(id,path_to_data="/usr/share/ABI-expression-data/"):
   """Finds image path for a given gene in the data library given the unique identifier SectionDataSetID"""

#	/usr/share/ABI-expression_data-data/Ermap/Ermap_P56_sagittal_68844875_200um/Ermap_P56_sagittal_68844875_200um_2dsurqec_mirrored.nii.gz
   path_s = glob.glob(os.path.join(path_to_data,"*/*_P56_sagittal_{}_200um/*_P56_sagittal_{}_200um_2dsurqec_mirrored.nii.gz".format(id,id)))
   path_c = glob.glob(os.path.join(path_to_data,"*/*_P56_coronal_{}_200um/*_P56_coronal_{}_200um_2dsurqec.nii.gz".format(id,id)))
   if len(path_s) == len(path_c) or len(path_s) > 1 or len(path_c)> 1 or (len(path_s) == 1 and len(path_c) == 1) or (len(path_s) != 1 and len(path_c) != 1):
      raise ValueError("No image or multiple images found with given name, id or path to data (id:{} and path {}.".format(str(id),path_to_data))
   if len(path_s) == 1: return path_s[0]
   if len(path_c) == 1: return path_c[0]


def get_image_path(name,id,path_to_data="/usr/share/ABI-expression-data/"):
   """Finds image path for a given gene in the data library given the unique identifier SectionDataSetID and gene name"""

#	/usr/share/ABI-expression_data-data/Ermap/Ermap_P56_sagittal_68844875_200um/Ermap_P56_sagittal_68844875_200um_2dsurqec_mirrored.nii.gz
   path_s = glob.glob(os.path.join(path_to_data,"{}/{}_P56_sagittal_{}_200um/{}_P56_sagittal_{}_200um_2dsurqec_mirrored.nii.gz".format(name,name,id,name,id)))
   path_c = glob.glob(os.path.join(path_to_data,"{}/{}_P56_coronal_{}_200um/{}_P56_coronal_{}_200um_2dsurqec.nii.gz".format(name,name,id,name,id)))
   if len(path_s) == len(path_c) or len(path_s) > 1 or len(path_c)> 1 or (len(path_s) == 1 and len(path_c) == 1) or (len(path_s) != 1 and len(path_c) != 1):
      raise ValueError("No image or nmultiple images found with given name, id or path to data (name: {}, id:{} and path {}.".format(name,str(id),path_to_data))
   if len(path_s) == 1: return path_s[0]
   if len(path_c) == 1: return path_c[0]

def get_top_hit(path):
   """Returns the top results (ID, gene name, and path) from the results with image similarity comparison. """
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
   """ Reads scores from the results with image similarity comparison. """
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
   """ Reads names from the results with image similarity comparisons. """
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


def write_res(all_result_genes,results,mode,out_prefix = "1"):
   """Writes genes found with subtraction/clustering (currently after every iteration) """
   out = os.path.basename(results).split("similarity_results_experiment")[0]
   out = "{}_{}_results_{}".format(out,mode,out_prefix)
   out_name = os.path.join(os.path.dirname(results),out)
   with open(out_name,'w') as f:
      for gene in all_result_genes:
         f.write("{},{},{}\n".format(gene[0],gene[1],gene[2]))


#if not already present, run similarity all against it
def run_sim_single(base_path,name,id,img_path,iteration,exclude = None,path_to_genes="/usr/share/ABI-expression_data",metric = "GC",
                     mode="cluster",comparison="experiment",radius_or_number_of_bins = 64,strategy="mean"):

   """ Runs similarity.measure_similarty_expression"""
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
   return path_


def all_in_exclude(exclude,input_components):
   """debugging/evaluation only"""
   for id in input_components:
      if str(id) not in exclude: return False
   return True

def add_images(imgs):
   """debugging/evaluation only"""
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
   """ Subtracts two input images (data matrix) """
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
   """Clusters genes according to their similarity score and saves results """
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
   """ 
   Reads results from clustering

   Parameters
   ----------
   path: str
      path to cluster results
   no: int
      Cluster label of rejected cluster
   rejected_genes: list
      IDs (???or names) of previously rejected genes

   Returns
   -------

   cluster: list
      IDs of non-rejected genes
   rejected_genes: list
      IDs of previously rejected genes, with added newly rejected genes from current iteration
   """
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
   Compares genes from the original similarity results to genes in the remaining cluster and rejected genes, returns first hit if any found.

   """

   original_values = read_list(original)
   original_names = read_names(original)
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



#TODO: paramterize clean up (delete mirrored, mask files etc if wanted to save space, or keep for faster calculations next time)
def main():

	parser = argparse.ArgumentParser(description="Similarity",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--stat_map','-s',type=str)
	parser.add_argument('--path_genes','-p',type=str)
	parser.add_argument('--comparison','-c',type=str, default='gene')
	parser.add_argument('--radius_or_number_of_bins','-r',type=int,default = 64)
	parser.add_argument('--metric','-m',type=str,default='MI')
	parser.add_argument('--strategy','-y',type=str, default='max')
	parser.add_argument('--percentile_treshold','-t',type=int,default=94)
	args=parser.parse_args()
	img = "/usr/share/ABI-expression-data/Mef2c/Mef2c_P56_sagittal_79677145_200um/Mef2c_P56_sagittal_79677145_200um_2dsurqec.nii.gz"

	#iimgs = glob.glob("/usr/share/ABI-expression-data/Mef2c/*/*2dsurqec.nii.gz")
	#print(imgs)
	#check_expression_level_dataset(imgs)
#img = "/usr/share/ABI-connectivity-data/Primary_motor_area-584903636/P79_coronal_584903636_200um_projection_density_2dsurqec.nii.gz"
# res = ants_measure_similarity("/home/gentoo/src/abi2dsurqec_geneexpression/ABI_geneexpression_data/Mef2c/Mef2c_P56_sagittal_79677145_200um/Mef2c_P56_sagittal_79677145_200um_2dsurqec.nii.gz","/home/gentoo/src/abi2dsurqec_geneexpression/ABI_geneexpression_data/Mef2c/Mef2c_P56_coronal_79567505_200um//Mef2c_P56_coronal_79567505_200um_2dsurqec.nii.gz")
	#	print(res)
#	measure_similarity_expression("/home/gentoo/ABI_data_full/data/Tlx2/Tlx2_P56_sagittal_81655554_200um/Tlx2_P56_sagittal_81655554_200um_2dsurqec_mirrored.nii.gz",metric = 'MI',path_to_genes="/home/gentoo/ABI_data_full/data",radius_or_number_of_bins = 64,comparison = 'gene')

	#measure_similarity_connectivity(img,metric = 'MI',radius_or_number_of_bins = 64)
	measure_similarity_expression(img,path_to_genes="small_dataset_exp",metric='MI')

#	measure_similarity_expression("/usr/share/ABI-expression-data/Kat6a/Kat6a_P56_sagittal_71764326_200um/Kat6a_P56_sagittal_71764326_200um_2dsurqec_mirrored.nii.gz",metric = 'GC',radius_or_number_of_bins = 64,comparison = 'experiment')
	#measure_similarity_expression(img,comparison = "experiment",exclude = [71489813],include = [79677145,71489813])



#	measure_similarity_expression("/home/gentoo/src/abi2dsurqec_geneexpression/ABI_geneexpression_data/Mef2c/Mef2c_P56_coronal_79567505_200um/Mef2c_P56_coronal_79567505_200um_2dsurqec.nii.gz",metric = 'MeanSquares',radius_or_number_of_bins = 64)

#	measure_similarity_expression("/home/gentoo/src/abi2dsurqec_geneexpression/ABI_geneexpression_data/Mef2c/Mef2c_P56_coronal_79567505_200um/Mef2c_P56_coronal_79567505_200um_2dsurqec.nii.gz")
#	create_mask(img)

if __name__ == "__main__":
				main()

