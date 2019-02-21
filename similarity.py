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
#from nistats.reporting import compare_niimgs
from scipy.stats import ttest_1samp
from mne.stats import permutation_t_test

def transform(x,y,z,affine):
	M = affine[:4, :4]
	A = np.linalg.inv(M)
	return np.round(A.dot([x,y,z,1]),decimals=0)

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

	#Copy image right/left(?) from the mid, but keep mid unchanged

	left_side = np.copy(img_data[(mid + 1):,:,: ])
	left_side = np.flip(left_side,0)

	right_side = np.copy(img_data[0:mid,:,:])
	right_side = np.flip(right_side,0)
#TODO: checkec for case 3, test for other cases as well
	#replace
	if np.shape(left_side)[0] > np.shape(img_data[0:mid,:,:])[0]:
#		print("case 1")
		#case 1: origin slightly to the left (or right??), need to trim left_side to the size of the right side
		replace_value = np.shape(left_side)[0] - np.shape(img_data[0:mid,:,:])[0]
		img_data[0:mid,:,:][img_data[0:mid,:,:]  == -1] = left_side[(replace_value-1):,:,:][img_data[0:mid,:,:]  == -1]
		#np.savetxt("Slice_case1.txt",img_data[:,(int(origin[1]) -5):(int(origin[1])) -2, (int(origin[2]) -5)])
		img_data[mid:np.shape(right_side[0]),:,:][img_data[mid:,:,:]  == -1] = right_side[:,:,:][img_data[mid:,:,:]  == -1]

	elif np.shape(left_side)[0] < np.shape(img_data[0:mid,:,:])[0]:
#		print("case 2")
		#case 2 : origin slightly to the right (or left??), need to
		replace_value = np.shape(img_data[0:mid,:,:])[0] - np.shape(left_side)[0]
		img_data[replace_value:mid,:,:] = left_side
		#np.savetxt("Slice_case2.txt",img_data[:,(int(origin[1]) -5):(int(origin[1])) -2, (int(origin[2]) -5)])

		img_data[mid:,:,:][img_data[mid:,:,:]  == -1] = right_side[0:np.shape(img_data[mid:,:,:]),:,:][img_data[mid:,:,:]  == -1]
	else:
#		print("case 3")
		#case 3: same size
		#TODO: There's got to be a better way to write this....  -> a_slice notation:: a[10:16] is a reference, not a copy!
		img_data[0:mid,:,:][img_data[0:mid,:,:]  == -1] = left_side[img_data[0:mid,:,:]  == -1]
		#np.savetxt("Slice_case3.txt",img_data[:,(int(origin[1]) -5):(int(origin[1])) -2, (int(origin[2]) -5)])
		img_data[mid+1:,:,:][img_data[mid+1:,:,:]  == -1] = right_side[[img_data[mid+1:,:,:]  == -1]]

	img_average = nibabel.Nifti1Image(img_data,img.affine)
	filename = str.split(os.path.basename(image),'.nii')[0] + '_mirrored.nii.gz'
	path_to_mirrored = os.path.join(os.path.dirname(image),filename)
	nibabel.save(img_average,path_to_mirrored)

	return path_to_mirrored

def create_mask(image,threshold):
	"""
	Creates and saves a binary mask file.

	Parameters
	----------
	image: str
		Path to NIfTI image.
	threshold: int or float
		threshold used for mask creation. All voxels equal or below threshold will be mapped to a value of zero,
		all voxels above threshold will be mapped to a value of one.

	Returns
	---------
	img_out: str
		Path to the mask in NIfTI format.
	"""
	#I think i need 0 to be in my mask. This seems not to be possible using fslmaths, so maybe do directly with numpy? thr sets all to zero below the value and bin uses image>0 to binarise.
	#mask = fsl.Threshold()
	#mask.inputs.thresh = 0
	#mask.inputs.args = '-bin'
	#mask.inputs.in_file = image
	#img_out = str.split(image,'.nii')[0] + '_mask_fsl.nii.gz'
	#mask.inputs.out_file = img_out
	#mask.run()
	mask_img = nibabel.load("/usr/share/mouse-brain-atlases/dsurqec_200micron_mask.nii")
	atlas_mask = mask_img.get_fdata()

	#using numpy instead of fslmaths
	img = nibabel.load(image)
	img_data = img.get_fdata()
	#apparently ambibuous:img_data[img_data >= 0 and atlas_mask == 1] = 1
#	print("mask")
#	print(np.min(img_data))
	#TODO: ensure shape mathces
	#TODO: also check that there are no values between -1 and 0 that would now not be masked
	img_data[np.logical_and(img_data > threshold,atlas_mask == 1)] = 1
	img_data[np.logical_or(img_data <= threshold,atlas_mask==0)] = 0
	img_out = str.split(image,'.nii')[0] + '_mask.nii.gz'
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
		Unique DataSetIF used by ABI to identfy experiment

	Returns
	---------
	energy: float
		Energy level for a given gene. This value denotes the fraction of voxels that show gene expression,
		modulated by signal intensity.
	density: float
		Density level for a given gene. This value denotes the fraction of voxels that show gene expression.
	"""
	#TODO: put that into the ABI-expression folder for a new version ??
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
	#TODO:what to do about 2 datasets??
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
		arbitrary_cut_off = 0.5
		if (abs(res[0] - res[1]) > 1) and (min(res) < arbitrary_cut_off):
			
			for key_id in en:
				if en[key_id] < arbitrary_cut_off:
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


#TODO:evaluate mean average, diff to and between single experiments, especially now that we include expression = false as well, with small expr, patterns. Also , habe a loo
#maybe exclude exp. with little expression (ABI MAIL)
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
	path_to_exp_average = os.path.join(os.path.dirname(imgs[0]),"..")
	path_to_exp_average = os.path.join(path_to_exp_average,filename)
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
	metric_weight:
		??
	radius_or_number_of_bins: int
		
	sampling_strategy: str {None,Regular,Random}
		
	sampling_percentage: float
		
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
	#TODO:well...
	except:
		print("something happened?")
		res = 0
	return res


def nistats_compare(ref_img,src_img):
	res = compare_niimgs([ref_img],[src_img],plot_hist=False)
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
	ax_1.imshow(plt.imread("0.png"))
	ax_2.imshow(plt.imread("1.png"))
	ax_3.imshow(plt.imread("2.png"))
	plt.savefig(fig_name_prefix + "_all.png",dpi=600)

#TODO: parameterize thresh_percentile and absolute threshold (???)
#TODO: maybe use the same cut coords for all plots, may prove difficult bc not possbile to use nilearns func directly
def plot_results(stat_map,results,hits = 3, template = "/usr/share/mouse-brain-atlases/ambmc2dsurqec_15micron_masked.obj",comparison='gene',vs = "expression",path_to_genes="usr/share/ABI-expression-data",percentile_threshold=94):
	"""
	Plots the input feature map as well as the top three scores

	Parameters:
	stat_map: str
		path to the input feature map NIfTI file
	results: ??
		sorted results of the similarity analyis
	template: str
		brain template .obj file to be used for 3D visualization
	percentile_threshold: int, optional
		percentile to determine the treshold used for displaying the feature maps
	path_to_genes: str
		path to folder of ABI-expression-library
	

	"""
	# TODO: put into stat3D or stat, to avoid loading the data twice threshold = fast_abs_percentile(stat_map)
	dis = dict()
	img_s = nibabel.load(stat_map)
	img_data_s = img_s.get_fdata()
	tresh_s = fast_abs_percentile(img_data_s[img_data_s >0],percentile=percentile_threshold)
	print(tresh_s)
	display_stat = maps.stat3D(stat_map,template="/usr/share/mouse-brain-atlases/dsurqec_200micron_masked.nii",save_as= '_stat.png',threshold=tresh_s,pos_values=True,figure_title=os.path.basename(stat_map))
	dis["main"] = display_stat
	for i in range(0,hits):
		gene_name = results[i][0].split("_")[0] #TODO:this should work in both cases (also for connectivity??)
		full_path_to_gene = results[i][1][1]  #TODO what ??????
		print("now plotting: ")
		print(full_path_to_gene)
		img = nibabel.load(full_path_to_gene)
		img_data = img.get_fdata()
		tresh = fast_abs_percentile(img_data[img_data > 0],percentile=98)
		display = maps.stat3D(full_path_to_gene,template="/usr/share/mouse-brain-atlases/dsurqec_200micron_masked.nii",save_as=str(i) + '.png',threshold=tresh,pos_values=True,figure_title=gene_name)
		dis[str(i)] = display
	_plot(dis,stat_map,vs)
	print(tresh_s)
	print(tresh)
	#TODO:sep.function?



	return


def calculate_significance_save(results):
	sorted_results = sorted(results.items(),key=lambda x: x[1][0])
	all_scores = list()
	for score in results.values():
		all_scores.append(score[0])
	all_scores = np.asarray(all_scores)

	#get significance scores for results
	for id in results.keys():
		t,prob = ttest_1samp(all_scores,results[id][0])
		results[id].append(t)
		results[id].append(prob)

		cohens = (np.mean(all_scores) - results[id][0]) / np.std(all_scores)
		results[id].append(cohens)
	
	for i in range(0,len(sorted_results)-1):
		score_1 = sorted_results[i][1][0]
		score_2 = sorted_results[i+1][1][0]
		diff_score = abs(score_1 - score_2)
		sorted_results[i][1].append(diff_score)




	return results,sorted_results




def calculate_significance(results):
	sorted_results = sorted(results.items(),key=lambda x: x[1][0])

	for i in range(0,len(sorted_results)-1):
		score_1 = sorted_results[i][1][0]
		score_2 = sorted_results[i+1][1][0]
		diff_score = abs(score_1 - score_2)
		sorted_results[i][1].append(diff_score)

	return results,sorted_results


def output_results(results,hits = 3,output_name=None):
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

	with open(output_name,'w') as f:
		for i in range(0,len(results)):
			f.write("%s,%s\n"%(results[i][0],results[i][1]))


	return
#TODO: sorted results: same score, sorting?? warning
def measure_similarity_expression(stat_map,path_to_genes="/usr/share/ABI-expression-data",metric = 'MI',radius_or_number_of_bins = 64,comparison = 'gene',strategy='mean',percentile_threshold=94,mirror=True,flip=False,include = None,exclude=None):
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


	"""

	#TODO: if mirrored or mask files are already present, don't make them again
#TODO:
	#TODO: create a mask for the stat map? or userprovided? or both possible? Or use a single mask. Also, if yes, include threshold in mask func
	#trehshold of -1 is good for the ABI-data, probably not the stat map
	mask_map = create_mask(stat_map,-1)
	#results = dict()
	if "sagittal" in stat_map and mirror == True: stat_map = mirror_sagittal(stat_map)
	results = defaultdict(list)
	#loop through all gene folders, either get data form single experiment or get combined data.
	if comparison == 'gene':
		for dir in os.listdir(path_to_genes):
			path = os.path.join(path_to_genes,dir)
			if not os.path.isdir(path):continue
			#print(path)
			#multiple experiment data available per gene
			if len(os.listdir(path)) > 1:
					img = []
					imgs = glob.glob(path + '/*/*_2dsurqec.nii.gz')
					for img_gene in imgs:
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
			#similarity = nistats_compare(stat_map,img_gene)
			print(stat_map,img_gene)
			print(str(similarity))
			results[dir].append(similarity)
			results[dir].append(img_gene)

	elif comparison == 'experiment':
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
				print("sth is in include")
				if "sagittal" in img_gene: img_gene = mirror_sagittal(img_gene)
				mask_gene = create_mask(img_gene,-1)
				experiment_id = os.path.basename(img_gene).split("_")[3]
				id = dir + "_" + experiment_id
				similarity = ants_measure_similarity(stat_map,img_gene,mask_gene = mask_gene,mask_map=mask_map,metric=metric,radius_or_number_of_bins=radius_or_number_of_bins)
				results[id].append(similarity)
				results[id].append(img_gene)
	#TODO: sort, or use sorted dict form beginning, or only keep top scores anyway?
	#print(comparison)
	#view = [ (v,k) for k,v in results.items() ]
	#view.sort(reverse=True)
	#for v,k in view:
	#	print( "%s: %f" % (k,v))

	#TODO: if metric = MSE, sort other way round
	#sort results:

	#results,sort = calculate_significance(results)

	sorted_results = sorted(results.items(),key=lambda x: x[1][0])
	#print(sorted_results)
#	print(sorted_results[4]) #5th highest result
#	print(sorted_results[4][0]) #key = gene or exp_id
#	print(sorted_results[4][1][0]) #similarity score
#	print(sorted_results[4][1][1]) #path
	#results_with_significance = calculate_significance(sorted_results)
	output_results(sorted_results,output_name="expression" + os.path.basename(stat_map), hits = 3)
	#plot_results(stat_map,sorted_results,vs="expression",hits=3)

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

#   , but bdb

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
#	measure_similarity_expression("/home/gentoo/src/abi2dsurqec_geneexpression/save_small_dataset/Mef2c/Mef2c_P56_coronal_79567505_200um/Mef2c_P56_coronal_79567505_200um_2dsurqec.nii.gz",metric = 'MI',radius_or_number_of_bins = 64,comparison = 'gene')
#	measure_similarity_expression("/home/gentoo/ABI_data_full/data/Tlx2/Tlx2_P56_sagittal_81655554_200um/Tlx2_P56_sagittal_81655554_200um_2dsurqec_mirrored.nii.gz",metric = 'MI',path_to_genes="/home/gentoo/ABI_data_full/data",radius_or_number_of_bins = 64,comparison = 'gene')

	#measure_similarity_connectivity(img,metric = 'MI',radius_or_number_of_bins = 64)
	#measure_similarity_expression(img,metric='MI',percentile_threshold=args.percentile_threshold)

#	measure_similarity_expression("/usr/share/ABI-expression-data/Kat6a/Kat6a_P56_sagittal_71764326_200um/Kat6a_P56_sagittal_71764326_200um_2dsurqec_mirrored.nii.gz",metric = 'GC',radius_or_number_of_bins = 64,comparison = 'experiment')
	id = 79677145
	include = [10,12,79677145,991,2313,243249]
	if id not in include:
		print("not in include")
	else:
		print("is in include")
	measure_similarity_expression(img,comparison = "experiment",exclude = [71489813],include = [79677145,71489813])


#	measure_similarity_expression("/home/gentoo/src/abi2dsurqec_geneexpression/ABI_geneexpression_data/Mef2c/Mef2c_P56_coronal_79567505_200um/Mef2c_P56_coronal_79567505_200um_2dsurqec.nii.gz",metric = 'CC',radius_or_number_of_bins = 4)

#	measure_similarity_expression("/home/gentoo/src/abi2dsurqec_geneexpression/ABI_geneexpression_data/Mef2c/Mef2c_P56_coronal_79567505_200um/Mef2c_P56_coronal_79567505_200um_2dsurqec.nii.gz",metric = 'Mattes',radius_or_number_of_bins = 32)

#	measure_similarity_expression("/home/gentoo/src/abi2dsurqec_geneexpression/ABI_geneexpression_data/Mef2c/Mef2c_P56_coronal_79567505_200um/Mef2c_P56_coronal_79567505_200um_2dsurqec.nii.gz",metric = 'MeanSquares',radius_or_number_of_bins = 64)

#	measure_similarity_expression("/home/gentoo/src/abi2dsurqec_geneexpression/ABI_geneexpression_data/Mef2c/Mef2c_P56_coronal_79567505_200um/Mef2c_P56_coronal_79567505_200um_2dsurqec.nii.gz")
#	create_mask(img)

if __name__ == "__main__":
				main()

