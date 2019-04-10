from utils import mirror_sagittal, create_experiment_average, output_results, nan_if, get_energy_density
from utils import ants_measure_similarity, create_mask
from collections import defaultdict
import os
import glob
import nibabel
from nilearn._utils.extmath import fast_abs_percentile
import samri.plotting.maps as maps

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
	results: str
		path to results of the similarity analysis
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
						if exclude is not None:
							id = (os.path.basename(img_gene).split("_")[3])
							if (id in exclude) or (int(id) in exclude):continue
						if "sagittal" in img_gene:
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



