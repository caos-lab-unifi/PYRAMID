import pandas as pd
import os
import argparse
from argparse import HelpFormatter
import sys
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour
from datetime import datetime
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
import warnings

# Suppress all FutureWarnings specifically
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

class c():
	INPUT=''
	OUT=''
	label=''
	metadata=''
	additional=''
	algorithm=''
	sampling_strategy=''
	threads=''
	seed=''
	neighbors=''

def log(msg):
	
	'''
	Logs a standard progress message with a timestamp.
	'''
	
	now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
	print(f"[{now}][Message] {msg}")

def warn(msg):
	
	'''
	Logs a warning message with a timestamp when a non-critical issue occurs.
	'''
	
	now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
	print(f"[{now}][Warning] {msg}")

def error(msg):
	
	'''
	Logs an error message with a timestamp, intended for use before a script exit.
	'''
	
	now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
	print(f"[{now}][Error] {msg}")

def mergeDFs(df1, df2):
	
	'''
	Merge two pandas dataframes based on their index.
	'''
	
	df_final = df1.merge(df2, left_index=True, right_index=True)
	return df_final

class CustomFormat(HelpFormatter):
	
	'''
	Custom help format for argparse.
	'''
	
	def _format_action_invocation(self, action):
		
		if not action.option_strings:
			
			default = self._get_default_metavar_for_positional(action)
			metavar, = self._metavar_formatter(action, default)(1)
			return metavar
		
		else:
			
			parts = []
			
			if action.nargs == 0:
				
				parts.extend(action.option_strings)
			
			else:
				
				default = self._get_default_metavar_for_optional(action)
				args_string = self._format_args(action, default)
				
				for option_string in action.option_strings:
					
					parts.append(option_string)
				
				return '%s %s' % (', '.join(parts), args_string)
			
			return ', '.join(parts)

	def _get_default_metavar_for_optional(self, action):
		
		return action.dest.upper()

def sampling(Algorithm, samplingStrategy, Xtrain, ytrain):

	'''
	Handles the resampling of the dataset using various imbalanced-learn algorithms.
	Automatically adjusts k_neighbors/n_neighbors based on the minority class size
	to prevent n_neighbors > n_samples_fit errors.
	'''

	if not (isinstance(samplingStrategy, (int, float)) or samplingStrategy in ["auto", "all"]):
		error(f"{samplingStrategy} is not a valid choice for --sampling-strategy")
		sys.exit(1)

	log(f"Performing {Algorithm} with strategy '{samplingStrategy}', seed={seed}, threads={threads}")

	# Compute a safe neighbor count: must be strictly less than minority class size
	min_class_count = min(Counter(ytrain).values())
	safe_k = max(1, min(2, min_class_count - 1))

	if safe_k < 2:
		
		warn(
			f"Minority class has only {min_class_count} sample(s). "
			f"Reducing k_neighbors/n_neighbors to {safe_k}. "
			f"Results may be unreliable — consider collecting more data."
		)

	try:

		if Algorithm == "SMOTE":
			
			sampler = SMOTE(
				k_neighbors=safe_k,
				sampling_strategy=samplingStrategy,
				random_state=seed
			)

		elif Algorithm == "SMOTETomek":
			
			tomek_strategy = samplingStrategy if samplingStrategy in ["auto", "all"] else "auto"
			tomek = TomekLinks(sampling_strategy=tomek_strategy, n_jobs=threads)
			smote = SMOTE(k_neighbors=safe_k, sampling_strategy=samplingStrategy, random_state=seed)
			sampler = SMOTETomek(
				sampling_strategy=samplingStrategy,
				smote=smote,
				tomek=tomek,
				random_state=seed
			)

		elif Algorithm == "SMOTEEN":
			
			enn_strategy = samplingStrategy if samplingStrategy in ["auto", "all"] else "auto"
			edited = EditedNearestNeighbours(
				sampling_strategy=enn_strategy,
				n_jobs=threads,
				n_neighbors=safe_k
			)
			smote = SMOTE(k_neighbors=safe_k, sampling_strategy=samplingStrategy, random_state=seed)
			sampler = SMOTEENN(
				sampling_strategy=samplingStrategy,
				smote=smote,
				enn=edited,
				random_state=seed
			)

		elif Algorithm == "AllKNN":
			
			sampler = AllKNN(
				sampling_strategy=samplingStrategy,
				n_jobs=threads,
				n_neighbors=safe_k
			)

		elif Algorithm == "CNN":
			
			sampler = CondensedNearestNeighbour(
				sampling_strategy=samplingStrategy,
				n_jobs=threads,
				random_state=seed
			)

		elif Algorithm == "RENN":
			
			sampler = RepeatedEditedNearestNeighbours(
				sampling_strategy=samplingStrategy,
				n_jobs=threads,
				n_neighbors=safe_k
			)

		elif Algorithm == "ADASYN":
			
			sampler = ADASYN(
				sampling_strategy=samplingStrategy,
				random_state=seed,
				n_neighbors=safe_k
			)

		else:
			
			error(f"Unknown algorithm '{Algorithm}'. Choose one of: SMOTE, SMOTETomek, SMOTEEN, AllKNN, CNN, RENN, ADASYN")
			sys.exit(1)

		X_resampled, y_resampled = sampler.fit_resample(Xtrain, ytrain)
		log(f"Resampling complete. Class distribution after {Algorithm}: {dict(Counter(y_resampled))}")
		return X_resampled, y_resampled

	except Exception as e:
		error(f"Unable to perform {Algorithm} resampling: {e}")
		sys.exit(1)

def run(parser, args):
	'''
	Main execution logic: loads data, merges covariates, and applies resampling.
	'''
	c.INPUT = args.input
	c.OUT = args.output
	c.label = args.label
	c.metadata = args.metadata
	c.additional = args.additional
	c.algorithm = args.algorithm
	c.threads = args.threads
	c.seed = args.seed
	c.neighbors = args.neighbors

	global threads 
	threads = c.threads
	global seed 
	seed = c.seed

	if args.sampling_strategy == "auto" or args.sampling_strategy == "all":
		
		c.sampling_strategy = str(args.sampling_strategy)
	
	else:	
		
		c.sampling_strategy = float(args.sampling_strategy)
	
	if not os.path.exists(c.OUT):
		
		try:
			
			os.makedirs(c.OUT)
		
		except:
			
			error("Cannot create the output folder")
			sys.exit(1)
	else:
		
		if not os.access(os.path.abspath(c.OUT), os.W_OK):
			
			error("Missing write permissions on the output folder")
			sys.exit(1)
		
		elif os.listdir(os.path.abspath(c.OUT)):
			
			error("The output folder is not empty: specify another output folder or clean the current one")
			sys.exit(1)

	try:
		
		dataframe = pd.read_csv(c.INPUT, sep='\t', index_col='ptid')
		dataframe = dataframe.drop(columns=[col for col in dataframe.columns if "extraction_ID" in col or "diagnostics" in col])
		dataframe = dataframe.select_dtypes(include=['number'])
		dataframe.dropna(inplace=True)
	
	except Exception as e:
		
		error(f"TSV {c.INPUT} does not exist, is not readable or is not a valid TSV: {e}")
		sys.exit(1)

	try:
		
		metadata = pd.read_csv(c.metadata, sep='\t', index_col='ptid')
	
	except Exception as e:
		
		error(f"TSV {c.metadata} does not exist, is not readable or is not a valid TSV: {e}")
		sys.exit(1)

	if c.additional:
		
		try:
			
			additional = pd.read_csv(c.additional, sep="\t", index_col='ptid')
		
		except Exception as e:
			
			error(f"TSV {c.additional} does not exist, is not readable or is not a valid TSV: {e}")
			sys.exit(1)

		log(f"Start Sampling with {seed} seed and {threads} threads and {c.additional} file to add covariates")
		dataframe = mergeDFs(dataframe, additional)
		dataframe = dataframe.select_dtypes(include=['number'])
		dataframe.dropna(inplace=True)

	log(f"Start Sampling with {seed} seed and {threads} threads and sampling strategy {c.sampling_strategy}")

	training = mergeDFs(dataframe, metadata[c.label])
	training.dropna(inplace=True)
	X_train = training.drop(columns=c.label)
	y_train = training[c.label]

	# Plot class distribution
	ax = sns.countplot(x=training[c.label], hue=training[c.label], palette=['#008000','#E50000'])
	
	for p in ax.patches:
		
		ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
	
	plt.savefig(os.path.join(c.OUT, 'barplot.train.pdf'))
	plt.clf()

	if c.sampling_strategy is not None:
		
		try:
			
			X_train_, y_train_ = sampling(c.algorithm, c.sampling_strategy, X_train, y_train)
		
		except Exception as e:
			
			error(f"Unable to perform SLIC: {e}")
			sys.exit(1)

		# Plot class distribution after resampling
		ax = sns.countplot(x=y_train_, hue=y_train_, palette=['#008000','#E50000'])
		
		for p in ax.patches:
			
			ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
		
		plt.savefig(os.path.join(c.OUT, f'barplot.{c.algorithm}.pdf'))
		plt.clf()

		log("Writing Output")
		X_train_final = mergeDFs(X_train_, y_train_)
		X_train_final.to_csv(os.path.join(c.OUT, 'train.sampled.tsv'), sep='\t', index=False)
		
		log("Sampling Done")
		sys.exit(0)
	
	else:
		
		log("Specify one of the valid sampling strategy [CNN,RENN,AllKNN,SMOTE,SMOTETomek,SMOTEEN,ADASYN]")
		sys.exit(0)