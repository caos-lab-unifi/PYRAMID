import argparse
from argparse import HelpFormatter
import sys
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline as pipe
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,confusion_matrix,precision_score,recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import itertools
import contextlib
import re 
import warnings
import json
from typing import Dict, Callable, List
from dataclasses import dataclass
from sklearn.base import BaseEstimator

class c():

	INPUT=''
	OUT=''
	label=''
	json=''
	feature_selection=''
	metadata=''
	search=''
	threshold=''
	metric=''
	additional=''
	threads=''
	verbose=''
	parameters=''
	seed=''
	n_splits=''
	n_repeats=''

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

class CustomFormat(HelpFormatter):

	'''
	Custom help format
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

class ModelParams:
	
	def __init__(self, json_file_path):
		"""
		Initialize ModelParams by loading parameters from a JSON file.
		
		Args:
			json_file_path (str): Path to the JSON file containing model parameters
		"""
		# Load parameters from JSON file
		
		with open(json_file_path, 'r') as f:
			
			params = json.load(f)
		
		# Dynamically set attributes based on JSON content
		for param_name, param_value in params.items():
			
			# Handle numpy arrays if present in the parameters
			if isinstance(param_value, dict):
				
				for key, value in param_value.items():
					
					if isinstance(value, list) and key in ['var_smoothing', 'min_samples_split', 'min_samples_leaf', 'subsample']:
						
						param_value[key] = np.array(value)
			
			# Set the attribute
			setattr(self, param_name, param_value)

	def get_params(self, model_name):
		
		"""Return parameters for a given model"""
		
		return getattr(self, f'parameters_{model_name}', None)

	def load_all_params(self):
		"""
		Load all parameters into a dictionary based on attribute names starting with 'parameters_'
		
		Returns:
			dict: Dictionary containing all model parameters
		"""
		return {
			name.replace('parameters_', ''): getattr(self, name)
			for name in dir(self)
			if name.startswith('parameters_')
		}

def load_model_parameters(json_path):
	
	"""
	Load all model parameters from a JSON file into a dictionary
	
	Args:
		json_path (str): Path to the JSON file containing model parameters
	
	Returns:
		dict: Dictionary where keys are model names and values are their parameters
	"""
	
	params = ModelParams(json_path)	
	return params.load_all_params()

@dataclass
class AlgorithmMapper:
	
	seed: int
	
	def __init__(self, seed: int):
		self.seed = seed
		self.algorithm_dict = self._create_algorithm_dict()
	
	def _create_algorithm_dict(self) -> Dict[str, BaseEstimator]:
		
		"""Creates a dictionary mapping algorithm names to their initialized instances"""
		
		return {
			'DecisionTreeClassifier': DecisionTreeClassifier(random_state=self.seed),
			'GaussianNB': GaussianNB(),
			'GradientBoost': GradientBoostingClassifier(random_state=self.seed),
			'KNeighborsClassifier': KNeighborsClassifier(),
			'LogisticRegression_elasticnet': LogisticRegression(random_state=self.seed),
			'LogisticRegression_l1': LogisticRegression(random_state=self.seed), 
			'LogisticRegression_l2': LogisticRegression(random_state=self.seed),
			'Perceptron_elasticnet': Perceptron(random_state=self.seed),
			'Perceptron_l1': Perceptron(random_state=self.seed),
			'Perceptron_l2': Perceptron(random_state=self.seed),
			'RandomForestClassifier': RandomForestClassifier(random_state=self.seed),
			'SupportVectorMachine': SVC(max_iter=1000, random_state=self.seed, probability=True),
			'SupportVectorMachine_linear': SVC(max_iter=1000, random_state=self.seed, probability=True),
			'SupportVectorMachine_mixed': SVC(max_iter=1000, random_state=self.seed, probability=True),
			'SupportVectorMachine_rbf': SVC(max_iter=1000, random_state=self.seed, probability=True),
			'SupportVectorMachine_sigmoid': SVC(max_iter=1000, random_state=self.seed, probability=True)
		}
	
	def get_estimators(self, algorithm_names: List[str]) -> List[BaseEstimator]:
		
		"""Returns a list of estimator instances for the specified algorithm names"""
		return [self.algorithm_dict[name] for name in algorithm_names]
	
	def get_all_estimators(self) -> List[BaseEstimator]:
		
		"""Returns a list of all available estimator instances"""
		return list(self.algorithm_dict.values())
	
	def get_algorithm_names(self) -> List[str]:
		
		"""Returns a list of all available algorithm names"""
		return list(self.algorithm_dict.keys())


def plot_search_results(search, algorithm, RFECV_FS, outdir, method='Grid'):

	'''
	Plot cross-validation results from a fitted GridSearchCV or RandomizedSearchCV.

	For Grid search: one subplot per hyperparameter showing mean train/test scores
	with error bars while all other parameters are fixed at their best values.
	Saves to <algorithm>_<RFECV_FS>_grid.pdf.

	For Random search: a single panel showing mean test and train scores sorted by
	descending test score across all sampled iterations, with a vertical marker at
	the best iteration. Saves to <algorithm>_<RFECV_FS>_random.pdf.
	'''

	results     = search.cv_results_
	means_test  = results['mean_test_score']
	stds_test   = results['std_test_score']
	means_train = results['mean_train_score']
	stds_train  = results['std_train_score']

	if method == 'Grid':

		## Getting indexes of values per hyper-parameter
		masks       = []
		masks_names = list(search.best_params_.keys())

		for p_k, p_v in search.best_params_.items():

			masks.append(list(results['param_' + p_k].data == p_v))

		parameter = search.param_grid

		## Plotting results
		fig, ax = plt.subplots(1, len(parameter), sharex='none', sharey='all', figsize=(20, 5))
		fig.suptitle('Score per parameter')
		fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')

		for i, p in enumerate(masks_names):

			if len(masks_names) > 1:

				m                = np.stack(masks[:i] + masks[i+1:])
				best_parms_mask  = m.all(axis=0)
				best_index       = np.where(best_parms_mask)[0]
				x  = np.array(parameter[p])
				y_1, e_1 = np.array(means_test[best_index]),  np.array(stds_test[best_index])
				y_2, e_2 = np.array(means_train[best_index]), np.array(stds_train[best_index])
				ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
				ax[i].errorbar(x, y_2, e_2, linestyle='-',  marker='^', label='train')
				ax[i].set_xlabel(p.upper())

			else:

				m                = np.array(masks)
				best_parms_mask  = m.all(axis=0)
				best_index       = np.where(best_parms_mask)[0]
				x  = np.array(parameter[p])
				y_1, e_1 = np.array(means_test),  np.array(stds_test)
				y_2, e_2 = np.array(means_train), np.array(stds_train)
				ax.errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
				ax.errorbar(x, y_2, e_2, linestyle='-',  marker='^', label='train')
				ax.set_xlabel(p.upper())

		plt.legend()
		plt.savefig(outdir + '/' + algorithm + '_' + RFECV_FS + '_grid.pdf')
		plt.clf()

	else:

		## Sort all sampled iterations by descending mean test score
		order       = np.argsort(means_test)[::-1]
		x           = np.arange(len(order))
		best_pos    = np.where(order == search.best_index_)[0][0]

		fig, ax = plt.subplots(figsize=(12, 5))
		ax.errorbar(x, means_test[order],  stds_test[order],  linestyle='--', marker='o', label='test')
		ax.errorbar(x, means_train[order], stds_train[order], linestyle='-',  marker='^', label='train')
		ax.axvline(best_pos, color='red', linestyle=':', label='best iteration')
		ax.set_xlabel('Iteration (sorted by test score)')
		ax.set_ylabel('Mean score')
		ax.set_title('RandomizedSearchCV – score across sampled iterations\n' + algorithm)
		ax.legend()
		plt.tight_layout()
		plt.savefig(outdir + '/' + algorithm + '_' + RFECV_FS + '_random.pdf')
		plt.clf()


def SearchCV_(Xtrain, ytrain, FSS, RFECV_FS, params, outdir, method='Grid'):

	'''
	Run hyperparameter search (GridSearchCV or RandomizedSearchCV) for each
	algorithm defined in params, using the selected feature subset FSS.
	Models whose best score meets the threshold are saved as .sav files and their
	results written to hyptuning_results_<RFECV_FS>.tsv; models that fall below
	the threshold are recorded in discarded_results_<RFECV_FS>.tsv.
	Returns a DataFrame of best scores and standard deviations for accepted models.
	'''

	hyperdf = pd.DataFrame(0,    index=indexes, columns=[metric, 'stds'])
	wastedf = pd.DataFrame(9999, index=indexes, columns=[metric, 'stds'])

	# Initialize the algorithm mapper
	algorithm_mapper = AlgorithmMapper(seed=c.seed)

	for i, (k, v) in enumerate(params.items()):

		try:

			if method == 'Grid':
				
				log('GridSearchCV tuning for ' + str(algorithm_mapper.get_estimators([k])[0]) + ' using ' + str(RFECV_FS) + ' selected features n= ' + str(len(FSS)))
				
				search = GridSearchCV(
					estimator=algorithm_mapper.get_estimators([k])[0],
					param_grid=v, n_jobs=threads, cv=cv,
					scoring=metric, error_score='raise', return_train_score=True)
			else:
				
				log('RandomizedSearchCV tuning for ' + str(algorithm_mapper.get_estimators([k])[0]) + ' using ' + str(RFECV_FS) + ' selected features n= ' + str(len(FSS)))
				search = RandomizedSearchCV(
					estimator=algorithm_mapper.get_estimators([k])[0],
					param_distributions=v, n_jobs=threads, cv=cv,
					scoring=metric, error_score='raise', return_train_score=True,
					random_state=seed)

			result = search.fit(Xtrain[FSS], ytrain)
			log((('GridSearchCV' if method == 'Grid' else 'RandomizedSearchCV') +
				' results for ' + str(algorithm_mapper.get_estimators([k])[0]) +
				' and ' + str(RFECV_FS) + ': best f1 ' + str(result.best_score_) +
				' using ' + str(result.best_params_)))

		except Exception as e:

			if method == 'Grid':
				
				error('GridSearchCV tuning failed for ' + str(algorithm_mapper.get_estimators([k])[0]) + ' using ' + str(RFECV_FS) + ' selected features: ' + str(e))
			
			else:
				
				warn('RandomizedSearchCV tuning failed for ' + str(algorithm_mapper.get_estimators([k])[0]) + ' using ' + str(RFECV_FS) + ' and ' + str(v) + ' hyperparameters ' + str(e))
			
			continue

		plot_search_results(result, indexes[i], RFECV_FS, outdir, method=method)

		ix         = result.best_index_
		beststds   = result.cv_results_['std_test_score'][ix]
		means      = result.cv_results_['mean_test_score']
		stds       = result.cv_results_['std_test_score']
		result_params = result.cv_results_['params']

		if result.best_score_ >= threshold:

			log('Writing ML models with best ' + str(metric) + ' score > ' + str(threshold))
			hyperdf.loc[indexes[i], metric]       = result.best_score_
			hyperdf.loc[indexes[i], 'stds']       = beststds
			hyperdf.loc[indexes[i], 'n_features'] = len(FSS)

			with open(outdir + '/' + indexes[i] + '_hyptuning_' + RFECV_FS + '.tsv', 'a') as fout:

				fout.write('Best {}: {} ({}) using {}\n'.format(metric, result.best_score_, beststds, result.best_params_))

				for mean, stdev, param in zip(means, stds, result_params):

					fout.write('{} : {} {} with {}\n'.format(metric, mean, stdev, param))

				joblib.dump(result.best_estimator_, outdir + '/' + indexes[i] + '_' + metric + '_' + RFECV_FS + '_alg.sav')

		else:

			log('Writing discarded ML models')
			wastedf.loc[indexes[i], metric] = result.best_score_
			wastedf.loc[indexes[i], 'stds'] = beststds

	if not hyperdf.empty and hyperdf.shape[0] > 0:

		hyperdf.to_csv(outdir + '/' + 'hyptuning_results_' + RFECV_FS + '.tsv', sep='\t')

	else:

		warn('No hyptuning models were written, considering decrease f1 threshold')

	if (wastedf[metric] != 9999).any():

		wastedf = wastedf[wastedf['f1'] != 9999]
		wastedf.to_csv(outdir + '/' + 'discarded_results_' + RFECV_FS + '.tsv', sep='\t')

	else:

		warn('No models were discarded with f1 threshold ' + str(threshold))

	return hyperdf



def run(parser,args):

	c.INPUT= args.input
	c.OUT=args.output
	c.label=args.label
	c.additional=args.additional
	c.feature_selection=args.feature_selection
	c.threshold=args.threshold
	c.search=args.search
	c.json=args.json
	c.metadata=args.metadata
	c.metric=args.metric
	c.threads=args.threads
	c.seed=args.seed
	c.verbose=args.verbose
	c.seed=args.seed
	c.n_splits=args.n_splits
	c.n_repeats=args.n_repeats

	global cv
	cv = RepeatedStratifiedKFold(n_splits=c.n_splits, n_repeats=c.n_repeats, random_state=c.seed)

	global seed
	seed=c.seed

	global params
	params=load_model_parameters(c.json)

	global indexes
	indexes = ['RandomForestClassifier', 'LogisticRegression_l2','LogisticRegression_elasticnet','LogisticRegression_l1','SVC_linear', 'SVC_rbf','SVC_sigmoid','SVC_mixed', 'KNeighborsClassifier', 'DecisionTreeClassifier' , 'Perceptron_l1','Perceptron_l2','Perceptron_elasticnet','GradientBoost','GaussianNB']

	global metric
	metric=c.metric
	
	global threads
	threads=c.threads

	global threshold
	threshold = c.threshold
	
	if not os.path.exists(c.OUT):

		try:

			os.makedirs(c.OUT)

		except:

			error('Cannot create the output folder')
			sys.exit(1)

	else:

		if not os.access(os.path.abspath(c.OUT),os.W_OK):

			error('Missing write permissions on the output folder')
			sys.exit(1)
			
		elif os.listdir(os.path.abspath(c.OUT)):

			error('The output folder is not empty: specify another output folder or clean the current one')
			sys.exit(1)

	try:

		dataframe = pd.read_csv(c.INPUT, sep='\t', index_col=None) #sampled 
		dataframe = dataframe.drop(columns=[col for col in dataframe.columns if "extraction_ID" in col or "diagnostics" in col])
		dataframe = dataframe.select_dtypes(include=['number'])
		dataframe.dropna(inplace=True)

	except:

		error('TSV ' + c.INPUT + ' does not exist, is not readable or is not a valid TSV')
		sys.exit(1)

	try:

		FS = pd.read_csv(c.feature_selection, sep='\t', index_col=0) #sampled 

	except:

		error('TSV ' + c.feature_selection + ' does not exist, is not readable or is not a valid TSV')
		sys.exit(1)

	if not c.verbose:
	
		log('Disable scikit-learn warnings')
		warnings.simplefilter("ignore")
		os.environ["PYTHONWARNINGS"] = "ignore"

	else: 

		log('Enable scikit-learn warnings')
	
	if not c.metadata:

		#drop label column
		X_train = dataframe.drop(columns = c.label)
		y_train = dataframe[c.label]

	else:

		try:

			metadata = pd.read_csv(c.metadata, sep='\t', index_col='ptid')

		except:

			error('TSV ' + c.metadata + ' does not exist, is not readable or is not a valid TSV')
			sys.exit(1)

		#drop label column 
		training = mergeDFs(dataframe, metadata[c.label])
		training.dropna(inplace=True)
		X_train = training.drop(columns = c.label)
		y_train = training[c.label]
	
	if c.additional:

		try:

			# Test dataset
			additional = pd.read_csv(c.additional, sep="\t", index_col='ptid')

		except:

			error('TSV' + c.additional + ' does not exist, is not readable or is not a valid TSV')
			sys.exit(1)

		training=mergeDFs(dataframe,additional)
		training.dropna(inplace=True)
		X_train = training.drop(columns = c.label)
		y_train = training[c.label]

	indexes = list(params.keys())
	log('Start Hyperparameters Tuning with ' + str(threads) + ' threads and ' + str(metric) + ' scorer')

	for col in FS.columns:

		FS_ = list(FS[col].dropna().sort_values().index)
		
		if FS_:
			
			if c.search=='GridSearchCV':

				# GridSearchCV to find the optimal number of features 
				log('Hyperparameters tuning with ' + c.search + ' ' + metric + ' scorer using ' + str(col))
				hyperdf_ = SearchCV_(X_train, y_train, FS_, col, params, c.OUT, method='Grid')

			else:

				# RandomizedSearchCV to find the optimal number of features 
				log('Hyperparameters tuning with ' + c.search + ' ' + metric + ' scorer using ' + str(col))
				hyperdf_ = SearchCV_(X_train, y_train, FS_, col, params, c.OUT, method='Random')

		else:

			warn('None features were selected by ' + str(col))
			continue
		
	log('Done')
	sys.exit(0)