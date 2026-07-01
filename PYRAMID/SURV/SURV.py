import argparse
from argparse import HelpFormatter
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
import warnings
from sklearn.exceptions import FitFailedWarning
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
	concordance_index_censored,
	concordance_index_ipcw,
	cumulative_dynamic_auc,
	integrated_brier_score,
)
import joblib
from joblib import Parallel, delayed
from sklearn.base import clone
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sklearn.inspection import permutation_importance
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
from lifelines.plotting import add_at_risk_counts
from contextlib import redirect_stdout, redirect_stderr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
# Create a "black hole" for all output

with open(os.devnull, 'w') as fnull:
	
	with redirect_stdout(fnull), redirect_stderr(fnull):
		
		from archetypes import AA

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


class c():

	INPUT=''
	validation=''
	OUT=''
	search=''
	metadata_training=''
	metadata_validation=''
	additional_training=''
	additional_validation=''
	feature_selection=''
	seed=''
	json=''
	n_splits=''
	n_repeats=''
	verbose=''
	transformation=''
	threads=''
	archetypes=''

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
	"""
	Inner-joins two DataFrames on their indices.

	Parameters
	----------
	df1 : pd.DataFrame
	df2 : pd.DataFrame

	Returns
	-------
	pd.DataFrame
		Merged DataFrame containing only rows present in both inputs.
	"""

	df_final = df1.merge(df2, left_index=True, right_index=True)

	return df_final


def format_(dataframe):
	"""
	Converts a DataFrame with OS.status and OS.time columns into a structured
	NumPy array suitable for scikit-survival estimators.
	"""

	newy = []

	for event in zip(dataframe['OS.status'],dataframe['OS.time']):

		if event[0] == 0:

			newy.append((False,event[1]))

		else:

			newy.append((True,event[1]))    

	array_= np.array(newy, dtype=[('Status', '?'), ('Survival_time', '<f8')])

	return array_

def plotEstimatedCurve(data_x, data_y, outdir):
	"""
	Plots a Kaplan-Meier estimated survival curve with log-log confidence intervals
	and saves it to <outdir>/estimated_curve.pdf.
	"""

	time, survival_prob, conf_int = kaplan_meier_estimator(
		data_y["Status"], data_y["Survival_time"], conf_type="log-log"
	)
	plt.step(time, survival_prob, where="post")
	plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
	plt.ylim(0, 1)
	plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
	plt.xlabel("time $t$")
	plt.savefig(outdir + "/" + "estimated_curve.pdf")
	plt.clf()


def transform(parser, args, transformation, Xtrain, Xtest=None):
	
	'''Set the data transforms'''
	
	has_test = Xtest is not None

	if transformation == "scaler":
		
		log('Perform standardization on train dataset')
		
		try:
			
			if has_test:
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]

			scaler = StandardScaler()
			scaler.fit(Xtrain)
			Xtrain = pd.DataFrame(scaler.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log('Perform standardization on test dataset')
				Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		
		except Exception as e:
			
			error(f'Standardization failed using {transformation} : {e}')
			sys.exit(1)

	elif transformation == "normalize":
		
		log('Perform normalization on train dataset')
		
		try:
			
			if has_test:
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]

			normalizer = MinMaxScaler()
			normalizer.fit(Xtrain)
			Xtrain = pd.DataFrame(normalizer.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log('Perform normalization on test dataset')
				Xtest = pd.DataFrame(normalizer.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		except Exception as e:
			
			error(f'Normalization failed using {transformation} : {e}')
			sys.exit(1)

	elif transformation == "mixture":
		
		log('Perform mixture (standardization + normalization) on train dataset')
		
		try:
			
			if has_test:
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]

			scaler = StandardScaler()
			scaler.fit(Xtrain)
			Xtrain = pd.DataFrame(scaler.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			normalizer = MinMaxScaler()
			normalizer.fit(Xtrain)
			Xtrain = pd.DataFrame(normalizer.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log('Perform mixture (standardization + normalization) on test dataset')
				Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
				Xtest = pd.DataFrame(normalizer.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		
		except Exception as e:
			
			error(f'Standardization and Normalization failed using {transformation} : {e}')
			sys.exit(1)

	elif transformation == "robust":
		
		if isinstance(args.robust_parameter[0], (int, float)) and isinstance(args.robust_parameter[1], (int, float)):
			
			log(f'Perform Robust transformation using the range {args.robust_parameter[0]:.2f} , {args.robust_parameter[1]:.2f}')
		
		else:
			
			error('--robust-parameter is not an int or float')
			sys.exit(1)

		log(f'Perform Robust transformation using the range {args.robust_parameter[0]:.2f} , {args.robust_parameter[1]:.2f} on train dataset')
		
		try:
			
			if has_test:
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]

			scaler = RobustScaler(quantile_range=(args.robust_parameter[0], args.robust_parameter[1]))
			scaler.fit(Xtrain)
			Xtrain = pd.DataFrame(scaler.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log(f'Perform Robust transformation using the range {args.robust_parameter[0]:.2f} , {args.robust_parameter[1]:.2f} on test dataset')
				Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		
		except Exception as e:
			
			error(f'Robust transformation failed using {transformation} with {args.robust_parameter[0]:.2f}, {args.robust_parameter[1]:.2f} : {e}')
			sys.exit(1)

	elif transformation == "yeo-johnson":
		
		log('Perform yeo-johnson transformation on train dataset')

		if has_test:
			common_cols = Xtrain.columns.intersection(Xtest.columns)
			Xtrain = Xtrain[common_cols]
			Xtest = Xtest[common_cols]
		
		try:
			
			power = PowerTransformer(method='yeo-johnson', standardize=True)
			power.fit(Xtrain)
			Xtrain = pd.DataFrame(power.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log('Perform yeo-johnson transformation on test dataset')
				Xtest = pd.DataFrame(power.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		
		except Exception as e:
			
			rng = np.random.default_rng(seed)
			warn(f'Power transformation failed using {transformation} : {e} due to many outliers')
			log('Perform Standardization prior to yeo-johnson on train dataset')
			X_scaled_train = (Xtrain - np.mean(Xtrain, axis=0)) / (np.std(Xtrain, axis=0) + 1e-10)
			log('Replace any infinite on train dataset')
			X_scaled_train = np.nan_to_num(X_scaled_train, posinf=1e10, neginf=-1e10)
			log('Handle constant values on train dataset')
			constant_mask = np.std(X_scaled_train, axis=0) < 1e-10
			X_scaled_train[:, constant_mask] = 0
			log('Add tiny noise to prevent perfect correlations on train dataset')
			X_scaled_train += rng.normal(0, 1e-10, X_scaled_train.shape)
			log('Apply yeo-johnson on train dataset')
			power = PowerTransformer(method='yeo-johnson', standardize=True)
			power.fit(X_scaled_train)
			Xtrain = pd.DataFrame(power.transform(X_scaled_train), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log('Perform Standardization prior to yeo-johnson on test dataset')
				X_scaled_test = (Xtest - np.mean(Xtest, axis=0)) / (np.std(Xtest, axis=0) + 1e-10)
				log('Replace any infinite on test dataset')
				X_scaled_test = np.nan_to_num(X_scaled_test, posinf=1e10, neginf=-1e10)
				log('Handle constant values on test dataset')
				constant_mask = np.std(X_scaled_test, axis=0) < 1e-10
				X_scaled_test[:, constant_mask] = 0
				log('Add tiny noise to prevent perfect correlations on test dataset')
				X_scaled_test += rng.normal(0, 1e-10, X_scaled_test.shape)
				log('Apply yeo-johnson on test dataset')
				Xtest = pd.DataFrame(power.transform(X_scaled_test), columns=Xtest.columns, index=Xtest.index)

	elif transformation == "box-cox":
		
		log('Perform box-cox transformation on train dataset')
		
		try:
			
			Xtrain = Xtrain.loc[:, Xtrain.nunique() > 1]  # filter constant cols from train

			if has_test:
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]

			scaler = MinMaxScaler(feature_range=(1, 2))
			scaler.fit(Xtrain)
			Xtrain = pd.DataFrame(scaler.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			scaler = PowerTransformer(method='box-cox', standardize=True)
			scaler.fit(Xtrain)
			Xtrain = pd.DataFrame(scaler.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)

			if has_test:
				
				log('Perform box-cox transformation on test dataset')
				Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		
		except Exception as e:
			
			error(f'Power transformation failed using box-cox : {e}')
			sys.exit(1)

	elif transformation == "quantile-uniform":
		
		if isinstance(args.n_quantiles, (int, float)):
			
			log(f'Perform Uniform Quantile transformation using n_quantile {args.n_quantiles} on train dataset')
		
		else:
			
			error('--n-quantiles is not an int or float')
			sys.exit(1)

		try:
			
			if has_test:
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]

			n_samples_train = Xtrain.shape[0]
			actual_n_quantiles = min(int(args.n_quantiles), n_samples_train)

			if actual_n_quantiles < args.n_quantiles:

				log(f'Adjusting n_quantiles from {args.n_quantiles} to {actual_n_quantiles} due to sample size.')
			
			with warnings.catch_warnings():
				
				warnings.simplefilter("ignore")
				scaler = QuantileTransformer(output_distribution='uniform', n_quantiles=actual_n_quantiles)
				scaler.fit(Xtrain)
				Xtrain = pd.DataFrame(scaler.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log(f'Perform Uniform Quantile transformation using n_quantile {actual_n_quantiles} on test dataset')
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		
		except Exception as e:
			
			error(f'Quantile transformation failed using distribution uniform with {args.n_quantiles} : {e}')
			sys.exit(1)

	elif transformation == "quantile-normal":
		
		if isinstance(args.n_quantiles, (int, float)):
			
			log(f'Perform Normal Quantile transformation using n_quantile {args.n_quantiles} on train dataset')
		
		else:
			
			error('--n-quantiles is not an int or float')
			sys.exit(1)

		try:
			
			if has_test:
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]

			n_samples_train = Xtrain.shape[0]
			actual_n_quantiles = min(int(args.n_quantiles), n_samples_train)

			if actual_n_quantiles < args.n_quantiles:

				warn(f'Adjusting n_quantiles from {args.n_quantiles} to {actual_n_quantiles} due to sample size.')

			with warnings.catch_warnings():
				
				warnings.simplefilter("ignore")
				scaler = QuantileTransformer(output_distribution='normal', n_quantiles=actual_n_quantiles)
				scaler.fit(Xtrain)
				Xtrain = pd.DataFrame(scaler.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log(f'Perform Normal Quantile transformation using n_quantile {actual_n_quantiles} on test dataset')
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		
		except Exception as e:
			
			error(f'Quantile transformation failed using normal distribution with {args.n_quantiles} : {e}')
			sys.exit(1)
			
	return (Xtrain, Xtest) if has_test else Xtrain

def Spearmanfeatures(df):

	"""
	Returns a boolean mask of features to keep after removing those with pairwise
	Spearman correlation >= 0.9. Constant columns (zero variance) are dropped first.
	"""
	# 1. Remove constant columns first (zero variance)
	df = df.loc[:, df.std() > 0]
	
	corr = df.corr(method='spearman')
	
	columns = np.full((corr.shape[0],), True, dtype=bool)
	
	for i in range(corr.shape[0]):
			
		for j in range(i+1, corr.shape[0]):
				
			if corr.iloc[i,j] >= 0.9:
				
				if columns[j]:
						
					columns[j] = False
	
	return columns

def fit_and_score_features(X, y):

	"""
	Fits a univariate CoxPH model for each feature independently and returns
	"""

	n_features = X.shape[1]
	scores = np.empty(n_features)
	coef= np.empty(n_features)
	m = CoxPHSurvivalAnalysis()
	
	for j in range(n_features):
		
		Xj = X[:, j : j + 1]
		m.fit(Xj, y)
		scores[j] = m.score(Xj, y)
		coef[j] = m.coef_
	
	return scores,coef

def plotGridResults(grid_results,grid_obj,penalization,outdir):

	"""
	Plots mean concordance index vs alpha from a CoxnetSurvivalAnalysis
	GridSearchCV, marking the best alpha with a vertical line.
	Saves to <outdir>/<penalization>.grid.results.pdf.
	"""

	#plot the results of GridSearchCV
	alphas = grid_results.param_alphas.map(lambda x: x[0])
	mean = grid_results.mean_test_score
	std = grid_results.std_test_score

	fig, ax = plt.subplots(figsize=(9, 6))
	ax.plot(alphas, mean)
	ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
	ax.set_xscale("log")
	ax.set_ylabel("concordance index")
	ax.set_xlabel("alpha")
	ax.axvline(grid_obj.best_params_["alphas"][0], c="C1")
	ax.axhline(0.5, color="grey", linestyle="--")
	ax.grid(True)
	plt.savefig(outdir + "/" + penalization + ".grid.results.pdf")
	plt.clf()

def plotAlphaDistribution(X,y,formula,n,penalty,alphas,outdir):

	"""
	Fits a penalized Cox model across a range of alpha values,
	collects coefficients, and calls plot_coefficients to generate
	a semi-log coefficient path plot.
	"""

	coefficients = {}
	cph = formula

	for alpha in alphas:
		
		cph.set_params(alpha=alpha)
		cph.fit(X, y)
		key = round(alpha, 5)
		coefficients[key] = cph.coef_

	coefficients = pd.DataFrame.from_dict(coefficients).rename_axis(index="feature", columns="alpha").set_index(X.columns)
	top20 = plot_coefficients(coefficients, n_highlight=n, penalization=penalty,outdir=outdir)

	return coefficients

def plot_coefficients(coefs, n_highlight, penalization,outdir): # n_highlight send in command line also outdir 

	"""
	Plots a semi-log coefficient path across alpha values for a penalized
	Cox model and labels the top n_highlight features at the smallest alpha.
	Saves to <outdir>/<penalization>.coefficients.pdf.
	"""
	
	_, ax = plt.subplots(figsize=(9, 6))
	alphas = coefs.columns
	
	for row in coefs.itertuples():
		
		ax.semilogx(alphas, row[1:], ".-", label=row.Index)

	alpha_min = alphas.min()
	top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
	
	for name in top_coefs.index:
		
		coef = coefs.loc[name, alpha_min]
		plt.text(
			alpha_min, coef, name + "   ",
			horizontalalignment="right",
			verticalalignment="center",
			fontdict={'size':8}
		)

	ax.yaxis.set_label_position("right")
	ax.yaxis.tick_right()
	ax.grid(True)
	ax.set_xlabel("alpha")
	ax.set_ylabel("coefficient")
	plt.tight_layout()
	plt.savefig(outdir + "/" + penalization + ".coefficients.pdf")
	plt.clf()
	
	return top_coefs

def NonZeroFeatures(gcv,Xt,penalization,outdir):

	"""
	Extracts non-zero coefficient features from the best GridSearchCV estimator,
	plots a horizontal bar chart, and saves it to <outdir>/<penalization>.coefficients.selected.pdf.
	"""

	best_model = gcv.best_estimator_
	best_coefs = pd.DataFrame(best_model.coef_, index=Xt.columns, columns=["coefficient"])

	non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
	log('Number of non-zero coefficients using '+ str(penalization) + ' penalization: ' + str(non_zero))

	non_zero_coefs = best_coefs.query("coefficient != 0")
	coef_order = non_zero_coefs.abs().sort_values("coefficient").index
	non_zero_coefs=non_zero_coefs.sort_values(by='coefficient', key=pd.Series.abs, ascending=False)
	_, ax = plt.subplots(figsize=(6, 8))
	non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
	ax.set_xlabel("coefficient")
	ax.grid(True)
	plt.tight_layout()
	plt.savefig(outdir + "/" + penalization + ".coefficients.selected.pdf")
	plt.clf()

	return non_zero_coefs

def GetTimes(ytest):

	"""
	Computes an array of 15 time-points spanning the 5th-to-95th percentile
	of event times, safe for use with cumulative_dynamic_auc.
	"""

	n_points = 15
	q1_rank = 5
	q3_rank = 95

	times_arr = ytest["Survival_time"]   # field name you used earlier
	events = ytest["Status"]              # adjust if your event field is named differently

	if np.sum(events) == 0:
		
		raise ValueError("No uncensored events in ytest — cannot compute cumulative_dynamic_auc.")

	t_min = times_arr.min()
	t_max_event = times_arr[events == 1].max()   # largest time where an event actually occurred

	percentiles = np.linspace(q1_rank, q3_rank, n_points)
	times = np.percentile(times_arr, percentiles)

	# Clip to the valid event range (keeps length = n_points)
	times = np.clip(times, t_min, t_max_event)

	# If clipping produced duplicate values (possible if many percentiles > t_max_event),
	# replace with an evenly spaced grid inside valid event window:
	if len(np.unique(times)) < n_points:
		
		times = np.linspace(t_min, t_max_event, n_points)

	return(times)

def evaluate_topk(obj, Xtrain, ytrain, features):

	"""
	Clones a model, fits it on a feature subset, and returns the training
	C-index. Used for parallelised top-k feature selection.
	"""
	
	model = clone(obj)
	model.fit(Xtrain[features], ytrain)
	
	return model.score(Xtrain[features], ytrain)

def plot_search_results(grid, tag, outdir):

	"""
	Params: 
		grid: A trained GridSearchCV object.
	"""
	## Results from grid search
	results = grid.cv_results_
	means_test = results['mean_test_score']
	stds_test = results['std_test_score']
	means_train = results['mean_train_score']
	stds_train = results['std_train_score']

	## Getting indexes of values per hyper-parameter
	masks=[]
	masks_names= list(grid.best_params_.keys())
	
	for p_k, p_v in grid.best_params_.items():
		
		masks.append(list(results['param_'+p_k].data==p_v))

	parameter=grid.param_grid

	## Ploting results
	fig, ax = plt.subplots(1,len(parameter),sharex='none', sharey='all',figsize=(20,5))
	fig.suptitle('Score per parameter')
	fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
	pram_preformace_in_best = {}
		
	for i, p in enumerate(masks_names):
		
		if len(masks_names) > 1:

			m = np.stack(masks[:i] + masks[i+1:])
			best_parms_mask = m.all(axis=0)
			best_index = np.where(best_parms_mask)[0]
			x = np.array(parameter[p])
			y_1 = np.array(means_test[best_index])
			e_1 = np.array(stds_test[best_index])
			y_2 = np.array(means_train[best_index])
			e_2 = np.array(stds_train[best_index])
			ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
			ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
			ax[i].set_xlabel(p.upper())

		else: 

			m = np.array(masks)
			best_parms_mask = m.all(axis=0)
			best_index = np.where(best_parms_mask)[0]
			x = np.array(parameter[p])
			y_1 = np.array(means_test)
			e_1 = np.array(stds_test)
			y_2 = np.array(means_train)
			e_2 = np.array(stds_train)
			ax.errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
			ax.errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
			ax.set_xlabel(p.upper())
		

	plt.legend()
	plt.savefig(outdir + "/" + tag + "_grid.pdf")
	plt.clf()

def PlotSurvivalFunction(obj,surv,outdir,tag):

	"""
	Plots predicted survival functions for each test subject and saves
	to <outdir>/survival_function.<tag>.pdf.
	"""

	fig, ax = plt.subplots(constrained_layout=True)

	for i, s in enumerate(surv):
		
		ax.step(obj.unique_times_, s, where="post", label=str(i))

	ax.set_ylabel("Survival probability")
	ax.set_xlabel("Time")
	# Legend outside bottom
	ax.legend(
		loc="upper center",
		bbox_to_anchor=(0.5, -0.2),
		ncol=4,
		frameon=False
	)
	ax.grid(True)

	fig.savefig(outdir + "/" + "survival_function." + tag + ".pdf",bbox_inches="tight")
	plt.close(fig)
	plt.clf()

def PlotCumulativeHazard(obj,surv,outdir,tag):

	"""
	Plots predicted cumulative hazard functions for each test subject
	and saves to <outdir>/cumulative_hazard_function.<tag>.pdf.
	"""

	fig, ax = plt.subplots(constrained_layout=True)
	
	for i, s in enumerate(surv):
		
		ax.step(obj.unique_times_, s, where="post", label=str(i))

	ax.set_ylabel("Survival probability")
	ax.set_xlabel("Time")
	# Legend outside bottom
	ax.legend(
		loc="upper center",
		bbox_to_anchor=(0.5, -0.2),
		ncol=4,
		frameon=False
	)
	ax.grid(True)
	fig.savefig(outdir + "/" + "cumulative_hazard_function." + tag + ".pdf",bbox_inches="tight")
	plt.close(fig)
	plt.clf()

def safe_get(var_name, idx=None):

	"""
	Safely retrieves a variable from the local or global scope by name,
	optionally indexing into it. Returns np.nan on any failure.
	"""
	
	if var_name in locals() or var_name in globals():
		
		val = locals().get(var_name, globals().get(var_name))
		
		if idx is not None:
			
			try:
				
				return val[idx]
			
			except Exception as e:
				
				return np.nan
		
		return val
	
	return np.nan



def EvaluateCoxModel(mdl,Xtrain, Xtest, ytrain, ytest, features,penalization,outdir):

	"""
	Fits a Cox model, computes time-dependent AUC on the test set,
	plots the AUC curve, saves the model to disk, and returns evaluation metrics.
	"""

	cph_=mdl.fit(Xtrain[features], ytrain)

	coef = np.ravel(cph_.coef_)
	hr = np.exp(coef)

	df_hr = pd.DataFrame({
		"Feature": Xtrain[features].columns,
		"Coefficient": coef,
		"Hazard_Ratio": hr
	})

	y_tot = np.concatenate((ytrain["Survival_time"], ytest["Survival_time"]))
	
	times = GetTimes(ytest)

	cph_risk_scores = cph_.predict(Xtest[features])
	cph_risk_scores_train = cph_.predict(Xtrain[features])
	cph_auc, cph_mean_auc = cumulative_dynamic_auc(ytrain, ytest, cph_risk_scores, times)

	plt.plot(times, cph_auc, marker="o")
	plt.axhline(cph_mean_auc, linestyle="--")
	plt.xlabel("time from enrollment")
	plt.ylabel("time-dependent AUC")
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(outdir + "/" + "ROC.time.dependent." + penalization + ".pdf")
	plt.clf()

	joblib.dump(cph_, outdir + '/' + 'cph_model_' + penalization + '.sav')

	return cph_auc, cph_mean_auc, df_hr, cph_risk_scores, cph_risk_scores_train

def fit_evaluate_cox(mdl, X_train, X_test, y_train, y_test, features, label, outdir):

	"""
	Wrapper around EvaluateCoxModel that additionally computes Harrell's C and
	Uno's C concordance indices and returns a summary DataFrame.
	"""
	
	auc, mean_auc, df_hr, risk_scores, risk_scores_train = EvaluateCoxModel(
		mdl, X_train, X_test, y_train, y_test, features, label, outdir
	)

	c_harrell = concordance_index_censored(y_test["Status"], y_test["Survival_time"], risk_scores)
	c_uno = concordance_index_ipcw(y_train, y_test, risk_scores)

	df_summary = pd.DataFrame({
		"mean_auc": [mean_auc],
		"c_harrell": [c_harrell[0] if isinstance(c_harrell, (tuple, list, np.ndarray)) else c_harrell],
		"c_uno": [c_uno[0] if isinstance(c_uno, (tuple, list, np.ndarray)) else c_uno],
		"n_features": [len(features)],
		"features": [",".join(map(str, features))]
	}, index=[f"cph_{label}"])

	return auc, mean_auc, df_hr, risk_scores, df_summary, risk_scores_train

def random_search_cox(X_train, y_train, estimator, param_grid, cv, threads, seed):

	"""
	Runs RandomizedSearchCV on a survival estimator and returns the fitted search object.
	"""
	
	search = RandomizedSearchCV(estimator, param_grid, cv=cv, n_jobs=threads, random_state=seed, return_train_score=True)
	search.fit(X_train, y_train)
	
	return search

def grid_search_cox(X_train, y_train, estimator, param_grid, cv, threads,tag,outdir):

	"""
	Runs GridSearchCV on a survival estimator, plots the results, and returns
	the fitted search object.
	"""
	
	search = GridSearchCV(estimator, param_grid, cv=cv, n_jobs=threads, return_train_score=True)
	search.fit(X_train, y_train)
	
	plot_search_results(search, tag, outdir)

	return search

def penalized_cox_analysis(X_train, y_train, l1_ratio, outdir, label,search):

	"""
	Fits a CoxnetSurvivalAnalysis model with the specified L1 ratio,
	performs hyperparameter tuning over the alpha path, and returns the
	best model and its non-zero features.
	"""
	
	log(f"Fitting {label} penalized Cox model")
	
	# 1. Force X into a pure NumPy array immediately
	X_pure = X_train.to_numpy()
	
	# 2. Force y into a 1D Structured NumPy Array
	# This is the most common cause of the indexing error in survival analysis
	if isinstance(y_train, (pd.DataFrame, pd.Series)):
		
		# We convert the DataFrame/Series to a structured array with named fields
		# Survival data requires (event, time) format
		y_pure = np.array(
			[tuple(x) for x in (y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train)],
			dtype=[('Status', bool), ('Survival_time', float)]
		)
	
	else:
		
		y_pure = y_train

	# Initialize the model
	cox = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01)
	
	# Single fit to get the alpha path
	cox.fit(X_pure, y_pure)
	
	# 3. Process Coefficients (using original X_train for column names)
	coefs = pd.DataFrame(cox.coef_, index=X_train.columns, columns=np.round(cox.alphas_, 5))
	top_features = plot_coefficients(coefs, 5, label, outdir)

	# 4. Hyperparameter Tuning with pure NumPy arrays
	estimated_alphas = cox.alphas_
	params={"alphas": [[v] for v in estimated_alphas]}

	if search == "RandomizedSearchCV":

		gcv = random_search_cox(X_pure, y_pure, cox, params, cv, threads, seed)
	
	else:

		gcv = grid_search_cox(X_pure, y_pure, cox, params, cv, threads,label,outdir)

	log(f"Best c-index for {label}: {gcv.best_score_} with alpha {float(gcv.best_params_['alphas'][0])}")
	
	# Use X_train here if NonZeroFeatures expects a DataFrame with headers
	non_zero_features = NonZeroFeatures(gcv, X_train, label, outdir)
	non_zero_features.to_csv(f"{outdir}/features_non_zero.{label}.txt", sep="\t")
	
	return gcv, non_zero_features


def FitAndEvaluate_(model, parameters, Xtrain, Xtest, ytrain, ytest, features, outdir, tag, cv, threads, seed=None, method='Grid'):

	"""
	Fits a survival model using Grid or Random hyperparameter search, evaluates it
	on the test set, saves the best model to disk, and returns performance metrics.
	"""
	log(f"Hyperparameter tuning for {model.__class__.__name__} using '{tag}' features")

	if method == 'Random':
		
		gcv = RandomizedSearchCV(model, param_distributions=parameters, cv=cv,
								 return_train_score=True, error_score=0.5,
								 n_jobs=threads, random_state=seed)
	else:
		
		gcv = GridSearchCV(model, param_grid=parameters, cv=cv,
						   return_train_score=True, error_score=0.5, n_jobs=threads)

	gcv.fit(Xtrain[features], ytrain)

	if method == 'Grid':
		
		plot_search_results(gcv, f"{model.__class__.__name__}_{tag}", outdir)

	best_model = gcv.best_estimator_
	best_model.fit(Xtrain[features], ytrain)

	# Compute predictions and AUC
	risk_scores = best_model.predict(Xtest[features])
	risk_scores_train = best_model.predict(Xtrain[features])
	c_index = best_model.score(Xtest[features], ytest)

	try:

		times = GetTimes(ytest)
		mod_auc, mod_mean_auc = cumulative_dynamic_auc(ytrain, ytest, risk_scores, times)
		c_harrell = concordance_index_censored(ytest["Status"], ytest["Survival_time"], risk_scores)
		c_uno = concordance_index_ipcw(ytrain, ytest, risk_scores)

	except Exception as e:

		warn(f'Unable to compute survival metrics: {e}')
		mod_auc = mod_mean_auc = c_harrell = c_uno = None

	# Save model
	joblib.dump(best_model, f"{outdir}/{model.__class__.__name__}_{tag}.sav")

	return risk_scores, mod_auc, mod_mean_auc, c_index, c_harrell, c_uno, best_model, risk_scores_train


def plot_feature_selection(results_dict, outpath, title="Feature selection"):

	"""
	Plots C-index vs number of top features and marks the best k with a
	vertical red dashed line. Saves to outpath.
	"""
	
	best_k = max(results_dict, key=results_dict.get)
	plt.plot(list(results_dict.keys()), list(results_dict.values()), marker="o")
	plt.axvline(best_k, color="red", linestyle="--", label=f"Best k = {best_k}")
	plt.xlabel("Number of top features")
	plt.ylabel("C-index")
	plt.title(title)
	plt.legend()
	plt.tight_layout()
	plt.savefig(outpath)
	plt.clf()
	
	return best_k

def remove_multicollinearity(df, threshold=5.0):

	"""
	Iteratively removes features with high Variance Inflation Factor (VIF).
	Run this AFTER your Spearman filter to save computation time.
	"""

	# Work on a copy and drop any remaining NaNs
	X = df.copy().dropna()
	
	# VIF requires a constant (intercept) to be mathematically accurate
	# We don't remove the constant, just use it for calculation
	X_with_const = add_constant(X)
	
	while True:
		
		# Calculate VIF for all remaining features
		# We skip the 'const' column (index 0) in the output
		vifs = [variance_inflation_factor(X_with_const.values, i) 
				for i in range(X_with_const.shape[1])]
		
		# Create a series to map VIFs to feature names
		vif_series = pd.Series(vifs, index=X_with_const.columns)
		
		# Exclude the constant from the removal consideration
		vif_series = vif_series.drop('const')
		
		# Find the highest VIF
		max_vif = vif_series.max()
		
		if max_vif > threshold:
			
			feature_to_remove = vif_series.idxmax()
			log(f"Removing '{feature_to_remove}' with VIF: {max_vif:.2f}")
			X_with_const = X_with_const.drop(columns=[feature_to_remove])
		
		else:
			break
			
	# Return the list of "safe" column names
	return [c for c in X_with_const.columns if c != 'const']


def RunModelPipeline(model_name, model, params, Xtrain, Xtest, ytrain, ytest, feature_sets, outdir, threads, search):

	"""
	Fits a survival model with Grid or Random hyperparameter search across
	multiple feature sets, computes evaluation metrics, plots survival and
	"""
	
	summaries = []
	risk_tables = {}
	risk_tables_train= {}  

	for tag, features in feature_sets.items():
		
		X_train_ = Xtrain[features]

		if len(X_train_.shape) == 1:# Convert to DataFrame if it's a Series

			log('X_train is a Series... Convert to DataFrame')
			X_train_ = X_train_.to_frame().astype(np.float64)
			#X_test = X_test.to_frame().astype(np.float64)

		elif X_train_.shape[1] ==1:

			log('Set X_train dtype as float')
			X_train_ = X_train_.astype(np.float64)

		try:

			# Fit and evaluate
			if search == "RandomizedSearchCV":

				results = FitAndEvaluate_(model=model, parameters=params, Xtrain=X_train_, Xtest=Xtest, ytrain=ytrain, ytest=ytest, features=features, outdir=outdir, tag=tag, cv=cv, threads=threads, seed=seed, method='Random')
			
			else:

				results = FitAndEvaluate_(model=model, parameters=params, Xtrain=X_train_, Xtest=Xtest, ytrain=ytrain, ytest=ytest, features=features, outdir=outdir, tag=tag, cv=cv, threads=threads, method='Grid')

			risk_scores, mod_auc, mod_mean_auc, c_index, c_harrell, c_uno, fitted_model, risk_scores_train = results

			# Plot survival & cumulative hazard
			if hasattr(fitted_model, "predict_survival_function") and hasattr(fitted_model, "predict_cumulative_hazard_function"):

				surv = fitted_model.predict_survival_function(Xtest[features], return_array=True)
				cumhaz = fitted_model.predict_cumulative_hazard_function(Xtest[features], return_array=True)
				PlotSurvivalFunction(fitted_model, surv, outdir, f"{model_name}_{tag}")
				PlotCumulativeHazard(fitted_model, cumhaz, outdir, f"{model_name}_{tag}")
			
			else:

				warn(f"Unable run survival function and cumulative hazard function for {model_name}_{tag}: object has no attribute 'predict_survival_function'")

			# Summary dataframe
			summaries.append(pd.DataFrame({
				"mean_auc": [mod_mean_auc],
				"c_harrell": [c_harrell[0] if isinstance(c_harrell, (tuple, list, np.ndarray)) else c_harrell],
				"c_uno": [c_uno[0] if isinstance(c_uno, (tuple, list, np.ndarray)) else c_uno],
				"n_features": [len(features)],
				"features": [",".join(map(str, features))]
			}, index=[f"{model_name}_{tag}"]))
			
			# Save risk table for this feature set
			risk_tables[f"{model_name}_{tag}"] = risk_scores
			risk_tables_train[f"{model_name}_{tag}"] = risk_scores_train

		except Exception as e:
			
			warn(f"Unable to fit {model_name}_{tag}: {e}")

		# Permutation importance if supported
		try:
			
			log(f"Permutation-based feature importance for {model_name}_{tag}")
			result_perm = permutation_importance(fitted_model, Xtest[features], ytest,
												 n_repeats=15, random_state=seed)
			feature_df = pd.DataFrame({
				"importances_mean": result_perm["importances_mean"],
				"importances_std": result_perm["importances_std"]
			}, index=features).sort_values("importances_mean", ascending=False)

			if feature_df.shape[0] > 1:

				# Select optimal number of features using top-k evaluation
				k_values = range(2, len(feature_df) + 1)
				results_list = Parallel(n_jobs=threads, backend="threading")(
					delayed(evaluate_topk)(fitted_model, X_train_, ytrain, feature_df.index[:k])
					for k in k_values
				)
				results_dict = dict(zip(k_values, results_list))
				best_k = max(results_dict, key=results_dict.get)

				# Plot feature selection results
				plt.plot(list(results_dict.keys()), list(results_dict.values()), marker="o")
				plt.axvline(best_k, color="red", linestyle="--", label=f"Best k = {best_k}")
				plt.xlabel("Number of top features")
				plt.ylabel("C-index")
				plt.title(f"Feature selection with permutation importance ({model_name}_{tag})")
				plt.legend()
				plt.tight_layout()
				plt.savefig(f"{outdir}/permutation_importance.{model_name}_{tag}.pdf")
				plt.clf()

				permutation_features = feature_df.index[:best_k]
				feature_df.iloc[:best_k].to_csv(f"{outdir}/permutation_importance.{model_name}_{tag}.txt", sep="\t")
		
			else:

				warn(f"Skipping 'Select optimal number of features using top-k evaluation' because only one feature is available.")
				feature_df.to_csv(f"{outdir}/permutation_importance.{model_name}_{tag}.txt", sep="\t")

		except Exception as e:
			
			warn(f"Permutation importance skipped for {model_name}_{tag}: {e}")
			permutation_features = features  # fallback to all features

	return {
			"summary": pd.concat(summaries),
			"risk_tables": risk_tables,
			"risk_tables_train": risk_tables_train
			}

def runFS(fsDF, Xtrain, Xtest, ytrain, ytest, outdir, run_aa,search,parameter):

	"""
	Runs the full model pipeline (CoxPH + ensemble models) for each column
	of a feature-selection DataFrame. Optionally runs Archetypal Analysis.
	Saves per-column model metrics to <outdir>/model.metrics.<col>.tsv.
	"""
	
	for col in fsDF.columns:
		
		FS_ = list(fsDF[col].dropna().sort_values().index)
		FS_dict = {str(col): FS_}
		
		results_to_process = []

		# 1. CoxPH
		try:
			log(f"Fitting CoxPH with {col} features")
			cph = CoxPHSurvivalAnalysis()
			(cph_auc_FS, cph_mean_auc_FS, df_hr_FS, 
			 cph_risk_scores_FS, df_cph_summary_FS, 
			 cph_risk_scores_FS_train) = fit_evaluate_cox(cph, Xtrain, Xtest, ytrain, ytest, FS_, str(col), outdir)
			
			if run_aa:
				
				results_to_process.append(("CoxPH", cph_risk_scores_FS, cph_risk_scores_FS_train, f"CoxPH_{col}"))
		
		except Exception as e:
			
			warn(f"Unable to run CoxPH using {col}: {e}")
			df_cph_summary_FS = pd.DataFrame()

		# 2. Ensemble Models
		models = [
			("RSF", RandomSurvivalForest(n_jobs=threads, random_state=seed), parameter["parameters_RF"]),
			("GBS", GradientBoostingSurvivalAnalysis(random_state=seed), parameter["parameters_GB"]),
			("CWGB", ComponentwiseGradientBoostingSurvivalAnalysis(random_state=seed), parameter["parameters_CWGB"]),
			("SSVM", FastSurvivalSVM(random_state=seed), parameter["parameters_SSVM"])
		]

		ensemble_summaries = []
		
		for name, model_obj, params in models:
			
			try:
				
				log(f"Fitting {name} with {col} features")
				res = RunModelPipeline(model_name=name, model=model_obj, params=params, Xtrain=Xtrain, Xtest=Xtest, ytrain=ytrain, ytest=ytest, feature_sets=FS_dict, outdir=outdir, threads=threads,search=search)
				ensemble_summaries.append(res["summary"])

				if run_aa:
					
					# The key in risk_tables usually follows the pattern "{ModelName}_{FeatureSetName}"
					key = f"{name}_{col}"
					
					if key in res["risk_tables"]:
						
						scores_test = res["risk_tables"][key]
						scores_train = res["risk_tables_train"][key]
						results_to_process.append((name, scores_test, scores_train, key))
			
			except Exception as e:
				warn(f"Unable to run {name} using {col}: {e}")

		# -------- Archetypal Analysis Execution -------- #
		if run_aa:
			
			for model_name, scores_test, scores_train, label in results_to_process:
				
				try:
					
					log(f"[AA] Running Archetypes for {label}")
					A_LL, A_LH = GetExtremeArchetypes(Xtrain, FS_, scores_train)
					opt_thresh = FindOptimalThreshold(Xtrain, ytrain, FS_, A_LL, A_LH)
					ProjectAndAnalyze(Xtest, ytest, FS_, outdir, scores_test, label, A_LL, A_LH, opt_thresh)
				
				except Exception as e:
					
					warn(f"Archetypal Analysis failed for {label}: {e}")

		# -------- Save Summaries -------- #
		summary_DF = pd.concat([df_cph_summary_FS] + ensemble_summaries)
		cols_to_update = ['mean_auc', 'c_harrell', 'c_uno']
		summary_DF[cols_to_update] = summary_DF[cols_to_update].applymap(lambda x: 1 - x if x < 0.5 else x)
		summary_DF = summary_DF.round(4)
		summary_DF.to_csv(f"{outdir}/model.metrics.{col}.tsv", sep='\t', index=True)

def plot_km_curves(df, y_test, group_col, title, filename, outdir,label):
	
	"""
	Function to create aesthetic KM plots with At-Risk tables.
	"""
	
	fig, ax = plt.subplots(figsize=(12, 12))
	kmfs = []
	
	unique_groups = (
		df[group_col]
		.dropna()
		.unique()
	)

	# Preserve category order if categorical
	if pd.api.types.is_categorical_dtype(df[group_col]):
		
		unique_groups = df[group_col].cat.categories
	
	else:
		
		unique_groups = sorted(unique_groups)
	
	for group in unique_groups:
		
		mask = (df[group_col] == group)
		
		if mask.any():
			
			kmf = KaplanMeierFitter()
			kmf.fit(y_test["Survival_time"][mask], 
					event_observed=y_test["Status"][mask], 
					label=group)
			kmf.plot_survival_function(ax=ax, show_censors=True,linewidth=2.5, alpha=0.8)
			kmfs.append(kmf)

	# Add At-Risk Table
	add_at_risk_counts(*kmfs, ax=ax)

	# Statistical Test
	if len(unique_groups) < 2:
		
		results = None
	
	elif len(unique_groups) > 2:
		
		results = multivariate_logrank_test(y_test["Survival_time"],df[group_col],y_test["Status"])

	else:
		
		g1, g2 = unique_groups[0], unique_groups[1]
		mask1 = df[group_col] == g1
		mask2 = df[group_col] == g2

		results = logrank_test(
			y_test["Survival_time"][mask1],
			y_test["Survival_time"][mask2],
			y_test["Status"][mask1],
			y_test["Status"][mask2]
		)


	# Styling
	ax.set_title(title, fontsize=16, fontweight='bold', pad=25)
	ax.set_xlabel("Time", fontsize=12)
	ax.set_ylabel("Survival Probability", fontsize=12)
	ax.grid(axis='y', linestyle='--', alpha=0.3)
	ax.legend(loc='best', frameon=True)
	
	# Annotate P-value
	p_val_text = f"Log-rank p: {results.p_value:.4f}" if results.p_value >= 0.001 else "Log-rank p < 0.001"
	ax.annotate(p_val_text, xy=(0.05, 0.15), xycoords='axes fraction', 
				fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cccccc", alpha=0.9))

	plt.tight_layout()
	plt.savefig(f"{outdir}/{filename}.{label}.pdf", bbox_inches='tight')
	plt.close()

def GetExtremeArchetypes(X_train, features, risk_scores_train):
	
	"""
	Finds archetypes using the top and bottom 25% of patients.
	Includes a polarity check to ensure high scores = high risk.
	"""
	
	X_feats = X_train[features].values
	
	# Polarity Correction: Ensure higher scores mean higher risk (shorter survival)
	# If scores are positively correlated with time, they are 'safety scores'; we flip them.
	# We use a simple correlation check or median comparison.
	# Note: this is done on training data to fix the model's 'direction'.
	return_scores = risk_scores_train.copy()
	
	# Define indices for extreme quartiles
	q_low = np.percentile(return_scores, 25)
	q_high = np.percentile(return_scores, 75)
	
	idx_low = return_scores <= q_low
	idx_high = return_scores >= q_high
	
	# Fit Archetypes (k=1)
	aa_low = AA(n_archetypes=1, random_state=seed)
	aa_low.fit(X_feats[idx_low])
	A_LL = aa_low.archetypes_.flatten()
	
	aa_high = AA(n_archetypes=1, random_state=seed)
	aa_high.fit(X_feats[idx_high])
	A_LH = aa_high.archetypes_.flatten()
	
	return A_LL, A_LH

def FindOptimalThreshold(X_train, y_train, features, A_LL, A_LH):
	
	"""
	Finds the threshold on high_risk_weight that maximizes Log-Rank separation.
	"""
	
	X_vals = X_train[features].values
	v = A_LL - A_LH
	den = np.sum(v**2)
	
	# Projection: 0 is A_LH (High), 1 is A_LL (Low)
	w_low = np.dot((X_vals - A_LH), v) / den
	w_high = 1 - np.clip(w_low, 0, 1)

	event_col, time_col = y_train.dtype.names[0], y_train.dtype.names[1]
	best_p = 1.0
	best_t = 0.5
	
	# Search for optimal cut-point between 30th and 70th percentile
	for t in np.linspace(0.3, 0.7, 15):
		
		mask = w_high >= t
		
		if mask.sum() > 10 and (~mask).sum() > 10:
			
			res = logrank_test(y_train[time_col][mask], y_train[time_col][~mask],
							   y_train[event_col][mask], y_train[event_col][~mask])
			
			if res.p_value < best_p:
				
				best_p = res.p_value
				best_t = t
	
	return best_t

def ProjectAndAnalyze(X_test, y_test, features, outdir, risk_scores_test, label, A_LL, A_LH, threshold):
	
	"""
	Projects test data onto the archetype axis and saves results.
	"""
	
	X_vals = X_test[features].values
	v = A_LL - A_LH
	den = np.sum(v**2)

	# Calculate Weights
	w_low = np.dot((X_vals - A_LH), v) / den
	w_low = np.clip(w_low, 0, 1)
	w_high = 1 - w_low

	results_df = pd.DataFrame({
		'ptid': X_test.index,
		'low_risk_weight': w_low,
		'high_risk_weight': w_high,
		'model_risk_score': risk_scores_test
	}).set_index('ptid')

	results_df['aa_label'] = np.where(results_df['high_risk_weight'] >= threshold, "High Risk", "Low Risk")
	
	# Plotting
	if results_df['aa_label'].nunique() >= 2:
		
		plot_km_curves(results_df, y_test, 'aa_label', 
					   f"Survival by Archetype Group ({label})", 
					   "km_binary_aa", outdir, label)

	else:
		warn(f"Skipping binary KM plot for {label}: <2 strata present")

	# Plotting Intensity Results
	results_df['risk_intensity'] = pd.cut(
		results_df['high_risk_weight'], 
		bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
		labels=['Pure Low', 'Lean Low', 'Borderline', 'Lean High', 'Pure High'],
		include_lowest=True
	)

	if results_df['risk_intensity'].dropna().nunique() >= 2:
		
		plot_km_curves(results_df, y_test, 'risk_intensity', 
					   f"Survival by Archetype Intensity ({label})", 
					   "km_intensity_aa", outdir, label)
	else:
		
		warn(f"Skipping intensity KM plot for {label}: <2 strata present")

	results_df.to_csv(f"{outdir}/archetype_analysis_results.{label}.tsv", sep='\t')
	
	#return results_df	
	
def run(parser,args):

	c.INPUT= args.input
	c.validation=args.validation
	c.OUT=args.output
	c.search=args.search
	c.json=args.json
	c.metadata_training=args.metadata_training
	c.metadata_validation=args.metadata_validation
	c.additional_training=args.additional_training
	c.additional_validation=args.additional_validation
	c.feature_selection=args.feature_selection
	c.seed=args.seed
	c.n_splits=args.n_splits
	c.n_repeats=args.n_repeats
	c.verbose=args.verbose
	c.transformation=args.transformation
	c.threads=args.threads
	c.archetypes=args.archetypes

	if not os.path.exists(c.OUT):

		try:

			os.makedirs(c.OUT)

		except Exception as e:

			error('Cannot create the output folder: ' + str(e))
			sys.exit(1)

	else:

		if not os.access(os.path.abspath(c.OUT),os.W_OK):

			error('Missing write permissions on the output folder')
			sys.exit(1)
			
		elif os.listdir(os.path.abspath(c.OUT)):

			error('The output folder is not empty: specify another output folder or clean the current one')
			sys.exit(1)

	try:
		
		X_train = pd.read_csv(c.INPUT, sep='\t', index_col='ptid')
		X_train = X_train.drop(columns=[col for col in X_train.columns if "extraction_ID" in col or "diagnostics" in col])
		X_train = X_train.select_dtypes(include=['number'])
		X_train.dropna(inplace=True)
	
	except Exception as e:

		warn('ptid column not in '+ c.INPUT + " try without searching for ptid column")

		try:
			
			X_train = pd.read_csv(c.INPUT, sep='\t', index_col=None)
			X_train = X_train.drop(columns=[col for col in X_train.columns if "extraction_ID" in col or "diagnostics" in col])
			X_train = X_train.select_dtypes(include=['number'])
			X_train.dropna(inplace=True)
		
		except Exception as e:
			
			error('TSV '+ c.INPUT + ' does not exist, is not readable or is not a valid TSV')
			sys.exit(1)
	
	if c.additional_training:

		try:

			# Test dataset
			additional_training = pd.read_csv(c.additional_training, sep="\t", index_col='ptid')


		except Exception as e:

			error('TSV'+ c.additional_training + ' does not exist, is not readable or is not a valid TSV: ' + str(e))
			sys.exit(1)

		X_train=mergeDFs(X_train,additional_training)
		X_train.dropna(inplace=True)

	try:

		X_test = pd.read_csv(c.validation, sep='\t', index_col='ptid') #sampled 
		X_test = X_test.drop(columns=[col for col in X_test.columns if "extraction_ID" in col or "diagnostics" in col])
		X_test = X_test.select_dtypes(include=['number'])
		X_test.dropna(inplace=True)

	except Exception as e:

		error('TSV '+ c.validation + ' does not exist, is not readable or is not a valid TSV: ' + str(e))
		sys.exit(1)

	if c.additional_validation:

		try:

			# Test dataset
			additional_validation = pd.read_csv(c.additional_validation, sep="\t", index_col='ptid')


		except Exception as e:

			error('TSV'+ c.additional_validation + ' does not exist, is not readable or is not a valid TSV: ' + str(e))
			sys.exit(1)

		X_test=mergeDFs(X_test,additional_validation)
		X_test.dropna(inplace=True)

	if not c.verbose:
	
		log('Disable scikit-survival warnings')    
		warnings.simplefilter("ignore")
		os.environ["PYTHONWARNINGS"] = "ignore"

	else: 

		log('Enable scikit-learn warnings')

	try:

		metadata_validation = pd.read_csv(c.metadata_validation, sep='\t', index_col='ptid')

	except Exception as e:

		error('TSV '+ c.metadata_validation + ' does not exist, is not readable or is not a valid TSV: ' + str(e))
		sys.exit(1)

	y_test = format_(metadata_validation)

	try:

		metadata_training = pd.read_csv(c.metadata_training, sep="\t", index_col='ptid')

	except Exception as e:

		error('TSV '+ c.metadata_training + ' does not exist, is not readable or is not a valid TSV: ' + str(e))
		sys.exit(1) 

	if c.feature_selection:

		try:

			FS = pd.read_csv(c.feature_selection, sep='\t', index_col=0) #sampled 

		except Exception as e:

			error('TSV '+ c.feature_selection + ' does not exist, is not readable or is not a valid TSV: ' + str(e))
			sys.exit(1)

	y_train = format_(metadata_training)        

	log('Start Survival Analysis') # with ' + str(threads) + ' threads')

	log('Plot the estimated curve') # with ' + str(threads) + ' threads')  
	plotEstimatedCurve(X_train,y_train,c.OUT)

	if c.transformation:

		# Apply transformation
		X_train_T, X_test_ = transform(parser,args,c.transformation,X_train, X_test)

	else:

		X_train_T, X_test_ = X_train, X_test

	# Remove Spearman correlated features 
	el_features = []

	log('Spearman correlation')

	for i,x in enumerate(Spearmanfeatures(X_train_T)):

		if x ==False:

			el_features.append(X_train_T.columns[i])

	with open(c.OUT + '/' + 'removed.spearman.txt', 'w') as fout:

		for feature in el_features:

			fout.write('{}\n'.format(feature))

	selected_columns = X_train_T.columns[Spearmanfeatures(X_train_T)]
	
	X_train_ = X_train_T[selected_columns]

	log(''+ str(len(el_features)) + ' features were removed by Spearman coefficient')

	log('Writing training dataset after removing Spearman correlated features')
	X_train_toW = mergeDFs(X_train_,metadata_training[['OS.status','OS.time']])
	X_train_toW.to_csv(c.OUT + '/' + 'train.spearman.tsv' ,sep='\t' ,index=False)

	log('Fit a Cox model to each variable individually and record the c-index on the training set') # with ' + str(threads) + ' threads')  
	scores,coef = fit_and_score_features(X_train_.values, y_train)
	univariate = (
	pd.DataFrame({
		"feature": X_train_.columns,
		"score": scores,
		"coef": coef
	})
	.sort_values("score", ascending=False)
	.reset_index(drop=True)
	)
	univariate.to_csv(c.OUT + '/' + 'univariate.analysis.tsv' ,sep='\t' ,index=False)

	global seed
	seed = c.seed

	global cv
	cv = RepeatedStratifiedKFold(n_splits=c.n_splits, n_repeats=c.n_repeats, random_state=seed) #KFold(n_splits=c.n_splits, random_state=seed, shuffle=True) #

	global threads
	threads=c.threads

	# === Main workflow ===

	# Initialize with your original spearman-selected columns
	working_features = selected_columns

	# Track which feature set was successfully used
	successful_base_features = None
	base_feature_label = None
	X_train_working = None  # Track the working training data

	try:

		log("Fit Cox model using spearman-selected features")
		cph = CoxPHSurvivalAnalysis()
		cph_auc, cph_mean_auc, df_hr, cph_risk_scores, df_cph_summary,cph_risk_scores_train = fit_evaluate_cox(
			cph, X_train_, X_test_, y_train, y_test, working_features, "spearman", c.OUT)
		df_hr.to_csv(f"{c.OUT}/df_HR.spearman.txt", sep="\t", index=False)
		df_risk_scores=pd.DataFrame(
		{"ptid": X_test_.index,
		"risk_score": cph_risk_scores})
		df_risk_scores.to_csv(f"{c.OUT}/df_risk_scores.spearman.txt", sep="\t", index=False)

		# Track successful features
		successful_base_features = selected_columns
		base_feature_label = "spearman"
		X_train_working = X_train_[selected_columns]  # Set working data

	except Exception as e:

		warn('Unable to fit Cox proportional hazards model to the training data using spearman feature selection: '+ str(e))
		log('Attempting additional feature cleaning to resolve numerical issues')
		
		try:

			vif_features = remove_multicollinearity(X_train_[selected_columns], threshold=5.0)

			# UPDATE the working features for the next step
			working_features = vif_features			

			#subset X_train_ and export 
			X_train_vif = X_train_[working_features]
			mergeDFs(X_train_vif,metadata_training[['OS.status','OS.time']]).to_csv(c.OUT + '/' + 'train.vif_selected.tsv' ,sep='\t' ,index=False)

			log("Fit Cox model using after removing collinearity features")
			cph = CoxPHSurvivalAnalysis()
			cph_auc, cph_mean_auc, df_hr, cph_risk_scores, df_cph_summary,cph_risk_scores_train = fit_evaluate_cox(
				cph, X_train_vif, X_test_, y_train, y_test, vif_features, "vif_selected", c.OUT)
			df_hr.to_csv(f"{c.OUT}/df_HR.refined.txt", sep="\t", index=False)
			df_risk_scores=pd.DataFrame(
			{"ptid": X_test_.index,
			"risk_score": cph_risk_scores})
			df_risk_scores.to_csv(f"{c.OUT}/df_risk_scores.refined.txt", sep="\t", index=False)

			# Track successful features
			successful_base_features = vif_features
			base_feature_label = "vif_selected"
			X_train_working = X_train_vif  # Set working data

		except Exception as e2:

			error('Failed to clean features by removing multicollinearity')
			sys.exit(1)

	try:

		# 2. Multivariate feature selection
		log(f"Choosing optimal number of features using {c.search}")
		pipe = Pipeline([
			("select", SelectKBest(fit_and_score_features, k=3)),
			("model", CoxPHSurvivalAnalysis())
		])
		param_grid = {"select__k": np.arange(1, X_train_working.shape[1] + 1)}

		if c.search == "RandomizedSearchCV":

			gcv_multivariate = random_search_cox(X_train_working, y_train, pipe, param_grid, cv, threads, seed)

		else:

			gcv_multivariate = grid_search_cox(X_train_working, y_train, pipe, param_grid, cv, threads,"multivariate", c.OUT)

		pipe.set_params(**gcv_multivariate.best_params_)
		pipe.fit(X_train_working, y_train)
		transformer, final_estimator = (s[1] for s in pipe.steps)
		FS_multivariate = pd.Series(final_estimator.coef_, index=X_train_working.columns[transformer.get_support()])
		FS_multivariate.to_csv(f"{c.OUT}/multivariate.features.txt", sep="\t")

		log(f"Optimal number of multivariate features: {int(gcv_multivariate.best_params_['select__k'])}")
		cph_multivariate_auc, cph_multivariate_mean_auc, df_hr_multivariate, cph_risk_scores_multivariate, df_cph_multivariate_summary,cph_risk_scores_multivariate_train = fit_evaluate_cox(
			gcv_multivariate.best_estimator_['model'], X_train_working, X_test_, y_train, y_test, FS_multivariate.index, "multivariate", c.OUT
		)
		df_hr_multivariate.to_csv(f"{c.OUT}/df_HR.multivariate.txt", sep="\t", index=False)
		df_risk_scores_multivariate=pd.DataFrame(
		{"ptid": X_test_.index,
		"risk_score": cph_risk_scores_multivariate})
		df_risk_scores_multivariate.to_csv(f"{c.OUT}/df_risk_scores.multivariate.txt", sep="\t", index=False)

	except Exception as e:

		warn('Unable to fit Cox proportional hazards model to the training data using multivariate feature selection: ' + str(e))
	
	try:

		# 3. Penalized Cox models
		gcv_lasso, lasso_features = penalized_cox_analysis(X_train_, y_train, l1_ratio=1.0, outdir=c.OUT, label="Lasso",search=c.search)
		cph_auc_Lasso, cph_mean_auc_Lasso, df_hr_Lasso, cph_risk_scores_Lasso, df_cph_Lasso_summary,cph_risk_scores_Lasso_train = fit_evaluate_cox(
			gcv_lasso.best_estimator_, X_train_, X_test_, y_train, y_test, lasso_features.index, "Lasso", c.OUT
		)
		df_hr_Lasso.to_csv(f"{c.OUT}/df_HR.lasso.txt", sep="\t", index=False)
		df_risk_scores_lasso=pd.DataFrame(
		{"ptid": X_test_.index,
		"risk_score": cph_risk_scores_Lasso})
		df_risk_scores_lasso.to_csv(f"{c.OUT}/df_risk_scores.lasso.txt", sep="\t", index=False)


	except Exception as e:

		warn('Unable to run Lasso penalized Cox analysis: '+ str(e))
		#sys.exit(1)

	try:

		gcv_EN, EN_features = penalized_cox_analysis(X_train_, y_train, l1_ratio=0.5, outdir=c.OUT, label="ElasticNet",search=c.search)
		cph_auc_EN, cph_mean_auc_EN, df_hr_EN, cph_risk_scores_EN, df_cph_EN_summary,cph_risk_scores_EN_train = fit_evaluate_cox(
			gcv_EN.best_estimator_, X_train_, X_test_, y_train, y_test, EN_features.index, "ElasticNet", c.OUT
		)
		df_hr_EN.to_csv(f"{c.OUT}/df_HR.ElasticNet.txt", sep="\t", index=False)
		df_risk_scores_EN=pd.DataFrame(
		{"ptid": X_test_.index,
		"risk_score": cph_risk_scores_EN})
		df_risk_scores_EN.to_csv(f"{c.OUT}/df_risk_scores.ElasticNet.txt", sep="\t", index=False)

	except Exception as e:

		warn('Unable to run ElasticNet penalized Cox analysis: '+ str(e))
		#sys.exit(1)

	# -------- Define your parameters -------- #

	with open(c.json, 'r') as f:
			
		params = json.load(f)

	# -------- Define feature sets -------- #
	feature_sets = {
		base_feature_label: successful_base_features,
		"Lasso": lasso_features.index,
		"ElasticNet": EN_features.index
	}

	# -------- Run models -------- #
	RSF_results = RunModelPipeline(model_name="RSF", model=RandomSurvivalForest(n_jobs=threads, random_state=seed),
									 params=params["parameters_RF"], Xtrain=X_train_, Xtest=X_test_, ytrain=y_train, ytest=y_test, feature_sets=feature_sets, outdir=c.OUT, threads=threads,search=c.search)
	RSF_summary = RSF_results["summary"]

	GBS_results = RunModelPipeline(model_name="GBS", model=GradientBoostingSurvivalAnalysis(random_state=seed),
									 params=params["parameters_GB"], Xtrain=X_train_, Xtest=X_test_, ytrain=y_train, ytest=y_test, feature_sets=feature_sets, outdir=c.OUT, threads=threads,search=c.search)	
	GBS_summary = GBS_results["summary"]
	
	CWGB_results = RunModelPipeline(model_name="CWGB", model=ComponentwiseGradientBoostingSurvivalAnalysis(random_state=seed),
									 params=params["parameters_CWGB"], Xtrain=X_train_, Xtest=X_test_, ytrain=y_train, ytest=y_test, feature_sets=feature_sets, outdir=c.OUT, threads=threads,search=c.search)
	CWGB_summary = CWGB_results["summary"]

	SSVM_results = RunModelPipeline(model_name="SSVM", model=FastSurvivalSVM(random_state=seed),
									 params=params["parameters_SSVM"], Xtrain=X_train_, Xtest=X_test_, ytrain=y_train, ytest=y_test, feature_sets=feature_sets, outdir=c.OUT, threads=threads,search=c.search)
	SSVM_summary = SSVM_results["summary"]

	# -------- Save all summaries -------- #
	all_summaries = pd.concat([df_cph_summary, df_cph_multivariate_summary,df_cph_Lasso_summary,df_cph_EN_summary,RSF_summary, GBS_summary, CWGB_summary, SSVM_summary])
	cols_to_update = ['mean_auc', 'c_harrell', 'c_uno']
	all_summaries[cols_to_update] = all_summaries[cols_to_update].applymap(
		lambda x: 1 - x if x < 0.5 else x
	)
	all_summaries = all_summaries.round(4)
	all_summaries.to_csv(c.OUT + '/' +'model.metrics.tsv', sep='\t', index=True)

	if c.feature_selection:

		runFS(fsDF=FS, Xtrain=X_train_T, Xtest=X_test_, ytrain=y_train, ytest=y_test, outdir=c.OUT,run_aa=bool(c.archetypes),search=c.search,parameter=params)

	# --- MAIN EXECUTION BLOCK ---

	if c.archetypes:

		feature_map = {
			base_feature_label.lower(): successful_base_features, 
			"multivariate": FS_multivariate.index,
			"lasso": lasso_features.index,
			"elasticnet": EN_features.index
		}

		# Cox-PH Tasks
		cox_tasks = [
			(base_feature_label.lower(), cph_risk_scores, cph_risk_scores_train, f"CoxPH_{base_feature_label}"),
			("multivariate", cph_risk_scores_multivariate, cph_risk_scores_multivariate_train, "CoxPH_multivariate"),
			("lasso", cph_risk_scores_Lasso, cph_risk_scores_Lasso_train, "CoxPH_lasso"),
			("elasticnet", cph_risk_scores_EN, cph_risk_scores_EN_train, "CoxPH_elasticnet")
		]

		for feat_key, test_scores, train_scores, task_label in cox_tasks:
			
			current_feats = feature_map.get(feat_key.lower())
			
			if current_feats is not None and len(current_feats) > 1:
				
				log(f"[AA] Optimization: {task_label}")
				A_LL, A_LH = GetExtremeArchetypes(X_train_, current_feats, train_scores)
				opt_thresh = FindOptimalThreshold(X_train_, y_train, current_feats, A_LL, A_LH)
				ProjectAndAnalyze(X_test_, y_test, current_feats, c.OUT, test_scores, task_label, A_LL, A_LH, opt_thresh)

		# Ensemble Models
		ensemble_map = {
			'RSF': (RSF_results, "RandomSurvivalForest"),
			'GBS': (GBS_results, "GradientBoosting"),
			'CWGB': (CWGB_results, "CompWiseGB"),
			'SSVM': (SSVM_results, "SurvivalSVM")
		}
		
		# We update the subspaces list to look for our dynamic base label (spearman OR vif_selected)
		subspaces = [base_feature_label, 'Lasso', 'ElasticNet']

		for prefix, (res_obj, model_name) in ensemble_map.items():
			
			for sub in subspaces:
				
				key = f"{prefix}_{sub}"
				
				if key in res_obj["risk_tables"]:
					
					current_features = feature_map.get(sub.lower())
					
					if current_features is not None and len(current_features) > 1:
						
						log(f"[AA] Optimization: {model_name} | {key}")
						test_scores = res_obj["risk_tables"][key]
						train_scores = res_obj["risk_tables_train"][key]
						
						event_col, time_col = y_train.dtype.names[0], y_train.dtype.names[1]
						corr = pd.Series(train_scores).corr(pd.Series(y_train[time_col]), method='spearman')
						
						if corr > 0:
							log(f"[Alert] Inverting risk scores for {key} (Correlation: {corr:.2f})")
							train_scores = -train_scores
							test_scores = -test_scores

						A_LL, A_LH = GetExtremeArchetypes(X_train_, current_features, train_scores)
						opt_thresh = FindOptimalThreshold(X_train_, y_train, current_features, A_LL, A_LH)
						ProjectAndAnalyze(X_test_, y_test, current_features, c.OUT, test_scores, key, A_LL, A_LH, opt_thresh)

	log("Survival analysis done!")
	sys.exit(0)