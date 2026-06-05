import pandas as pd
import numpy as np
import os
import argparse
from argparse import HelpFormatter
import sys
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline as pipe
import math
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import os 
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,roc_auc_score
from sklearn.exceptions import ConvergenceWarning
import itertools
from scipy.stats import f
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm
from typing import Dict, Callable, List
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

class c():

	INPUT=''
	OUT=''
	algorithms=''
	metric=''
	label=''
	metadata=''
	additional=''
	threads=''
	verbose=''
	seed='',
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

def mergeDFs(df1, df2):

	'''
	Merge two pandas dataframe
	'''

	df_final = df1.merge(df2, left_index=True, right_index=True)

	return df_final

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

def MI_score(Xtrain, ytrain,func):

	'''
	Mutual Information for feature Selection
	'''
	
	steps = []
	steps.append(('vt', VarianceThreshold()))
	steps.append(('MI', SelectKBest(score_func=func, k='all')))
	pipeline = pipe(steps=steps)

	# configure to select all features
	fs = pipeline
	# learn relationship from training data
	fs.fit(Xtrain, ytrain)

	return fs

def F_score(Xtrain, ytrain,func):

	'''
	ANOVA F-test for Feature Selection
	'''
	
	steps = []
	steps.append(('vt', VarianceThreshold()))
	steps.append(('F_score', SelectKBest(score_func=func, k='all')))
	pipeline = pipe(steps=steps)

	fs = pipeline
	fs.fit(Xtrain, ytrain)

	return fs

def AnovaFScore(Xtrain,ytrain,func):

	'''
	Compute ANOVA F-scores for all features after variance thresholding.
	Returns a DataFrame with features ranked by descending F-score.
	'''

	# ANOVA F-score for feature selection
	fs = F_score(Xtrain, ytrain,func)
	X_Fscore = Xtrain[Xtrain.columns[fs['vt'].get_support()]]
	rankdictF = {'Features':[x for x in X_Fscore.columns], 'Score_ANOVA':[fs['F_score'].scores_[i] for i in range(len(fs['F_score'].scores_))]}
	rankdfF = pd.DataFrame.from_dict(rankdictF)
	rankdfF.set_index('Features', inplace=True)
	rankdfF = rankdfF.sort_values('Score_ANOVA', ascending=False)
	rankdfF.insert(0, 'rank_ANOVA', range(1, len(rankdfF) + 1))

	return rankdfF

def ANOVASelector(classif,Xtrain,ytrain,outdir):

	'''
	Select features using ANOVA F-test and plot score distributions.
	Returns a DataFrame of features passing the 90th percentile F-value threshold.
	'''

	filt = 'Score_' + classif
	num_list = []
	col_list = []

	log('ANOVA f-test')
	sorted_feature_scores=AnovaFScore(Xtrain,ytrain,f_classif)

	log('ANOVA f-test Feature Selection using F values')

	for i in range(sorted_feature_scores.shape[0]):
	   
	   num_list.append((sorted_feature_scores[filt][i]))
	   col_list.append((sorted_feature_scores.index[i]))

	dfn = 1 # Inter-level degrees of freedom(2-1)
	dfd = Xtrain.shape[0] - dfn # Intra-level degrees of freedom(2345-2)

	fig, ax = plt.subplots(1, 1)

	plt.xlim(-1,26)
	plt.ylim(0,1)
	x = np.linspace(f.ppf(0.0000000001, dfn, dfd),f.ppf(0.9999999999, dfn, dfd), 100)
	ax.plot(x, f.pdf(x, dfn, dfd), 'r-')
	ax.axvline(f.ppf(0.95, dfn, dfd), ls = "--", color = "navy")
	plt.title('Ranking features by ' + classif)
	plt.savefig(outdir + '/' + classif + '.selection.pdf')
	plt.clf()
	#print('upper 5%:', f.ppf(0.95, dfn, dfd))

	df = pd.DataFrame(num_list,index=col_list,columns=['importance'])
	DE = df[df["importance"]>f.ppf(0.90, dfn, dfd)]

	fig = plt.figure(figsize=(20,20))
	plt.bar(col_list[0:len(DE.index)],num_list[0:len(DE.index)])
	plt.xticks(rotation=90)
	plt.title('Ranking features by ' + classif)
	plt.savefig(outdir + '/' + classif + '.top.pdf')
	plt.clf()
	
	DE_=sorted_feature_scores.loc[DE.index]

	return DE_


def MIScore(Xtrain,ytrain):
	
	'''
	Mutual Information for feature Selection
	'''
	
	MI=mutual_info_classif(Xtrain,ytrain,random_state=seed)
	rankdfMI = pd.DataFrame(pd.Series(MI))
	rankdfMI.index = Xtrain.columns
	rankdfMI.columns = ['Score_MI']
	rankdfMI.sort_values('Score_MI',ascending=False, inplace=True)
	rankdfMI.insert(0, 'rank_MI', range(1, len(rankdfMI) + 1))

	return rankdfMI

def MISelector(classif,Xtrain,ytrain,outdir):

	'''
	Select features using Mutual Information and plot score distributions.
	Returns a DataFrame of top-10-percentile features ranked by MI score.
	'''

	filt = 'Score_' + classif
	num_list = []
	col_list = []

	log('Mutual Information')

	sorted_feature_scores=MIScore(Xtrain,ytrain)

	log('Mutual Information Feature Selection using MI values')

	for i in range(Xtrain.shape[1]):
	   
	   num_list.append((sorted_feature_scores[filt][i]))
	   col_list.append((sorted_feature_scores.index[i]))

	# To select indices of top 10% of scores in the "Score_MI" column
	top_percentile = 90  # change this to your desired percentile
	DE = sorted_feature_scores[sorted_feature_scores['Score_MI'] >= sorted_feature_scores['Score_MI'].quantile(top_percentile/100)]

	fig = plt.figure(figsize=(20,20))
	plt.bar(col_list[0:len(DE.index)],num_list[0:len(DE.index)])
	plt.xticks(rotation=90)
	plt.title('Ranking features by ' + classif)
	plt.savefig(outdir + '/' + classif + '.top.pdf')
	plt.clf()

	return DE

def Spearmanfeatures(df):
	
	'''
	Identify features to keep after removing highly correlated pairs (Spearman r >= 0.9).
	Returns a boolean array where True marks features to retain.
	'''

	corr = df.corr(method='spearman')
	
	columns = np.full((corr.shape[0],), True, dtype=bool)
	
	for i in range(corr.shape[0]):
			
		for j in range(i+1, corr.shape[0]):
				
			if corr.iloc[i,j] >= 0.9:
				
				if columns[j]:
						
					columns[j] = False
	
	return columns

def PlotViolinFeatures(df,outdir):

	'''
	Draw a violin plot of metric scores grouped by feature-selection parameter,
	annotated with the number of selected features. Saves to violin.pdf.
	'''

	# Violin plot with n features
	categories = df['parameter'].unique()
	plt.figure(figsize=(40, 30))
	sns.set_style("whitegrid")
	sns.violinplot(x="parameter", y="f1", data=df, color="0.8")
	v = sns.stripplot(x='parameter', y=metric, data=df, hue=df.index, dodge=True)

	# Add text above each violin
	for i, category in enumerate(categories):
		n_features = df[df['parameter'] == category]['n_features'].unique()[0]
		max_f1 = df[df['parameter'] == category][metric].max()  # Position text above the max f1 value
		plt.text(i, max_f1 + 0.02, f'n_feat: {n_features}', ha='center', va='bottom', fontsize=8, color='black')

	# Move the legend and adjust plot elements
	sns.move_legend(v, "upper left", bbox_to_anchor=(1, 1))
	plt.xticks(rotation=45)
	plt.title('Feature Profile', fontsize=40)
	plt.tight_layout()  # Adjust layout to make sure everything fits without overlap
	plt.savefig(outdir + "/" + "violin.pdf")
	plt.clf()


def plotHeatmap(df,mask,n,outdir):

	'''
	Draw an Euclidean-distance clustermap with an optional mask and save to heatmap_<n>.pdf.
	'''

	#heatmap
	fig = plt.figure(figsize=(12,12))
	r = sns.clustermap(df,metric="euclidean", mask=mask)
	plt.savefig(outdir + '/' + 'heatmap_'+ str(n) + '.pdf')
	plt.clf()

def DictPandas(df):

	'''
	Split a DataFrame into a dictionary of sub-DataFrames keyed by the unique
	values of the 'parameter' column.
	'''

	categories = df['parameter'].unique()
	
	dictA = {}

	for category in categories:

		dictA[category] = df[df['parameter'] == category]

	return dictA

def SearchPattern(pattern, text):

	'''
	Return True if the regex pattern is found anywhere in text, False otherwise.
	'''

	# Compile the pattern to a regular expression object
	compiled_pattern = re.compile(pattern)
	
	# Search for the pattern in the text
	match = compiled_pattern.search(text)
	
	# If the pattern is found, return the original text
	if match:
		
		return True

	else :

		return False

def PlotRFECV(RFECV_fit,metric,rank,estimator,outdir):

	'''
	Plot RFECV cross-validation scores vs number of selected features.
	Saves a full-range plot (<estimator>.total.pdf) and a zoomed optimal-range
	plot (<estimator>.optimal.pdf) to outdir.
	'''

	#Plot
	cv_results = pd.DataFrame(RFECV_fit.cv_results_)
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Mean test " + metric)
	plt.errorbar(
		x=cv_results["n_features"],
		y=cv_results["mean_test_score"],
		yerr=cv_results["std_test_score"],
	)
	plt.title("Recursive Feature Elimination \nwith correlated features")
	plt.savefig(outdir + "/" + estimator + ".total.pdf")
	plt.clf()

	#Plot
	cv_results = pd.DataFrame(RFECV_fit.cv_results_)
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Mean test " + metric)
	plt.errorbar(
		x=cv_results["n_features"][:len(rank)],
		y=cv_results["mean_test_score"][:len(rank)],
		yerr=cv_results["std_test_score"][:len(rank)],
	)
	plt.title("Recursive Feature Elimination \nwith correlated features")
	plt.savefig(outdir + "/" + estimator + ".optimal.pdf")
	plt.clf()



def RFECV_(Xtrain, ytrain, outdir, variant='LR'):

	'''
	Run Recursive Feature Elimination with Cross-Validation (RFECV) using a
	configurable estimator and return a ranked feature-importance DataFrame.
	'''

	# --- estimator catalogue ---------------------------------------------------
	estimator_catalogue = {
		'LR':         ('Logistic Regression',                    LogisticRegression(penalty=None, random_state=seed, n_jobs=threads)),
		'LR_L1':      ('Logistic Regression with L1 penalty',    LogisticRegression(penalty='l1', solver='liblinear', random_state=seed, n_jobs=threads)),
		'LR_L2':      ('Logistic Regression with L2 penalty',    LogisticRegression(penalty='l2', random_state=seed, n_jobs=threads)),
		'LR_EN':      ('Logistic Regression with ElasticNet',    LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=seed, n_jobs=threads)),
		'RF':         ('Random Forest',                          RandomForestClassifier(random_state=seed, n_jobs=threads)),
		'Perceptron': ('Perceptron',                             Perceptron(random_state=seed, n_jobs=threads)),
		'GB':         ('Gradient Boost',                         GradientBoostingClassifier(random_state=seed)),
		'SVM':        ('Support Vector Machine',                 SVC(kernel='linear', probability=False, max_iter=1000, random_state=seed)),
	}

	if variant not in estimator_catalogue:
		error(f'Unknown RFECV variant "{variant}". Choose from: {", ".join(estimator_catalogue)}')
		sys.exit(1)

	label, estimator = estimator_catalogue[variant]
	score_col  = f'Score_RFECV_{variant}'
	rank_col   = f'rank_RFECV_{variant}'
	plot_label = f'RFECV_{variant}'

	log(f'RFECV using {label} estimator and {metric} scorer')

	# --- fit RFECV -------------------------------------------------------------
	rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring=metric,
	              n_jobs=threads, min_features_to_select=1)
	rfecv.fit(Xtrain, ytrain)

	ranked_features = list(Xtrain.columns[rfecv.support_])

	# --- extract feature weights -----------------------------------------------
	# Tree-based estimators expose feature_importances_; linear ones expose coef_
	if variant in ('RF', 'GB'):
		weights = rfecv.estimator_.feature_importances_.tolist()
	else:
		weights = np.absolute(rfecv.estimator_.coef_).tolist()[0]

	# --- build ranking DataFrame -----------------------------------------------
	rankdf = pd.DataFrame({'Features': ranked_features, score_col: weights})
	rankdf.set_index('Features', inplace=True)
	rankdf.sort_values(score_col, ascending=False, inplace=True)
	rankdf.insert(0, rank_col, range(1, len(rankdf) + 1))

	PlotRFECV(rfecv, metric, ranked_features, plot_label, outdir)

	return rankdf


def plot_coefficients(coefs, n_highlight, reg, outdir):

	'''
	Plot feature coefficients across a range of alpha values on a semi-log scale,
	labelling the top n_highlight coefficients at the smallest alpha.
	Saves the figure to coefficients.<reg>.pdf in outdir.
	'''

	_, ax = plt.subplots(figsize=(9, 6))
	alphas = coefs.columns.to_numpy()
	
	for row in coefs.itertuples():
		
		ax.semilogx(alphas, np.array(row[1:]), ".-", label=row.Index)

	alpha_min = alphas.min()
	top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
	
	for name in top_coefs.index:
		
		coef = coefs.loc[name, alpha_min]
		plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

	ax.yaxis.set_label_position("right")
	ax.yaxis.tick_right()
	ax.grid(True)
	ax.set_xlabel("alpha")
	ax.set_ylabel("coefficient")
	plt.savefig(outdir + '/' + 'coefficients.' + reg + '.pdf')

def GetCoeff(reg_name, Xtrain, ytrain, alphas, outdir):

	'''
	Fit a regularised classifier (Lasso, Ridge, or ElasticNet) across a sequence
	of alpha values and collect the resulting coefficients.
	Calls plot_coefficients to visualise the coefficient paths and saves to outdir.
	'''

	reg_classes = {
		'Lasso': LogisticRegression(penalty='l1', solver='liblinear', random_state=seed),
		'ElasticNet': LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=0.5,random_state=seed)
	}
	
	reg_class = reg_classes.get(reg_name)
	
	if reg_class is None:
	
		raise ValueError(f"Unsupported regression type: {reg_name}")
	
	coefficients = {}
	
	for alpha in alphas:
	
		reg_class.set_params(C=alpha)
		reg_class.fit(Xtrain, ytrain)
		key = round(alpha, 5)
		coefficients[key] = reg_class.coef_[0]
	
	coefficients = pd.DataFrame.from_dict(coefficients).rename_axis(index="feature", columns="alpha").set_index(Xtrain.columns)
	
	plot_coefficients(coefficients, 0, reg_name, outdir)

def GetResultsGridReg(grid, Xtrain, reg, outdir):

	'''
	Summarise GridSearchCV results for a regularised classifier.
	Plots mean CV score vs alpha (grid_results.<reg>.pdf) and, when non-zero
	coefficients are found, a bar chart of those coefficients (nonzero.<reg>.pdf).
	Returns a DataFrame of non-zero coefficients, or an all-NaN DataFrame if none
	are found.
	'''

	cv_results = pd.DataFrame(grid.cv_results_)
	alphas = cv_results.param_C.to_numpy()
	mean = cv_results.mean_test_score.to_numpy()
	std = cv_results.std_test_score.to_numpy()


	fig, ax = plt.subplots(figsize=(9, 6))
	ax.plot(alphas, mean)
	ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
	ax.set_xscale("log")
	ax.set_ylabel("f1")
	ax.set_xlabel("alpha")
	ax.axvline(grid.best_params_['C'])
	ax.axhline(0.5, color="grey", linestyle="--")
	ax.grid(True)
	plt.savefig(outdir + '/' + 'grid_results.' + reg + '.pdf')
	plt.clf()

	#plot Non zero
	best_model = grid.best_estimator_ #gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
	colname = 'coefficients_' + reg
	best_coefs = pd.DataFrame(best_model.coef_[0], index=Xtrain.columns, columns=[colname])

	non_zero = np.sum(best_coefs.iloc[:, 0] != 0)

	if non_zero > 0:

		log('Number of non-zero coefficients: ' + str(non_zero) + ' for ' + reg + ' regression')	

		non_zero_coefs = best_coefs.query("coefficients_" + reg + "!= 0")
		coef_order = non_zero_coefs.abs().sort_values("coefficients_" + reg).index

		_, ax = plt.subplots(figsize=(35, 20))
		non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
		ax.set_xlabel("coefficients_" + reg)
		ax.grid(True)
		plt.savefig(outdir + '/' + 'nonzero.' + reg + '.pdf')
		plt.clf()

		return non_zero_coefs

	else:

		NA_coefs = pd.DataFrame(np.nan, index=Xtrain.columns, columns=[colname])

		return NA_coefs


def GridReg(reg_name, Xtrain, ytrain, outdir):

	'''
	Run a GridSearchCV over a log-spaced range of alpha values for a regularised
	classifier (Lasso, or ElasticNet).
	If the optimal model yields no non-zero coefficients, the alpha range is
	automatically widened towards weaker regularisation and the search is repeated.
	Returns a ranked DataFrame of non-zero feature coefficients.
	'''

	reg_classes = {
		'Lasso': LogisticRegression(penalty='l1', solver='liblinear', random_state=seed),
		'ElasticNet': LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=0.5,random_state=seed)
	}
		
	reg_class = reg_classes.get(reg_name)
	
	if reg_class is None:
	
		raise ValueError(f"Unsupported regression type: {reg_name}")

	alphas = 10.0 ** np.linspace(-5, 5, 50)

	log('Draw features contribution along alpha values for ' + reg_name + ' regression')	
	GetCoeff(reg_name, Xtrain, ytrain, alphas, outdir)

	log('GridSearchCV to find the optimal alpha value for ' + reg_name + ' regression')	
	gcv = GridSearchCV(reg_class, param_grid={"C": [float(v) for v in alphas]}, cv=cv,n_jobs=threads, scoring=metric)
	gcv.fit(Xtrain, ytrain)

	colname='coefficients_' + reg_name
	df_res = GetResultsGridReg(gcv, Xtrain,reg_name, outdir)

	if df_res[colname].notna().all():

		results = df_res.loc[df_res[colname].abs().sort_values().index] #plotGridReg(gcv, reg_name, outdir).abs().sort_values("coefficient")
		results.insert(0, 'rank_' + reg_name, range(1, len(results) + 1))

	else:

		alphas = 10.0 ** np.linspace(0.5, 3, 50)

		warn('Number of non-zero coefficients for ' + reg_name + ' regression are 0 increase alpha possible values to weaker regularization')
		log('Draw features contribution along alpha values for ' + reg_name + ' regression with increasing alpha interval')	
		GetCoeff(reg_name, Xtrain, ytrain, alphas, outdir)
		#mod = reg_class(random_state=seed)

		log('GridSearchCV to find the optimal alpha value for ' + reg_name + ' regression')	
		gcv = GridSearchCV(reg_class, param_grid={"C": [float(v) for v in alphas]}, cv=cv,n_jobs=threads, scoring=metric)
		gcv.fit(Xtrain, ytrain)

		colname='coefficients_' + reg_name
		df_res = GetResultsGridReg(gcv, Xtrain,reg_name, outdir)		

		if df_res[colname].notna().all():

			results = df_res.loc[df_res[colname].abs().sort_values().index] #plotGridReg(gcv, reg_name, outdir).abs().sort_values("coefficient")
			results.insert(0, 'rank_' + reg_name, range(1, len(results) + 1))

		else:

			return df_res

	return results

def EvalML(Xtrain, ytrain, outdir):

	'''
	Evaluate a panel of ML classifiers using repeated stratified k-fold
	cross-validation across accuracy, precision, recall, F1, and ROC-AUC.
	Results are written incrementally to ML_all_features.tsv in outdir.
	'''

	#Evaluate the model in terms of accuracy, precision, recall and f1score with all the features
	df_ = pd.DataFrame(0, index=indexes, columns= estimators)

	for i,alg in enumerate(ML_algorithm):
				
		for es in estimators:

			try:
				n_scores = cross_val_score(alg, Xtrain, ytrain, scoring=es, cv=cv, n_jobs=threads, error_score='raise')
				log('Cross Validation results for ' + str(alg) + ' considering ' + str(es) +  ': ' + str(np.mean(n_scores)) + '±' + str(np.std(n_scores)))
				df_.loc[indexes[i], es] = "{:n}±{:n}".format(np.mean(n_scores), np.std(n_scores))

			except Exception as e:
							
				warn('Cross-validation failed for ' + str(alg) +  ' with ' + str(es) + ' : ' + str(e))
				pass

		df_.to_csv(outdir + '/' + 'ML_all_features.tsv', sep='\t', index=True)



def get_cluster_features(data, cluster_labels, n_features):

	'''
	For each agglomerative cluster, select the top n_features most discriminative
	features by absolute Cohen's d effect size, ensuring no feature is assigned to
	more than one cluster.
	Returns a tuple of (cluster_features dict, all_cohens_d dict).
	'''

	cluster_features = {}
	used_features = set()  # Keep track of features already selected
	
	all_cohens_d = {}
	
	for cluster in np.unique(cluster_labels):
		
		cluster_data = data[data['Cluster'] == cluster]
		other_data = data[data['Cluster'] != cluster]
		
		# Calculate mean differences between this cluster and others
		cluster_means = cluster_data.drop('Cluster', axis=1).mean()
		other_means = other_data.drop('Cluster', axis=1).mean()
		
		# Calculate standard deviations
		cluster_stds = cluster_data.drop('Cluster', axis=1).std()
		other_stds = other_data.drop('Cluster', axis=1).std()
		
		# Calculate effect size (Cohen's d)
		cohens_d = (cluster_means - other_means) / np.sqrt((cluster_stds**2 + other_stds**2) / 2)
		all_cohens_d[f'Cluster_{cluster}'] = cohens_d
	
	# Select features for each cluster
	for cluster in np.unique(cluster_labels):
		
		cohens_d = all_cohens_d[f'Cluster_{cluster}']
		
		# Get sorted features by absolute effect size
		sorted_features = cohens_d.abs().sort_values(ascending=False)
		
		# Select top n features that haven't been used yet
		selected_features = []
		
		for feature in sorted_features.index:
			
			if feature not in used_features:
				
				selected_features.append(feature)
				used_features.add(feature)
				
				if len(selected_features) == n_features:
					
					break
		
		# If we couldn't find enough unused features, take the best remaining ones
		if len(selected_features) < n_features:
			
			remaining_features = [f for f in sorted_features.index if f not in selected_features]
			additional_features = remaining_features[:n_features - len(selected_features)]
			selected_features.extend(additional_features)
		
		cluster_features[f'Cluster_{cluster}'] = selected_features
	
	return cluster_features, all_cohens_d


def plot_cluster_features(important_features, cohens_d, n_features, outdir):

	'''
	Draw one horizontal bar chart per cluster showing the absolute Cohen's d
	effect size of its top discriminative features.
	Saves the figure to AgglomerativeClustering.top.pdf in outdir.
	'''

	n_clusters = len(important_features)
	fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 5*n_clusters))
	
	if n_clusters == 1:
		
		axes = [axes]
	
	for i, (cluster, info) in enumerate(important_features.items()):
		
		#features = info['features'][:n_features]
		#effect_sizes = info['effect_sizes'][:n_features]
		
		features = info
		effect_sizes=cohens_d[cluster].loc[features]
		
		# Create bar plot
		sns.barplot(x=np.abs(effect_sizes), y=features, ax=axes[i])
		axes[i].set_title(f'{cluster} - Top Features by Effect Size')
		axes[i].set_xlabel('Absolute Effect Size')
		axes[i].set_ylabel('Features')
	
	plt.tight_layout()
	plt.savefig(outdir + '/' + 'AgglomerativeClustering.top.pdf')
	plt.clf()


# Optionally, create a heatmap of feature distributions across clusters
def plot_cluster_heatmap(df, cluster_labels, features, outdir):

	'''
	Plot a heatmap of per-cluster mean values for the selected features,
	providing a visual summary of how feature distributions differ across clusters.
	Saves the figure to AgglomerativeClustering.heatmap.distribution.pdf in outdir.
	'''

	cluster_means = pd.DataFrame()
	
	for cluster in np.unique(cluster_labels):
		
		cluster_data = df[df['Cluster'] == cluster]
		cluster_means[f'Cluster_{cluster}'] = cluster_data[features].mean()
	
	# Create heatmap
	plt.figure(figsize=(15, 12))
	sns.heatmap(cluster_means, annot=True, cmap='RdYlBu_r', center=0)
	plt.title('Feature Distributions Across Clusters')
	plt.savefig(outdir + '/' + 'AgglomerativeClustering.heatmap.distribution.pdf')
	plt.clf()

def silhouette(df, outdir):

	''' 
	Perform Silhouette Analysis

	'''
	n_samples = df.shape[0]
	# Ensure we don't try to create more clusters than samples - 1
	potential_clusters = [2, 3, 4, 5, 6]
	range_n_clusters = [n for n in potential_clusters if n < n_samples]	

	#range_n_clusters = [2,3,4,5,6]
	outdict={}
	fig,axs = plt.subplots(len(range_n_clusters),2)
	fig.set_size_inches(10,30)
	
	for i,n_clusters in enumerate(range_n_clusters):

		axs[i,0].set_xlim([-1, 1])
		axs[i,0].set_ylim([0, len(df) + (n_clusters + 1) * 10])
		clusterer = AgglomerativeClustering(n_clusters=n_clusters,metric='euclidean', linkage='ward')
		cluster_labels = clusterer.fit_predict(df)
		silhouette_avg = silhouette_score(df, cluster_labels)
		log('For n_clusters = ' + str(n_clusters) + ' the average silhouette_score is : ' + str(silhouette_avg))
		sample_silhouette_values = silhouette_samples(df, cluster_labels)
		outdict[str(n_clusters)] = float(silhouette_avg)
		y_lower = 10
		
		for l in range(n_clusters):

			ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == l]
			ith_cluster_silhouette_values.sort()
			size_cluster_i = ith_cluster_silhouette_values.shape[0]
			y_upper = y_lower + size_cluster_i
			color = cm.nipy_spectral(float(l) / n_clusters)
			axs[i,0].fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_values,facecolor=color, edgecolor=color, alpha=0.7)
			y_lower = y_upper + 10 

		axs[i,0].set_title('Silhouette plot with number of clusters = ' + str(n_clusters))
		axs[i,0].set_xlabel('Silhouette coefficient value')
		axs[i,0].set_ylabel('Clusters')
		axs[i,0].axvline(x=silhouette_avg, color="red", linestyle="--")
		axs[i,0].set_yticks([])
		axs[i,0].set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

		colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
		axs[i,1].scatter(df.values[:, 0], df.values[:, 1], marker='.', s=30, lw=0, alpha=0.7,c=colors, edgecolor='k')
		axs[i,1].set_title('Cluster plot with number of clusters = ' + str(n_clusters))
		axs[i,1].set_xlabel('Feature space (1st feature)')
		axs[i,1].set_ylabel('Feature space (2nd feature)')
		plt.rc('xtick',labelsize=8)
		plt.rc('ytick',labelsize=8)

	plt.tight_layout()
	plt.savefig(outdir + '/' + 'AgglomerativeClustering.silhouette.pdf')

	reslist = sorted(outdict.items(), key=lambda item: item[1], reverse=True)
	
	return reslist


def ClusteringSelector(Xtrain,outdir):

	'''
	Select features via Agglomerative Clustering: find the optimal cluster count
	with silhouette analysis, then rank features by Cohen's d effect size.
	Returns a DataFrame with a rank_AgglomerativeClustering column.
	'''

	log('Agglomerative Clustering Feature Selection')

	try:
		log('Calculating Silhouette score to find optimal number of cluster')
		score = silhouette(Xtrain,outdir)

	except Exception as e:

		error('Silhouette failed :' + str(e))
		sys.exit(1)

	# Perform Agglomerative Clustering
	log('The optimal number of clusters selected by silhoutte score is ' + str(score[0][0]))
	n_clusters = int(score[0][0])  # Set your desired number of clusters
	clustering = AgglomerativeClustering(n_clusters=n_clusters,metric='euclidean', linkage='ward')
	cluster_labels = clustering.fit_predict(Xtrain)
	
	# Add cluster labels to the original dataframe
	Xtrain_copy = Xtrain.copy()
	Xtrain_copy['Cluster'] = cluster_labels

	# Get important features for each cluster
	important_features, cohens_d = get_cluster_features(Xtrain_copy, cluster_labels,5)
	
	# Plot the results
	log('Plot the most relevant feaures of each cluster')
	plot_cluster_features(important_features,cohens_d,5,outdir)

	# Get unique features across all clusters
	all_important_features = set()

	for info in important_features.values():
		
		all_important_features.update(info)

	# Plot heatmap
	log('Plot the distribution of the features in the cluster')
	plot_cluster_heatmap(Xtrain_copy, cluster_labels, list(all_important_features),outdir)

	dfs = []

	for k,v in important_features.items():

		dfs.append(cohens_d[k].loc[v])

	newdf = pd.concat(dfs, axis=1)

	newdf['rank_AgglomerativeClustering'] = range(1, len(newdf.index) + 1)

	return newdf.iloc[:, [len(newdf.columns)-1]]

@dataclass
class AlgorithmMapper:
		
	def __init__(self,algorithms: List[str], Xtrain,ytrain,outdir):		
		self.algorithm_dict = self._create_algorithm_dict(algorithms,Xtrain,ytrain,outdir)
	
	def _create_algorithm_dict(self,algorithms: List[str], Xtrain,ytrain,outdir) -> Dict[str, BaseEstimator]:

		algorithm_dict = {}

		for name in algorithms:

			if name == 'ANOVA':

				algorithm_dict['ANOVA'] = ANOVASelector('ANOVA', Xtrain, ytrain, outdir)

			elif name == 'MI':

				algorithm_dict['MI'] = MISelector('MI', Xtrain, ytrain, outdir)

			elif name == 'RFECV_LR':

				algorithm_dict['RFECV_LR'] = RFECV_(Xtrain, ytrain, outdir, variant='LR')

			elif name == 'RFECV_LR_L1':

				algorithm_dict['RFECV_LR_L1'] = RFECV_(Xtrain, ytrain, outdir, variant='LR_L1')
			
			elif name == 'RFECV_LR_L2':

				algorithm_dict['RFECV_LR_L2'] = RFECV_(Xtrain, ytrain, outdir, variant='LR_L2')
			
			elif name == 'RFECV_LR_EN':

				algorithm_dict['RFECV_LR_EN'] = RFECV_(Xtrain, ytrain, outdir, variant='LR_EN')
			
			elif name == 'RFECV_RF':

				algorithm_dict['RFECV_RF'] = RFECV_(Xtrain, ytrain, outdir, variant='RF')
			
			elif name == 'RFECV_GB':

				algorithm_dict['RFECV_GB'] = RFECV_(Xtrain, ytrain, outdir, variant='GB')
			
			elif name == 'RFECV_Perceptron':

				algorithm_dict['RFECV_Perceptron'] = RFECV_(Xtrain, ytrain, outdir, variant='Perceptron')
			
			elif name == 'RFECV_SVM':

				algorithm_dict['RFECV_SVM'] = RFECV_(Xtrain, ytrain, outdir, variant='SVM')
			
			elif name == 'Agglomerative':

				algorithm_dict['Agglomerative'] = ClusteringSelector(Xtrain,outdir)

			elif name == 'Lasso':

				algorithm_dict['Lasso'] = GridReg('Lasso',Xtrain,ytrain,outdir)

			elif name == 'ElasticNet':

				algorithm_dict['ElasticNet'] = GridReg('ElasticNet',Xtrain,ytrain,outdir)

		return algorithm_dict
	
	def get_estimators(self, algorithm_names: List[str]) -> List[BaseEstimator]:
		"""Returns a list of estimator instances for the specified algorithm names"""
		return [self.algorithm_dict[name] for name in algorithm_names]
	
	def get_all_estimators(self) -> List[BaseEstimator]:
		"""Returns a list of all available estimator instances"""
		return list(self.algorithm_dict.values())
	
	def get_algorithm_names(self) -> List[str]:
		"""Returns a list of all available algorithm names"""
		return list(self.algorithm_dict.keys())

def run(parser,args):

	c.INPUT= args.input
	c.OUT=args.output
	c.algorithms=args.algorithms
	c.label=args.label
	c.additional=args.additional
	c.metadata=args.metadata
	c.metric=args.metric
	c.threads=args.threads
	c.seed=args.seed
	c.verbose=args.verbose
	c.n_splits=args.n_splits
	c.n_repeats=args.n_repeats

	global cv
	cv = RepeatedStratifiedKFold(n_splits=c.n_splits, n_repeats=c.n_repeats, random_state=c.seed)

	global metric
	metric=c.metric
	
	global threads
	threads=c.threads

	global seed 
	seed = c.seed
	
	#define algorithm names	
	algorithm_names = ['ANOVA','MI','RFECV_LR','RFECV_LR_L1','RFECV_LR_L2','RFECV_LR_EN', 'RFECV_Perceptron', 'RFECV_RF','RFECV_GB','RFECV_SVM', 'Agglomerative','Lasso','ElasticNet','all']

	if not all(item in algorithm_names for item in c.algorithms[0].split(',')):

		not_in = str(', '.join(f"{item}" for item in c.algorithms[0].split(',')))
		error('argument -a/--algorithm: invalid choice: ' + not_in + ' choose from ANOVA, MI, RFECV_LR, RFECV_LR_L1, RFECV_LR_L2, RFECV_LR_EN, RFECV_Perceptron, RFECV_RF, RFECV_GB, RFECV_SVC, Agglomerative, Lasso, ElasticNet, all or a combination of valid arguments')
		sys.exit(1)

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
		
		training = pd.read_csv(c.INPUT, sep='\t', index_col='ptid')
		training = pd.read_csv(c.INPUT, sep='\t', index_col=None)
		training = training.drop(columns=[col for col in training.columns if "extraction_ID" in col or "diagnostics" in col])
		training = training.select_dtypes(include=['number'])

	except ValueError:

		try:
			training = pd.read_csv(c.INPUT, sep='\t', index_col=None)
			training = training.drop(columns=[col for col in training.columns if "extraction_ID" in col or "diagnostics" in col])
			training = training.select_dtypes(include=['number'])
			training.dropna(inplace=True)

		except Exception as e:
			
			error('TSV ' + c.INPUT + ' does not exist, is not readable or is not a valid TSV')
			sys.exit(1)
	
	except Exception as e:
		
		now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
		error('TSV ' + c.INPUT + ' does not exist, is not readable or is not a valid TSV')
		sys.exit(1)

	if not c.verbose:
	
		log('Disable scikit-learn warnings')
		warnings.simplefilter("ignore")
		os.environ["PYTHONWARNINGS"] = "ignore"

	else: 

		log('Enable scikit-learn warnings')	

	if not c.metadata:

		#drop label column
		X_train = training.drop(columns = c.label)
		y_train = training[c.label]

	else:

		try:

			metadata = pd.read_csv(c.metadata, sep='\t', index_col='ptid')

		except:

			error('TSV ' + c.metadata + ' does not exist, is not readable or is not a valid TSV')
			sys.exit(1)

		#drop label column 
		training = mergeDFs(training, metadata[c.label])
		training.dropna(inplace=True)
		X_train = training.drop(columns = c.label)
		y_train = training[c.label]

	if c.additional:

		try:

			# Test dataset
			additional = pd.read_csv(c.additional, sep="\t", index_col='ptid')


		except:

			error('TSV ' + c.additional + ' does not exist, is not readable or is not a valid TSV')
			sys.exit(1)

		training=mergeDFs(training,additional)
		training.dropna(inplace=True)
		X_train = training.drop(columns = c.label)
		y_train = training[c.label]

	#Feature Selection
	log('Start feature Selection with ' + str(c.seed) + ' seed, ' + str(', '.join(f"{item}" for item in c.algorithms[0].split(','))) + ' parameter and ' + str(c.threads) + ' threads')
	log('Correlation Heatmap')

	#Correlation analysis remove one of two features that have a correlation higher than 0.9 (pearson coefficient)
	labels = X_train.columns 
	fig = plt.figure(figsize=(12,12))
	r = sns.heatmap(X_train.corr(method='spearman'))
	r.set_xticks(range(0,len(labels)))
	r.set_yticks(range(0,len(labels)))
	#r.set_yticklabels(labels)
	#r.set_xticklabels(labels)
	r.set_title("Heatmap of data features")
	plt.savefig(c.OUT + '/' + 'Correlation_matrix.pdf')
	plt.clf()

	corr = X_train.corr()

	el_features = []

	log('Spearman correlation')

	for i,x in enumerate(Spearmanfeatures(X_train)):

		if x ==False:

			el_features.append(X_train.columns[i])

	with open(c.OUT + '/' + 'removed.spearman.txt', 'w') as fout:

		for feature in el_features:

			fout.write('{}\n'.format(feature))

	selected_columns = X_train.columns[Spearmanfeatures(X_train)]
	X_train = X_train[selected_columns]

	log(str(len(el_features)) + ' features were removed by Spearman coefficient')

	log('Writing training dataset after removing Spearman correlated features')
	X_train_ = mergeDFs(X_train,y_train)
	X_train_.to_csv(c.OUT + '/' + 'train.sampled.filtered.tsv' ,sep='\t' ,index=False)

	if c.algorithms[0] == 'all':

		algorithm_mapper = AlgorithmMapper(algorithm_names, X_train, y_train, c.OUT)

	else:

		algorithm_names = c.algorithms[0].split(',')

		if len(algorithm_names) ==1 and algorithm_names[0] == 'all':

			#algorithm_names = ['ANOVA','MI','RFECV_LR','RFECV_LR_L1','RFECV_LR_L2','RFECV_LR_EN', 'RFECV_Perceptron', 'RFECV_RF','RFECV_GB','RFECV_SVM', 'Agglomerative','all']
			algorithm_mapper = AlgorithmMapper(algorithm_names, X_train, y_train, c.OUT)

		else:

			algorithm_mapper = AlgorithmMapper(algorithm_names, X_train, y_train, c.OUT)

	#concat dataframes with features ranking
	FS=pd.concat(algorithm_mapper.algorithm_dict.values(), axis=1)

	# Evaluatiuon of the ML algorithm with all the features
	global ML_algorithm

	ML_algorithm = [
    RandomForestClassifier(random_state=seed),
    LogisticRegression(penalty='l2', random_state=seed),
    LogisticRegression(solver='saga', l1_ratio=0.5, penalty='elasticnet', random_state=seed),
    LogisticRegression(penalty='l1', solver='liblinear', random_state=seed),
    SVC(kernel='linear',  max_iter=100, random_state=seed),
    SVC(kernel='rbf',     max_iter=100, random_state=seed),
    SVC(kernel='sigmoid', max_iter=100, random_state=seed),
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=3),
    DecisionTreeClassifier(random_state=seed),
    Perceptron(random_state=seed),
    GradientBoostingClassifier(random_state=seed)
]

	global indexes
	indexes = ['RandomForestClassifier', 'LogisticRegression_l2','LogisticRegression_elasticnet','LogisticRegression_l1','SVC_linear', 'SVC_rbf','SVC_sigmoid','GaussianNB', 'KNeighborsClassifier', 'DecisionTreeClassifier' , 'Perceptron', 'GradientBoost']

	global estimators 
	estimators = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

	log('Evaluate Machine Learning Classificators using all the Spearman selected features')

	EvalML(X_train,y_train,c.OUT)

	## parameters to filter 

	FS_rank = FS.filter(like="rank_")
	log('Writing seleted features by ' + str(', '.join(f"{item}" for item in c.algorithms[0].split(','))))
	FS_rank.to_csv(c.OUT + '/' + 'FS.results.tsv', sep='\t', index=True,na_rep="NA")  
	log('Done')
	sys.exit(0)