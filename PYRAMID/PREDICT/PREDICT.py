import argparse
from argparse import HelpFormatter
import sys
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib
import os 
import re
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,confusion_matrix,precision_score,recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
import warnings
from sklearn.utils import shuffle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold


class c():

	INPUT=''
	validation=''
	OUT=''
	label=''
	feature_selection=''
	tunedir=''
	additional_training=''
	additional_validation=''
	metadata_validation=''
	metadata_training=''
	permutation=False
	n_perm=''
	verbose=''
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

def mergeDFs(df1, df2):

	'''
	Inner-join two DataFrames on their indices and return the merged result.

	Parameters
	----------
	df1 : pd.DataFrame
	df2 : pd.DataFrame
	'''

	df_final = df1.merge(df2, left_index=True, right_index=True)

	return df_final

def SearchPattern(pattern, text):

	'''
	Return True if the regex pattern matches anywhere in text, False otherwise.
	'''

	# Compile the pattern to a regular expression object
	compiled_pattern = re.compile(pattern)
	
	# Search for the pattern in the text
	match = compiled_pattern.search(text)
	
	if match:
		
		return True

	else :

		return False

def permutation_test(Xtrain, Xtest, ytest, ytrain, model, FSS, metric=f1_score, n_permutations=100, random_state=None):

	'''
	Assess model significance via a permutation test.
	Fits the model on true labels, records the original metric score, then
	repeats fitting on randomly shuffled labels n_permutations times.
	Returns the original score, the array of permuted scores, and the
	empirical p-value (proportion of permuted scores >= original score).
	'''

	# Train the model on the true labels and calculate the performance metric
	model.fit(Xtrain[FSS], ytrain)
	y_pred = model.predict(Xtest[FSS])
	original_score = metric(ytest, y_pred)
	
	# Initialize an array to store the scores of the permuted datasets
	permuted_scores = np.zeros(n_permutations)
	
	for i in range(n_permutations):
		
		# Randomize (shuffle) the labels
		y_train_permuted = shuffle(ytrain, random_state=random_state)
		
		# Train the model on the permuted labels
		model.fit(Xtrain[FSS], y_train_permuted)
		y_pred_permuted = model.predict(Xtest[FSS])
		
		# Calculate the performance metric for the permuted data
		permuted_scores[i] = metric(ytest, y_pred_permuted)
	
	# Calculate the p-value: proportion of permuted scores that are better or equal to the original score
	p_value = np.mean(permuted_scores >= original_score)
	
	return original_score, permuted_scores, p_value

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


def plot_ROC_curve(pos_probs, ytest, clf, outdir):

	'''
	Plot the ROC curve for a classifier against a no-skill baseline and
	save the figure to <clf>_roc_curve.pdf in outdir.
	'''

	# plot no skill roc curve
	plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
	# calculate roc curve for model
	fpr, tpr, _ = roc_curve(ytest, pos_probs)
	# plot model roc curve
	plt.plot(fpr, tpr, marker='.', label='Logistic')
	# axis labels
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	# show the legend
	plt.legend()

	plt.savefig(outdir + "/" + clf + "_roc_curve.pdf")
	plt.clf()


def plot_PR_curve(pos_probs, ytest, ytrain, clf, outdir):

	'''
	Plot the precision-recall curve for a classifier against a no-skill
	baseline (positive class prevalence) and save the figure to
	<clf>_pr_curve.pdf in outdir.
	'''

	# calculate the no skill line as the proportion of the positive class
	no_skill = len(ytrain[ytrain==1]) / len(ytrain)
	# plot the no skill precision-recall curve
	plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	# calculate model precision-recall curve
	precision, recall, _ = precision_recall_curve(ytest, pos_probs)
	# plot the model precision-recall curve
	plt.plot(recall, precision, marker='.', label='Logistic')
	# axis labels
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	# show the legend
	plt.legend()

	plt.savefig(outdir + "/" + clf + "_pr_curve.pdf")
	plt.clf()

def Prediction(Xtrain, Xtest, ytest, ytrain, pattern, FSS, permutation, tunedir, outdir):

	'''
	Load all .sav model files matching pattern from tunedir, run predictions
	on Xtest using FSS features, and compute a full set of classification
	metrics (accuracy, precision, recall, F1, AP, ROC-AUC, PR-AUC, log-loss,
	MAE, MSE, predicted probabilities). Optionally runs a permutation test.
	Saves ROC curve, PR curve, confusion matrix PDFs, and a TSV metrics table
	per pattern to outdir. Returns a dict mapping filename to metric values.
	'''

	metrics_ = ['accuracy', 'pr', 're', 'f1', 'AP', 'roc_auc', 'pr_re_auc','log_loss', 'MAE', 'MSE', 'prob']
	resdict = {}

	sav_files = [f for f in os.listdir(tunedir) if f.endswith(pattern)]

	if not sav_files:

		error('No .sav files found in ' + str(tunedir) + ' using ' + str(pattern.split('_alg.sav')[0]))

	for file in sav_files:
		
		log('Prediction using ' + str(file) + ' algorithm and ' + str(len(FSS)) + ' features selected by ' + str(pattern))

		name = file.split('.sav')[0]
		clf_file = os.path.join(tunedir, file)

		if not os.path.exists(clf_file):

			error(str(file) + ' does not exist, is not readable or is not a valid .sav file')
			sys.exit(1)

		try:

			clf = joblib.load(clf_file)

		except Exception as e:

			error(str(file) + ' does not exist, is not readable or is not a valid .sav file ' + str(e))
			continue
			sys.exit(1)

		try:

			log('Fitting ' + str(file) + ' and ' + str(len(FSS)) + ' features selected by ' + str(pattern.split('.sav')[0]))
			y_pred = clf.predict(Xtest[FSS])

		except Exception as e:

			error('Unable to fit ' + str(file) + ': ' + str(e))
			sys.exit(1)

		try:

			log('Calculate positive observation probability for ' + str(file) + ' with predict_proba and ' + str(len(FSS)) + ' features selected by ' + str(pattern.split('.sav')[0]))
			yhat = clf.predict_proba(Xtest[FSS])[:, 1]
							
		except Exception as e:

			warn('Unable to calculate positive observation probability for ' + str(file) + ' with predict_proba and ' + str(len(FSS)) + ' features selected by ' + str(pattern.split('.sav')[0]) + ': ' + str(e))
			log('Calculate positive observation probability for ' + str(file) + ' wrapping in CalibratedClassifierCV ' + str(len(FSS)) + ' features selected by ' + str(pattern.split('.sav')[0]))
			base_clf = clf
			cal_clf = CalibratedClassifierCV(base_clf, cv=cv)
			cal_clf.fit(Xtrain[FSS], ytrain)
			yhat = cal_clf.predict_proba(Xtest[FSS])[:, 1]

		log('Calculating classification metrics for ' + str(file) + ' algorithm and ' + str(len(FSS)) + ' features selected by ' + str(pattern.split('.sav')[0]))
			
		acc, pr, re, f1 = accuracy_score(ytest, y_pred), precision_score(ytest, y_pred, zero_division=0), recall_score(ytest, y_pred, zero_division=0), f1_score(ytest, y_pred, zero_division=0)
		roc_auc, AP = roc_auc_score(ytest, yhat), average_precision_score(ytest, yhat)
		precision,recall,_ = precision_recall_curve(ytest, yhat)
		auc_score = auc(recall,precision)
		log_loss_ = log_loss(ytest, yhat)
		MAE, MSE = mean_absolute_error(ytest, y_pred), mean_squared_error(ytest, y_pred)

		resdict[file] = [acc, pr, re, f1, AP, roc_auc, auc_score, log_loss_, MAE, MSE, ",".join(f"{v:.6f}" for v in yhat)]

		if permutation:
						
			log('Prediction using permutation test for each classifier')
			metrics_.extend(['original_score', 'mean_permuted_score', 'p_value'])
			log('Performing permutation test for ' + str(file) + ' algorithm and ' + str(len(FSS)) + ' features selected by ' + str(pattern.split('.sav')[0]))

			try:

				original, permuted, pvalue = permutation_test(Xtrain, Xtest, ytest, ytrain, clf, FSS, metric=f1_score, n_permutations=c.n_perm)
				resdict[file].extend([original, permuted, pvalue])

			except Exception as e:

				warn('Permutation test failed for ' + str(file) + ' algorithm and ' + str(len(FSS)) + ' features selected by ' + str(pattern.split('.sav')[0]))

		log('Plot ROC curve for ' + str(file) + ' algorithm and ' + str(len(FSS)) + ' features selected by ' + str(pattern.split('.sav')[0]))
		plot_ROC_curve(yhat,ytest,name,c.OUT)

		log('Plot PR curve for ' + str(file) + ' algorithm and ' + str(len(FSS)) + ' features selected by ' + str(pattern.split('.sav')[0]))
		plot_PR_curve(yhat,ytest,ytrain,name,c.OUT)

		log('Plot Confusion Matrix for ' + str(file) + ' algorithm and ' + str(len(FSS)) + ' features selected by ' + str(pattern.split('.sav')[0]))
		sns.heatmap(confusion_matrix(ytest, y_pred), annot=True)
		plt.savefig(outdir + '/' + name + '.cm.pdf')
		plt.clf()

		model_metrics = pd.DataFrame.from_dict(resdict, orient='index', columns=metrics_)

		name =pattern.split('_alg.sav')[0]
		model_metrics['Analysis'] = pattern
		model_metrics['n_features']=len(FSS)
		
		model_metrics.to_csv(outdir + "/" + "model_metrics." + name + ".tsv", sep="\t")
	
	return resdict


def run(parser,args):

	c.INPUT= args.input
	c.validation=args.validation
	c.OUT=args.output
	c.label=args.label
	c.additional_training=args.additional_training
	c.additional_validation=args.additional_validation
	c.feature_selection=args.feature_selection
	c.metadata_validation=args.metadata_validation
	c.metadata_training=args.metadata_training
	c.permutation=args.permutation
	c.tunedir=args.dir
	c.verbose=args.verbose
	c.n_perm=args.n_perm
	c.seed=args.seed
	c.n_splits=args.n_splits
	c.n_repeats=args.n_repeats

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
		
		dataframe = pd.read_csv(c.INPUT, sep='\t', index_col='ptid')
		dataframe = dataframe.drop(columns=[col for col in dataframe.columns if "extraction_ID" in col or "diagnostics" in col])
		dataframe = dataframe.select_dtypes(include=['number'])
		dataframe.dropna(inplace=True)
	
	except:

		warn('ptid column not in ' + c.INPUT + ' try without searching for ptid column')

		try:
			
			dataframe = pd.read_csv(c.INPUT, sep='\t', index_col=None)
			dataframe = dataframe.drop(columns=[col for col in dataframe.columns if "extraction_ID" in col or "diagnostics" in col])
			dataframe = dataframe.select_dtypes(include=['number'])
			dataframe.dropna(inplace=True)
		
		except Exception as e:
			
			error('TSV ' + c.INPUT + ' does not exist, is not readable or is not a valid TSV')
			sys.exit(1)
	
	if c.additional_training:

		try:

			# Test dataset
			additional_training = pd.read_csv(c.additional_training, sep="\t", index_col='ptid')


		except:

			error('TSV' + c.additional_training + ' does not exist, is not readable or is not a valid TSV')
			sys.exit(1)

		dataframe=mergeDFs(dataframe,additional_training)
		dataframe.dropna(inplace=True)

	try:

		dataframe_finalTest = pd.read_csv(c.validation, sep='\t', index_col='ptid') #sampled 
		dataframe_finalTest = dataframe_finalTest.drop(columns=[col for col in dataframe_finalTest.columns if "extraction_ID" in col or "diagnostics" in col])
		dataframe_finalTest = dataframe_finalTest.select_dtypes(include=['number'])
		dataframe_finalTest.dropna(inplace=True)

	except:

		error('TSV ' + c.validation + ' does not exist, is not readable or is not a valid TSV')
		sys.exit(1)

	if c.additional_validation:

		try:

			# Test dataset
			additional_validation = pd.read_csv(c.additional_validation, sep="\t", index_col='ptid')


		except:

			error('TSV' + c.additional_validation + ' does not exist, is not readable or is not a valid TSV')
			sys.exit(1)

		dataframe_finalTest=mergeDFs(dataframe_finalTest,additional_validation)
		dataframe_finalTest.dropna(inplace=True)

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

	try:

		metadata_validation = pd.read_csv(c.metadata_validation, sep='\t', index_col='ptid')

	except:

		error('TSV ' + c.metadata_validation + ' does not exist, is not readable or is not a valid TSV')
		sys.exit(1)

	if c.metadata_training:

		try:

			metadata_training = pd.read_csv(c.metadata_training, sep="\t", index_col='ptid')
			dataframe = mergeDFs(dataframe, metadata_training[c.label])
			dataframe.dropna(inplace=True)

		except:

			error('TSV' + c.metadata_training + ' does not exist, is not readable or is not a valid TSV')
			sys.exit(1)

	log('Start Prediction')

	global seed
	seed=c.seed

	global cv
	cv = RepeatedStratifiedKFold(n_splits=c.n_splits, n_repeats=c.n_repeats, random_state=seed)

	#drop colu
	X_train = dataframe.drop(columns = c.label)
	y_train = dataframe[c.label]

	validation = mergeDFs(dataframe_finalTest, metadata_validation[c.label])
	validation.dropna(inplace=True)
	X_test = validation.drop(columns = c.label)
	y_test = validation[c.label]

	for col in FS.columns:

		searched= col + '_alg.sav'#r'ANOVA.*\.sav$'
		FS_ = list(FS[col].dropna().sort_values().index)
		resPred=Prediction(X_train,X_test,y_test,y_train,searched,FS_,c.permutation,c.tunedir,c.OUT)

	log('Done')
	sys.exit(0)