import pandas as pd
import numpy as np
import os
import argparse
from argparse import HelpFormatter
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from datetime import datetime
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class c():
	OUT= ''
	INPUT=''
	VALIDATION=''
	additional_training=''
	additional_validation=''
	transformation=''
	n_quantiles=''

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
	Logs an error message with a timestamp and intended for use before sys.exit().
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
	
	'''Custom help format'''
	
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

def transform(args, transformation, Xtrain, Xtest=None):
	
	'''Set the data transforms'''
	
	has_test = Xtest is not None

	if transformation == "scaler":
		
		log('Perform standardization on train dataset')
		
		try:
			
			scaler = StandardScaler()
			scaler.fit(Xtrain)
			Xtrain = pd.DataFrame(scaler.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log('Perform standardization on test dataset')
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]
				Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		
		except Exception as e:
			
			error(f'Standardization failed using {transformation} : {e}')
			sys.exit(1)

	elif transformation == "normalize":
		
		log('Perform normalization on train dataset')
		
		try:
			
			normalizer = MinMaxScaler()
			normalizer.fit(Xtrain)
			Xtrain = pd.DataFrame(normalizer.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log('Perform normalization on test dataset')
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]
				Xtest = pd.DataFrame(normalizer.transform(Xtest), columns=Xtest.columns, index=Xtest.index)

		except Exception as e:
			
			error(f'Normalization failed using {transformation} : {e}')
			sys.exit(1)

	elif transformation == "mixture":
		
		log('Perform standardization on train dataset')
		
		try:
			
			scaler = StandardScaler()
			scaler.fit(Xtrain)
			Xtrain = pd.DataFrame(scaler.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			log('Perform normalization on train dataset')
			normalizer = MinMaxScaler()
			normalizer.fit(Xtrain)
			Xtrain = pd.DataFrame(normalizer.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log('Perform standardization on test dataset')
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]
				Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
				log('Perform normalization on test dataset')
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
			
			scaler = RobustScaler(quantile_range=(args.robust_parameter[0], args.robust_parameter[1]))
			scaler.fit(Xtrain)
			Xtrain = pd.DataFrame(scaler.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log(f'Perform Robust transformation using the range {args.robust_parameter[0]:.2f} , {args.robust_parameter[1]:.2f} on test dataset')
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]
				Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		
		except Exception as e:
			
			error(f'Robust transformation failed using {transformation} with {args.robust_parameter[0]:.2f}, {args.robust_parameter[1]:.2f} : {e}')
			sys.exit(1)

	elif transformation == "yeo-johnson":
		
		log('Perform yeo-johnson transformation on train dataset')
		
		try:
			
			power = PowerTransformer(method='yeo-johnson', standardize=True)
			power.fit(Xtrain)
			Xtrain = pd.DataFrame(power.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log('Perform yeo-johnson transformation on test dataset')
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]
				Xtest = pd.DataFrame(power.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		
		except Exception as e:
			
			warn(f'Power transformation failed using {transformation} : {e} due to many outliers')
			
			# Compute train statistics once, to be reused on test (no data leakage)
			train_mean = np.mean(Xtrain.values, axis=0)
			train_std = np.std(Xtrain.values, axis=0)
			
			log('Perform Standardization prior to yeo-johnson on train dataset')
			X_scaled_train = (Xtrain.values - train_mean) / (train_std + 1e-10)
			log('Replace any infinite on train dataset')
			X_scaled_train = np.nan_to_num(X_scaled_train, posinf=1e10, neginf=-1e10)
			log('Handle constant values on train dataset')
			constant_mask = np.std(X_scaled_train, axis=0) < 1e-10
			X_scaled_train[:, constant_mask] = 0
			log('Add tiny noise to prevent perfect correlations on train dataset')
			X_scaled_train += np.random.normal(0, 1e-10, X_scaled_train.shape)
			log('Apply yeo-johnson on train dataset')
			power = PowerTransformer(method='yeo-johnson', standardize=True)
			power.fit(X_scaled_train)
			Xtrain = pd.DataFrame(power.transform(X_scaled_train), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]
				
				# FIX: use train mean/std to standardize test — no data leakage
				log('Perform Standardization prior to yeo-johnson on test dataset')
				X_scaled_test = (Xtest.values - train_mean) / (train_std + 1e-10)
				log('Replace any infinite on test dataset')
				X_scaled_test = np.nan_to_num(X_scaled_test, posinf=1e10, neginf=-1e10)
				log('Handle constant values on test dataset')
				constant_mask_test = np.std(X_scaled_test, axis=0) < 1e-10
				X_scaled_test[:, constant_mask_test] = 0
				log('Add tiny noise to prevent perfect correlations on test dataset')
				X_scaled_test += np.random.normal(0, 1e-10, X_scaled_test.shape)
				log('Apply yeo-johnson on test dataset')
				Xtest = pd.DataFrame(power.transform(X_scaled_test), columns=Xtest.columns, index=Xtest.index)

	elif transformation == "box-cox":
		
		log('Perform box-cox transformation on train dataset')
		
		try:
			
			Xtrain = Xtrain.loc[:, Xtrain.nunique() > 1]  # filter constant cols from train
			minmax_scaler = MinMaxScaler(feature_range=(1, 2))
			minmax_scaler.fit(Xtrain)
			Xtrain = pd.DataFrame(minmax_scaler.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			power_scaler = PowerTransformer(method='box-cox', standardize=True)
			power_scaler.fit(Xtrain)
			Xtrain = pd.DataFrame(power_scaler.transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
			
			if has_test:
				
				log('Perform box-cox transformation on test dataset')
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]
				Xtest = pd.DataFrame(minmax_scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
				Xtest = Xtest.clip(lower=1e-6)
				Xtest = pd.DataFrame(power_scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		
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
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		
		except Exception as e:
			
			error(f'Quantile transformation failed using distribution uniform with {args.n_quantiles} : {e}')
			sys.exit(1)

	elif transformation == "quantile-normal":
		
		if isinstance(args.n_quantiles, (int, float)):
			
			# FIX: use log instead of warn — this is an informational message, not a warning
			log(f'Perform Normal Quantile transformation using n_quantile {args.n_quantiles} on train dataset')
		
		else:
			
			error('--n-quantiles is not an int or float')
			sys.exit(1)

		try:
			
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
				common_cols = Xtrain.columns.intersection(Xtest.columns)
				Xtrain = Xtrain[common_cols]
				Xtest = Xtest[common_cols]
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
		
		except Exception as e:
			
			error(f'Quantile transformation failed using normal distribution with {args.n_quantiles} : {e}')
			sys.exit(1)
			
	return (Xtrain, Xtest) if has_test else Xtrain

def run(parser, args):

	c.INPUT = args.input
	c.OUT = args.output
	c.VALIDATION = args.validation
	c.transformation = args.transformation
	c.additional_training = args.additional_training
	c.additional_validation = args.additional_validation
	c.n_quantiles = args.n_quantiles

	if not os.path.exists(c.OUT):
		
		try:
			
			os.makedirs(c.OUT)
		
		except:
			
			error('Cannot create the output folder')
			sys.exit(1)
	
	else:
		
		if not os.access(os.path.abspath(c.OUT), os.W_OK):
			
			error('Missing write permissions on the output folder')
			sys.exit(1)
		
		elif os.listdir(os.path.abspath(c.OUT)):
			
			error('The output folder is not empty: specify another output folder or clean the current one')
			sys.exit(1)

	try:
		
		training = pd.read_csv(c.INPUT, sep='\t', index_col='ptid')
		training = training.drop(columns=[col for col in training.columns if "extraction_ID" in col or "diagnostics" in col])
		training_ = training.select_dtypes(include=['number'])
		training_.dropna(inplace=True)
	
	except:
		
		error(f'TSV {c.INPUT} does not exist, is not readable or is not a valid TSV')
		sys.exit(1)

	if c.VALIDATION:
		
		try:
			
			validation = pd.read_csv(c.VALIDATION, sep="\t", index_col='ptid')
			validation = validation.drop(columns=[col for col in validation.columns if "extraction_ID" in col or "diagnostics" in col])
			validation.dropna(inplace=True)
			validation_ = validation.select_dtypes(include=['number'])
		
		except:
			
			error(f'TSV {c.VALIDATION} does not exist, is not readable or is not a valid TSV')
			sys.exit(1)

		if c.additional_validation:
			
			try:
				
				additional_validation = pd.read_csv(c.additional_validation, sep="\t", index_col='ptid')
			
			except:
				
				error(f'TSV {c.additional_validation} does not exist, is not readable or is not a valid TSV')
				sys.exit(1)
			
			validation = mergeDFs(validation, additional_validation)
			validation.dropna(inplace=True)
			validation_ = validation.select_dtypes(include=['number'])

	if c.additional_training:
		
		try:
			
			additional_training = pd.read_csv(c.additional_training, sep="\t", index_col='ptid')
		
		except:
			
			error(f'TSV {c.additional_training} does not exist, is not readable or is not a valid TSV')
			sys.exit(1)
		
		log(f'Start Data transformation using {c.additional_training} file to add covariates')
		training_ = mergeDFs(training, additional_training)
		training_.dropna(inplace=True)
		training_ = training_.select_dtypes(include=['number'])
		
	log('Start Data transformation')

	if c.VALIDATION:
		
		X_train_, X_test_ = transform(args, c.transformation, training_, validation_)
		log('Transformation Done')
		log('Writing output')
		X_train_.to_csv(os.path.join(c.OUT, 'train.transformed.tsv'), sep='\t', index=True)
		X_test_.to_csv(os.path.join(c.OUT, 'test.transformed.tsv'), sep='\t', index=True)
	
	else:
		
		X_train_ = transform(args, c.transformation, training_)
		log('Transformation Done')
		log('Writing output')
		X_train_.to_csv(os.path.join(c.OUT, 'train.transformed.tsv'), sep='\t', index=True)

	sys.exit(0)