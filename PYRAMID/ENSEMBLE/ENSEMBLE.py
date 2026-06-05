import argparse
import sys
from argparse import HelpFormatter
import numpy as np
import pandas as pd
from datetime import datetime
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve,average_precision_score
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class c():

	OUT= ''
	INPUT=''
	metadata_validation=''
	label=''
	validation=''

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

class TSVEnsemble:
	
	"""
	Ensemble class for combining predictions from TSV file containing model results
	Based on the methodology from https://doi.org/10.1016/j.acra.2023.07.024
	"""
	
	def __init__(self, df: pd.DataFrame, score_column: str = 'roc_auc'):
		
		"""
		Initialize the ensemble with data from TSV file
		
		Args:
			df: DataFrame containing model results with 'prob' column and performance metrics
			score_column: Column name to use for ranking models (default: 'roc_auc')
		"""
		self.df = df.copy()
		self.score_column = score_column
		self.top_models_df = None
		self.top_predictions = None
		
		# Parse predictions from prob column
		self._parse_predictions()
		
	def _parse_predictions(self):
		
		"""Parse the comma-separated probabilities from the 'prob' column"""
		
		predictions_list = []
		
		for idx, row in self.df.iterrows():
			
			try:
				
				prob_str = str(row['prob'])
				
				if prob_str and prob_str != 'nan':
					
					# Parse comma-separated probabilities
					prob_values = [float(x.strip()) for x in prob_str.split(',')]
					predictions_list.append(np.array(prob_values))
				
				else:
					
					warn('Empty or invalid prob for row ' + str(idx))
					predictions_list.append(np.array([]))
			
			except Exception as e:
				
				warn('Could not parse probabilities for row ' + str(idx) + ': ' + str(e))
				predictions_list.append(np.array([]))
		
		self.df['parsed_predictions'] = predictions_list 
		self.df['binary_predictions'] = self.df['parsed_predictions'].apply(
												lambda arr: (arr > 0.5).astype(int))

		# Remove rows with empty predictions
		valid_mask = self.df['parsed_predictions'].apply(lambda x: len(x) > 0)
		self.df = self.df[valid_mask].reset_index(drop=True)

		# Use all models directly
		self.top_models_df = self.df
		self.top_predictions = list(self.df['parsed_predictions'])
		self.top_model_names = [f'Model_{i}' for i in range(len(self.df))]

		log('Successfully parsed predictions for ' + str(len(self.df)) + ' models')
		
	def get_ensemble_predictions(self) -> np.ndarray:
		
		"""
		Generate ensemble predictions by averaging posterior probabilities
		
		Returns:
			Averaged posterior probabilities from top models
		"""
		
		if self.top_predictions is None:
			
			raise ValueError("No valid predictions available")
			sys.exit(1)

		if not self.top_predictions:
			
			raise ValueError("No valid predictions available for ensemble")
			sys.exit(1)
		
		# Check if all predictions have the same length
		lengths = [len(pred) for pred in self.top_predictions]
		
		if len(set(lengths)) > 1:

			warn('Predictions have different lengths: ' + str(set(lengths)))
			
			# Use minimum length to make them compatible
			min_length = min(lengths)
			self.top_predictions = [pred[:min_length] for pred in self.top_predictions]
			warn('Truncated all predictions to length ' + str(min_length))
		
		# Average the posterior probabilities
		ensemble_predictions = np.mean(self.top_predictions, axis=0)
		
		log('Generated ensemble predictions from ' + str(len(self.top_predictions)) + ' models')
		log('Ensemble prediction shape: ' + str(ensemble_predictions.shape))
		log('Prediction range: [' + f'{ensemble_predictions.min():.4f}' + ', ' + f'{ensemble_predictions.max():.4f}' + ']')

		#self.ensemble_predictions = ensemble_predictions

		#try weighted means
		#weights=self.df['n_features_prediction']

		# Weighted average across models
		#ensemble_predictions_weighted = np.average(self.top_predictions, axis=0, weights=weights)

		return ensemble_predictions

	def evaluate_ensemble(self, y_true: np.ndarray) -> Dict[str, float]:
		"""
		Evaluate ensemble performance
		
		Args:
			y_true: True labels
			
		Returns:
			Dictionary of performance metrics
		"""
		ensemble_pred = self.get_ensemble_predictions()
		
		# Handle different prediction formats
		if len(ensemble_pred) != len(y_true):

			warn('Prediction length (' + str(len(ensemble_pred)) + ') != y_true length (' + str(len(y_true)) + ')')
			min_len = min(len(ensemble_pred), len(y_true))
			ensemble_pred = ensemble_pred[:min_len]
			y_true = y_true[:min_len]
		
		# Convert probabilities to binary predictions using 0.5 threshold
		y_pred_binary = ROC_threshold(y_true,ensemble_pred) #(ensemble_pred > 0.5).astype(int)
		
		metrics = {}
		
		try:
			metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
			metrics['precision'] = precision_score(y_true, y_pred_binary)#, average='weighted', zero_division=0)
			metrics['recall'] = recall_score(y_true, y_pred_binary)#, average='weighted', zero_division=0)
			metrics['f1_score'] = f1_score(y_true, y_pred_binary)#, average='weighted', zero_division=0)
			metrics['AP']= average_precision_score(y_true, y_pred_binary)

			# ROC-AUC using probabilities
			if np.max(ensemble_pred) <= 1.0 and np.min(ensemble_pred) >= 0.0:
				
				metrics['roc_auc'] = roc_auc_score(y_true, ensemble_pred)
			
			else:
				
				warn('Predictions not in [0,1] range, skipping ROC-AUC')
				
		except Exception as e:
			
			warn('Could not calculate some metrics: ' + str(e))
			
		return metrics
	
	def get_top_models_info(self) -> pd.DataFrame:
		"""
		Get information about the selected top models
		
		Returns:
			DataFrame with information about top models
		"""
		if self.top_models_df is None:
			
			raise ValueError("No valid predictions available")
			sys.exit(1)
		
		# Return relevant columns
		info_cols = ['Analysis_prediction', self.score_column, 'ML', 'FS', 'transformation', 
					'sampling', 'n_features_prediction', 'f1_prediction', 'accuracy']
		available_cols = [col for col in info_cols if col in self.top_models_df.columns]
		
		return self.top_models_df[available_cols].copy()
	
	def plot_ensemble_analysis(self, y_true: np.ndarray = None, figsize: Tuple[int, int] = (20, 17),save_path: str = None):
		"""
		Create comprehensive visualization of ensemble analysis
		
		Args:
			y_true: True labels for evaluation plots (optional)
			figsize: Figure size for the plots
		"""
		if self.top_models_df is None:
			
			raise ValueError("No valid predictions available")
			sys.exit(1)
		
		# Set up the plot style
		plt.style.use('default')
		fig = plt.figure(figsize=figsize)
		
		# Create subplot layout
		if y_true is not None:
			gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
		else:
			gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
		
		# 1. Score distribution of all models vs top models
		ax1 = fig.add_subplot(gs[0, 0])
		self._plot_score_distribution(ax1)
		
		# 2. Model type distribution in top models
		ax2 = fig.add_subplot(gs[0, 1])
		self._plot_model_types(ax2)
		
		# 3. Feature selection methods in top models
		ax3 = fig.add_subplot(gs[0, 2])
		self._plot_feature_selection(ax3)
		
		# 4. Prediction distribution
		ax4 = fig.add_subplot(gs[1, 0])
		self._plot_prediction_distribution(ax4)
		
		# 5. Individual vs ensemble predictions
		ax5 = fig.add_subplot(gs[1, 1])
		self._plot_individual_vs_ensemble(ax5)
		
		# 6. Top model scores
		ax6 = fig.add_subplot(gs[1, 2])
		self._plot_top_model_scores(ax6)
		
		if y_true is not None:
			ensemble_pred = self.get_ensemble_predictions()
			
			# Ensure same length
			min_len = min(len(ensemble_pred), len(y_true))
			ensemble_pred = ensemble_pred[:min_len]
			y_true = y_true[:min_len]
			
			# 7. ROC Curve
			ax7 = fig.add_subplot(gs[2, 0])
			self._plot_roc_curve(ax7, y_true, ensemble_pred)
			
			# 8. Precision-Recall Curve
			ax8 = fig.add_subplot(gs[2, 1])
			self._plot_precision_recall_curve(ax8, y_true, ensemble_pred)
			
			# 9. Prediction vs True labels
			ax9 = fig.add_subplot(gs[2, 2])
			self._plot_predictions_vs_truth(ax9, y_true, ensemble_pred)
		
		plt.suptitle(f'Ensemble Analysis: Top {len(self.top_predictions)} Models', 
					 fontsize=16, fontweight='bold')
		plt.tight_layout(rect=[0, 0, 1, 0.97])
		
		if save_path:
	
			plt.savefig(save_path + '/' + 'ensemble_analysis.pdf')
			
		else:
	
			plt.show()
	
	def _plot_score_distribution(self, ax):
		"""Plot distribution of scores for all models vs top models"""
		ax.hist(self.df[self.score_column], bins=30, alpha=0.6, label='All Models', color='lightblue')
		ax.hist(self.top_models_df[self.score_column], bins=20, alpha=0.8, label=f'Top {len(self.top_models_df)}', color='darkblue')
		ax.set_xlabel(f'{self.score_column.upper()} Score')
		ax.set_ylabel('Frequency')
		ax.set_title('Score Distribution')
		ax.legend()
		ax.grid(True, alpha=0.3)
	
	def _plot_model_types(self, ax):
		"""Plot distribution of ML model types in top models"""
		if 'ML' in self.top_models_df.columns:
			ml_counts = self.top_models_df['ML'].value_counts()
			colors = plt.cm.Set3(np.linspace(0, 1, len(ml_counts)))
			wedges, texts, autotexts = ax.pie(ml_counts.values, labels=ml_counts.index, autopct='%1.1f%%', colors=colors)
			ax.set_title('ML Model Types in Top Models')
		else:
			ax.text(0.5, 0.5, 'ML column not available', ha='center', va='center', transform=ax.transAxes)
			ax.set_title('ML Model Types')
	
	def _plot_feature_selection(self, ax):
		"""Plot distribution of feature selection methods in top models"""
		if 'FS' in self.top_models_df.columns:
			fs_counts = self.top_models_df['FS'].value_counts()
			colors = plt.cm.Set2(np.linspace(0, 1, len(fs_counts)))
			ax.bar(range(len(fs_counts)), fs_counts.values, color=colors)
			ax.set_xticks(range(len(fs_counts)))
			ax.set_xticklabels(fs_counts.index, rotation=45, ha='right')
			ax.set_ylabel('Count')
			ax.set_title('Feature Selection Methods')
		else:
			ax.text(0.5, 0.5, 'FS column not available', ha='center', va='center', transform=ax.transAxes)
			ax.set_title('Feature Selection Methods')
	
	def _plot_prediction_distribution(self, ax):
		"""Plot distribution of ensemble predictions"""
		ensemble_pred = self.get_ensemble_predictions()
		ax.hist(ensemble_pred, bins=30, alpha=0.7, color='green', edgecolor='black')
		ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
		ax.set_xlabel('Ensemble Prediction Probability')
		ax.set_ylabel('Frequency')
		ax.set_title('Ensemble Prediction Distribution')
		ax.legend()
		ax.grid(True, alpha=0.3)
	
	def _plot_individual_vs_ensemble(self, ax):
		"""Plot individual model predictions vs ensemble"""
		if len(self.top_predictions) > 10:
			# Sample 10 random models for visualization
			sample_idx = np.random.choice(len(self.top_predictions), 10, replace=False)
			sample_predictions = [self.top_predictions[i] for i in sample_idx]
			sample_names = [self.top_model_names[i] for i in sample_idx]
		else:
			sample_predictions = self.top_predictions
			sample_names = self.top_model_names
		
		ensemble_pred = self.get_ensemble_predictions()
		
		# Plot first few samples
		n_samples = min(100, len(ensemble_pred))
		x = range(n_samples)
		
		# Plot individual models (faded)
		for i, pred in enumerate(sample_predictions):
			ax.plot(x, pred[:n_samples], alpha=0.3, linewidth=0.5, color='gray')
		
		# Plot ensemble (bold)
		ax.plot(x, ensemble_pred[:n_samples], color='red', linewidth=2, label='Ensemble')
		ax.set_xlabel('Sample Index')
		ax.set_ylabel('Prediction Probability')
		ax.set_title('Individual Models vs Ensemble Predictions')
		ax.legend()
		ax.grid(True, alpha=0.3)
	
	def _plot_top_model_scores(self, ax):
		"""Plot scores of top models"""
		scores = self.top_models_df[self.score_column].values
		x = range(len(scores))
		ax.plot(x, scores, 'o-', color='purple', markersize=4)
		ax.set_xlabel('Model Rank')
		ax.set_ylabel(f'{self.score_column.upper()} Score')
		ax.set_title(f'Top {len(scores)} Model Scores')
		ax.grid(True, alpha=0.3)
	
	def _plot_roc_curve(self, ax, y_true, ensemble_pred):
		"""Plot ROC curve"""
		try:
			fpr, tpr, _ = roc_curve(y_true, ensemble_pred)
			roc_auc = roc_auc_score(y_true, ensemble_pred)
			
			ax.plot(fpr, tpr, color='darkorange', lw=2, 
				   label=f'ROC Curve (AUC = {roc_auc:.3f})')
			ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
			ax.set_xlim([0.0, 1.0])
			ax.set_ylim([0.0, 1.05])
			ax.set_xlabel('False Positive Rate')
			ax.set_ylabel('True Positive Rate')
			ax.set_title('ROC Curve')
			ax.legend(loc="lower right")
			ax.grid(True, alpha=0.3)
		except Exception as e:
			ax.text(0.5, 0.5, f'ROC curve error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
	
	def _plot_precision_recall_curve(self, ax, y_true, ensemble_pred):
		"""Plot Precision-Recall curve"""
		try:
			precision, recall, _ = precision_recall_curve(y_true, ensemble_pred)
			
			ax.plot(recall, precision, color='blue', lw=2, label='PR Curve')
			ax.set_xlim([0.0, 1.0])
			ax.set_ylim([0.0, 1.05])
			ax.set_xlabel('Recall')
			ax.set_ylabel('Precision')
			ax.set_title('Precision-Recall Curve')
			ax.legend(loc="lower left")
			ax.grid(True, alpha=0.3)
		except Exception as e:
			ax.text(0.5, 0.5, f'PR curve error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
	
	def _plot_predictions_vs_truth(self, ax, y_true,ensemble_pred): #
		"""Plot predictions vs true labels"""
		# Create bins for visualization
		y_pred_binary = ROC_threshold(y_true,ensemble_pred)
		
		# Scatter plot with jitter
		jitter = 0.1
		y_true_jitter = y_true + np.random.normal(0, jitter, len(y_true))
		
		colors = ['red' if yt != yp else 'green' for yt, yp in zip(y_true, y_pred_binary)]
		ax.scatter(ensemble_pred, y_true_jitter, c=colors, alpha=0.6, s=20)
		
		#ax.axvline(0.5, color='black', linestyle='--', alpha=0.7, label='Decision Threshold')
		ax.set_xlabel('Ensemble Prediction Probability')
		ax.set_ylabel('True Label (with jitter)')
		ax.set_title('Predictions vs True Labels')
		ax.set_ylim([-0.5, 1.5])
		ax.legend(['Correct', 'Incorrect', 'Threshold'])
		ax.grid(True, alpha=0.3)
	
def ROC_threshold(y_true, final_predictions):

	'''
	Find the optimal classification threshold from the ROC curve (point closest
	to the top-left corner) and return binary predictions reclassified at that
	threshold.
	'''

	## find optimal threshold with roc analysis 
	#Get ROC curve data
	fpr, tpr, thresholds = roc_curve(y_true, final_predictions)

	# Find the best threshold (closest to top-left corner)
	distances = np.sqrt((1 - tpr)**2 + fpr**2)
	optimal_idx = np.argmin(distances)
	optimal_threshold = thresholds[optimal_idx]

	# Reclassify using optimal threshold
	binary_predictions = (final_predictions >= optimal_threshold).astype(int)

	return binary_predictions

def load_tsv_ensemble(tsv_file: str, score_column: str = 'roc_auc') -> TSVEnsemble:
	
	"""
	Load ensemble from TSV file containing model results

	Args:
		tsv_file: Path to TSV file (e.g., 'top1.models.chain.tsv')
		score_column: Column to use for ranking models
	"""
	
	log('Loading data from ' + tsv_file)
	
	try:
		# Try to load with different separators
		if tsv_file.endswith('.tsv'):
			df = pd.read_csv(tsv_file, sep='\t')
		else:
			df = pd.read_csv(tsv_file)
			
		#print(f"Loaded {len(df)} models from {tsv_file}")
		#print(f"Columns: {list(df.columns)}")
		
		# Check required columns
		if 'prob' not in df.columns:
			
			raise ValueError("'prob' column not found in the file")
			sys.exit(1)

		if score_column not in df.columns:
			warn("'" + score_column + "' column not found. Available columns: " + str(list(df.columns)))
			# Try common alternatives
			score_alternatives = ['roc_auc', 'f1_prediction', 'accuracy', 'AP']
			for alt in score_alternatives:
				
				if alt in df.columns:
					
					log("Using '" + alt + "' instead of '" + score_column + "'")
					score_column = alt
					break
			else:
				raise ValueError('No suitable score column found. Available: ' + str(list(df.columns)))
				sys.exit(1)
		
		return TSVEnsemble(df, score_column)
		
	except Exception as e:
		
		error('loading ' + tsv_file + ': ' + str(e))
		raise
		sys.exit(1)


def inspect_tsv_file(tsv_file: str, n_rows: int = 5):
	
	"""
	Inspect the TSV file to understand its structure
	
	Args:
		tsv_file: Path to TSV file
		n_rows: Number of rows to display
	"""
	
	try:
		
		if tsv_file.endswith('.tsv'):
			
			df = pd.read_csv(tsv_file, sep='\t')
		
		else:
			
			df = pd.read_csv(tsv_file)
			
		print(f"File: {tsv_file}")
		print(f"Shape: {df.shape}")
		print(f"Columns: {list(df.columns)}")
		
		# Show sample of prob column
		if 'prob' in df.columns:
			
			print(f"\nSample 'prob' values:")
			
			for i in range(min(3, len(df))):
				
				prob_str = str(df['prob'].iloc[i])
				print(f"  Row {i}: {prob_str[:100]}{'...' if len(prob_str) > 100 else ''}")
		
		# Show performance metrics
		score_cols = ['accuracy', 'roc_auc', 'f1_prediction', 'AP', 'precision', 'recall']
		available_scores = [col for col in score_cols if col in df.columns]
		
		if available_scores:
			
			print(f"\nAvailable performance metrics: {available_scores}")
			print(df[available_scores].describe())
		
		# Show first few rows
		print(f"\nFirst {n_rows} rows:")
		display_cols = ['Analysis_prediction', 'ML', 'FS'] + available_scores[:3]
		display_cols = [col for col in display_cols if col in df.columns]
		print(df[display_cols].head(n_rows).to_string())
		
	except Exception as e:
		
		print(f"Error inspecting {tsv_file}: {e}")

def create_performance_summary_table(ensemble: TSVEnsemble, y_true: np.ndarray) -> pd.DataFrame:
	
	"""
	Create a performance summary table comparing:
	- Ensemble's overall performance
	- Median performance of top individual models (from top_models_df)

	Args:
		ensemble: TSVEnsemble instance
		y_true: Ground truth labels

	Returns:
		pd.DataFrame: Summary with metrics from ensemble and median of top models
	"""
	
	if ensemble.top_models_df is None:
		
		raise ValueError("No valid predictions available")
		sys.exit(1)

	# Evaluate ensemble predictions
	ensemble_metrics = ensemble.evaluate_ensemble(y_true)

	# Mapping: Ensemble metric name -> column in top_models_df
	metric_column_map = {
		'accuracy': 'accuracy',
		'precision': 'pr',
		'recall': 're',
		'f1_score': 'f1_prediction',
		'roc_auc': 'roc_auc',
		'AP' : 'AP'
	}

	# Compute median metrics from top models
	model_medians = {}
	
	for metric, col in metric_column_map.items():
		
		if col in ensemble.top_models_df.columns:
			
			model_medians[metric] = ensemble.top_models_df[col].median()
		
		else:
			
			model_medians[metric] = np.nan

	# Create summary DataFrame
	summary_df = pd.DataFrame({
		'Metric': list(ensemble_metrics.keys()),
		'Ensemble': [ensemble_metrics[k] for k in ensemble_metrics],
		'Median_Top_Models': [model_medians[k] for k in ensemble_metrics]
	})

	return summary_df

def run(parser,args):

	'''
	Check arguments, run functions
	'''

	c.INPUT= args.input
	c.OUT=args.output
	c.metadata_validation=args.metadata_validation
	c.label=args.label  
	c.validation=args.validation

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

		ensemble = load_tsv_ensemble(c.INPUT, score_column='roc_auc')

	except:

		error('TSV ' + c.INPUT + ' does not exist, is not readable or is not a valid TSV')
		sys.exit(1)

	try:

		dataframe_finalTest = pd.read_csv(c.validation, sep='\t', index_col='ptid') #sampled 
		dataframe_finalTest = dataframe_finalTest.drop(columns=[col for col in dataframe_finalTest.columns if "extraction_ID" in col or "diagnostics" in col])
		dataframe_finalTest = dataframe_finalTest.select_dtypes(include=['number'])
		dataframe_finalTest.dropna(inplace=True)

	except:

		error('TSV ' + c.validation + ' does not exist, is not readable or is not a valid TSV')
		sys.exit(1)

	try:

		metadata_validation = pd.read_csv(c.metadata_validation, sep='\t', index_col='ptid')

	except:

		error('TSV ' + c.metadata_validation + ' does not exist, is not readable or is not a valid TSV')
		sys.exit(1)


	validation = mergeDFs(dataframe_finalTest, metadata_validation[c.label])
	validation.dropna(inplace=True)
	X_test = validation.drop(columns = c.label)
	y_true = validation[c.label]

	# Load TSV file

	ensemble.df['pipeline'] = (
		ensemble.df['transformation'].astype(str) + '_' +
		ensemble.df['sampling'].astype(str) + '_' +
		ensemble.df['FS'].astype(str) + '_' +
		ensemble.df['ML'].astype(str)
	)
	ensemble.top_model_names = list(ensemble.df['pipeline'])

	log('Ensemble of ' + str(ensemble.df.shape[0]) + ' pipelines by averaging their posterior probabilities')

	# Get ensemble predictions
	final_predictions =ensemble.get_ensemble_predictions()

	#store results
	y_pred_binary= ROC_threshold(y_true,final_predictions) #(final_predictions > 0.5).astype(int)
	table_pred = y_true.to_frame()
	table_pred['y_pred_binary']=y_pred_binary 
	table_pred.to_csv(c.OUT + "/" + "table_pred.tsv",sep="\t")

	# Evaluate the ensemble
	if y_true is not None:
		
		metrics = ensemble.evaluate_ensemble(y_true)
		log('Top ' + str(ensemble.df.shape[0]) + ' Ensemble Performance')
		
		for metric, value in metrics.items():
			
			log(str(metric) + ': ' + f'{value:.4f}')

	#Create comprehensive visualizations
	ensemble.plot_ensemble_analysis(y_true=y_true,figsize=(30,30),save_path=c.OUT)

	# Get performance summary table
	log('Writing summary output')
	summary = create_performance_summary_table(ensemble, y_true=y_true)
	summary.to_csv(c.OUT + "/" + "summary_table.tsv",sep="\t", index=None)

	sys.exit(0)