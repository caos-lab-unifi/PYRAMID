#!/usr/bin/python3 env

import sys
import argparse
from argparse import HelpFormatter

from PYRAMID import __version__


def main():

    parser = argparse.ArgumentParser(prog='PYRAMID', description='''PYthon Radiomics And Machine learning Data analysis''', epilog='''This program was developed by Simone Romagnoli at the Cancer Omics Laboratory at University of Florence. Extensive documentation is available at: https://littleisland8.github.io/pyramiddoc/''', formatter_class=CustomFormat) 

    subparsers = parser.add_subparsers(title='modules', dest='command', metavar='DaTrax,SLIC,FS,FeatX,HyPerTune,PREDICT,ENSEMBLE,SURV')

    ## Datrax ##


    parser_datrax = subparsers.add_parser('DaTrax', help='Data Transform')

    required = parser_datrax.add_argument_group('Required I/O arguments')

    required.add_argument('-i', '--input', help='TSV file contained pyradiomics feature-extracted for training dataset', metavar='TSV', required=True)
    required.add_argument('-o', '--output', help='Output folder', metavar='DIR', required=True)
    required.add_argument('-t','--transformation' ,help='Transformation to apply to the dataset {scaler|normalize|mixture|robust|yeo-johnson|box-cox|quantile-uniform|quantile-normal}',choices=["scaler","normalize","mixture","robust","yeo-johnson","box-cox","quantile-uniform","quantile-normal"],required=True,metavar='str')
    
    additionals=parser_datrax.add_argument_group('Additional arguments')

    additionals.add_argument('-v', '--validation', help='TSV file contained pyradiomics feature-extracted for test dataset', metavar='TSV')
    additionals.add_argument('--additional_training', help='TSV containg additional covariates for training', metavar='TSV')
    additionals.add_argument('--additional_validation', help='TSV containg additional covariates for validation', metavar='TSV')    
    additionals.add_argument('--robust-parameter', help='Quantile range for robust scaler', type=parse_robust_parameter, metavar='VAL1,VAL2', default=[25.0,75.0])
    additionals.add_argument('--n-quantiles', help='Number of quantiles for Quantile Transformation',type=int, default=100)

    parser_datrax.set_defaults(func=run_subtool)

    ## SLIC ##
    

    parser_slic = subparsers.add_parser('SLIC', help='Sampling for Leveling Imbalanced Classes')

    required = parser_slic.add_argument_group('Required I/O arguments')

    required.add_argument('-i', '--input', help='TSV file contained pyradiomics feature-extracted for training dataset', metavar='TSV', required=True)
    required.add_argument('-m', '--metadata', help='TSV file contained label to be predicted', metavar='TSV', required=True)
    required.add_argument('-l', '--label', help='Label to be predicted', metavar='str', required=True)
    required.add_argument('-o', '--output', help='Output folder', metavar='DIR', required=True)
    required.add_argument('-a', '--algorithm', help='Algorithm to apply for sampling procedures {SMOTE|ADASYN|SMOTETomek|SMOTEEN|AllKNN|CNN|RENN}', required=True, choices=["SMOTE","ADASYN","SMOTETomek","SMOTEEN","AllKNN","CNN","RENN"], metavar='str')

    additionals=parser_slic.add_argument_group('Additional arguments')
    additionals.add_argument('--additional', help='TSV containg additional covariates for training', metavar='TSV')
    additionals.add_argument('--sampling-strategy', help='Sampling percentage to resample the minority class [auto]', metavar="float|str",type=str, choices=[*map(lambda x: str(round(x * 0.1, 1)), range(0, 11))] + ['auto'] + ['all'], default='auto')
    additionals.add_argument('-t', '--threads', help='Number of threads for Sampling methods [10]', type=int, default=10, metavar='int')
    additionals.add_argument('--seed', help='Seed for sampling strategy [None]', metavar="int", type=int, default=None)
    additionals.add_argument('--neighbors', help='n neighbors to be used for sampling strategy [5]', metavar="int", type=int, default=5)

    parser_slic.set_defaults(func=run_subtool)

    ## FS ##
    

    parser_fs = subparsers.add_parser('FS', help='Feature Selection using one or more algorithms')

    required = parser_fs.add_argument_group('Required I/O arguments')

    required.add_argument('-i', '--input', help='TSV file contained pyradiomics feature-extracted for training dataset', metavar='TSV', required=True)
    required.add_argument('-o', '--output', help='Output folder', metavar='DIR', required=True)
    required.add_argument('-l', '--label', help='Label to be predicted', metavar='str', required=True)
    required.add_argument('-a', '--algorithms', help='Algorithm for feature selection {ANOVA|MI|RFECV_LR|RFECV_LR_L1|RFECV_LR_L2|RFECV_LR_EN|RFECV_Perceptron|RFECV_RF|RFECV_GB|RFECV_SVM|Agglomerative|Lasso|ElasticNet|all}', required=True, nargs='+', metavar='str')

    additionals=parser_fs.add_argument_group('Additional arguments')
    additionals.add_argument('-t', '--threads', help='Number of threads for embedded feature selection method [20]', type=int, default=10, metavar='int')
    additionals.add_argument('-m', '--metric', help='Metric for embedded feature selection method {f1|precision|recall|roc_auc} [f1]', choices=["f1","precision","recall","roc_auc"], default="f1", type=str, metavar='str')
    additionals.add_argument('-v', '--verbose', help='Enable scikit-learn Warnings messages [False]', action='store_true')
    additionals.add_argument('--metadata', help='TSV file contained label to be predicted', metavar='TSV')
    additionals.add_argument('--additional', help='TSV containg additional covariates for training', metavar='TSV')
    additionals.add_argument('--seed', help='Seed for Cross-Validation [None]', metavar="int", type=int, default=None)
    additionals.add_argument('--n_splits', help='Splits for CV strategy [3]', metavar="int", type=int, default=3)
    additionals.add_argument('--n_repeats', help='Repeats for CV strategy [10]', metavar="int", type=int, default=10)

    parser_fs.set_defaults(func=run_subtool)


    ## FeatX ##


    parser_featx = subparsers.add_parser('FeatX', help='Feature Selection using custom rank algorithm')

    required = parser_featx.add_argument_group('Required I/O arguments')

    required.add_argument('-i', '--input', help='TSV file contained pyradiomics feature-extracted for training dataset', metavar='TSV', required=True)
    required.add_argument('-o', '--output', help='Output folder', metavar='DIR', required=True)
    required.add_argument('-l', '--label', help='Label to be predicted', metavar='str', required=True)
    required.add_argument('-a', '--algorithms', help='Algorithms for feature selection {ANOVA|MI|RFECV_LR|RFECV_LR_L1|RFECV_LR_L2|RFECV_LR_EN|RFECV_Perceptron|RFECV_RF|RFECV_GB|RFECV_SVM|Agglomerative|Lasso|ElasticNet|all [all]}',default=['all'],nargs='+')
    
    additionals=parser_featx.add_argument_group('Additional arguments')
    additionals.add_argument('-t', '--threads', help='Number of threads for embedded feature selection method [20]', type=int, default=10, metavar='int')
    additionals.add_argument('-m', '--metric', help='Metric for embedded feature selection method {f1|precision|recall|roc_auc} [f1]', choices=["f1","precision","recall","roc_auc"], default="f1", type=str, metavar='str') 
    additionals.add_argument('-n', '--n-max-allowed-features', help='Maximum number of allowed features [20]', default=20, type=int)
    additionals.add_argument('-r', '--rank', help='Maximum allowed rank to retain a feature [20]', default=20, type=int)
    additionals.add_argument('-s', '--supporting-algorithm', help='Maximum number of supporting feature selection algorithm [3]', default=3, type=int)
    additionals.add_argument('--threshold', help='scoring threshold to take model into consideration [0.6]', default=0.6, type=float)
    additionals.add_argument('-v', '--verbose', help='Enable scikit-learn Warnings messages [False]', action='store_true')
    additionals.add_argument('--metadata', help='TSV file contained label to be predicted', metavar='TSV')
    additionals.add_argument('--additional', help='TSV containg additional covariates for training', metavar='TSV')
    additionals.add_argument('--seed', help='Seed for Cross-Validation [None]', metavar="int", type=int, default=None)
    additionals.add_argument('--n_splits', help='Splits for CV strategy [3]', metavar="int", type=int, default=3)
    additionals.add_argument('--n_repeats', help='Repeats for CV strategy [10]', metavar="int", type=int, default=10)

    parser_featx.set_defaults(func=run_subtool)

    ## HyPerTune ##


    parser_hypertune = subparsers.add_parser('HyPerTune', help='HyPerparameters Tuning for Machine Learning Algorithm')

    required = parser_hypertune.add_argument_group('Required I/O arguments')

    required.add_argument('-i', '--input', help='TSV file contained pyradiomics feature-extracted for training dataset', metavar='TSV', required=True)
    required.add_argument('-f', '--feature-selection', help='TSV file contained the feature selection results', metavar='TSV', required=True)
    required.add_argument('-l', '--label', help='Label to be predicted', metavar='str', required=True)
    required.add_argument('-j', '--json', help='JSON file with Machine Learning algorithm parameters to tune', metavar='JSON')    
    required.add_argument('-o', '--output', help='Output folder', metavar='DIR', required=True)
    required.add_argument('-s', '--search', help=' Whether tune with Grid or Randomized search CV {GridSearchCV|RandomizedSearchCV}', choices=["GridSearchCV","RandomizedSearchCV"], metavar='str', required=True)

    additionals=parser_hypertune.add_argument_group('Additional arguments')
    additionals.add_argument('--threshold', help='Minimun score required to write a ML model [0.9]', type=float, default=0.9, metavar='float')
    additionals.add_argument('-t', '--threads', help='Number of threads for embedded feature selection method [20]', type=int, default=10, metavar='int')
    additionals.add_argument('-m', '--metric', help='Metric for hyperparameters tuning {f1|precision|recall|roc_auc} [f1]', choices=["f1","precision","recall","roc_auc"], default="f1", type=str,metavar='str')
    additionals.add_argument('--metadata', help='TSV file contained label to be predicted', metavar='TSV')
    additionals.add_argument('--additional', help='TSV containg additional covariates for training', metavar='TSV')
    additionals.add_argument('-v', '--verbose', help='Enable scikit-learn Warnings messages [False]', action='store_true')
    additionals.add_argument('--seed', help='Seed for GridSearchCV [None]', metavar="int", type=int, default=None)
    additionals.add_argument('--n_splits', help='Splits for CV strategy [3]', metavar="int", type=int, default=3)
    additionals.add_argument('--n_repeats', help='Repeats for CV strategy [10]', metavar="int", type=int, default=10)

    parser_hypertune.set_defaults(func=run_subtool)

    ## PREDICT ##


    parser_predict = subparsers.add_parser('PREDICT', help='Prediction on test dataset')

    required = parser_predict.add_argument_group('Required I/O arguments')

    required.add_argument('-i', '--input', help='TSV file contained pyradiomics feature-extracted for training dataset', metavar='TSV', required=True)
    required.add_argument('-v', '--validation', help='TSV file contained transformed feature-extracted for test dataset', metavar='TSV', required=True)
    required.add_argument('-f', '--feature-selection', help='TSV file contained the feature selection results', metavar='TSV', required=True)
    required.add_argument('-o', '--output', help='Output folder', metavar='DIR', required=True)
    required.add_argument('-l', '--label', help='Label to be predicted', metavar='str', required=True)
    required.add_argument('-d', '--dir', help='Folder in which the result of HyPerTune is stored', metavar='DIR', required=True)
    required.add_argument('-mv', '--metadata_validation', help='TSV file contained label to be predicted in the test dataset', metavar='DIR', required=True)

    additionals=parser_predict.add_argument_group('Additional arguments')
    additionals.add_argument('-mt', '--metadata_training', help='TSV file contained label to be predicted in the training', metavar='DIR')
    additionals.add_argument('--additional_training', help='TSV containg additional covariates for training', metavar='TSV')
    additionals.add_argument('--additional_validation', help='TSV containg additional covariates for validation', metavar='TSV')    
    additionals.add_argument('--permutation', help='Apply a permutation test for each classifier [False]', action='store_true')
    additionals.add_argument('--n_perm', help='Number of permutations [100]', metavar='int', type=int, default=100)
    additionals.add_argument('--verbose', help='Enable scikit-learn Warnings messages [False]', action='store_true')
    additionals.add_argument('--seed', help='Seed for GridSearchCV [None]', metavar="int", type=int, default=None)
    additionals.add_argument('--n_splits', help='Splits for CV strategy [3]', metavar="int", type=int, default=3)
    additionals.add_argument('--n_repeats', help='Repeats for CV strategy [10]', metavar="int", type=int, default=10)

    parser_predict.set_defaults(func=run_subtool)

    ## ENSEMBLE ##


    parser_ensemble = subparsers.add_parser('ENSEMBLE', help='Ensemble of the top performing workflows by averaging their posterior probabilities')

    required = parser_ensemble.add_argument_group('Required I/O arguments')

    required.add_argument('-i', '--input', help='TSV file with the top performing workflows', metavar='TSV', required=True)
    required.add_argument('-o', '--output', help='Output folder', metavar='DIR', required=True)
    required.add_argument('-mv', '--metadata_validation', help='TSV file contained label to be predicted in the test dataset', metavar='DIR', required=True)
    required.add_argument('-l', '--label', help='Label to be predicted', metavar='str', required=True)
    required.add_argument('-v', '--validation', help='TSV file contained transformed feature-extracted for test dataset', metavar='TSV', required=True)
    parser_ensemble.set_defaults(func=run_subtool)


    ## SURV ##


    parser_surv = subparsers.add_parser('SURV', help='Survival Analysis using Machine Learning algorithms')

    required = parser_surv.add_argument_group('Required I/O arguments')

    required.add_argument('-i', '--input', help='TSV file contained pyradiomics feature-extracted for training dataset', metavar='TSV', required=True)
    required.add_argument('--validation', help='TSV file contained pyradiomics feature-extracted for test dataset', metavar='TSV', required=True)
    required.add_argument('-o', '--output', help='Output folder', metavar='DIR', required=True)
    required.add_argument('-mv', '--metadata_validation', help='TSV file contained label to be predicted in the test dataset', metavar='DIR', required=True)
    required.add_argument('-mt', '--metadata_training', help='TSV file contained label to be predicted in the training dataset', metavar='DIR', required=True)
    required.add_argument('-s', '--search', help=' Whether tune with Grid or Randomized search CV {GridSearchCV|RandomizedSearchCV}', choices=["GridSearchCV","RandomizedSearchCV"], metavar='str', required=True)
    required.add_argument('-j', '--json', help='JSON file with Machine Learning algorithm parameters to tune', metavar='JSON')    
    
    additionals=parser_surv.add_argument_group('Additional arguments')
    additionals.add_argument('-t','--transformation' ,help='Transformation to apply to the dataset {scaler|normalize|mixture|robust|yeo-johnson|box-cox|quantile-uniform|quantile-normal}',choices=["scaler","normalize","mixture","robust","yeo-johnson","box-cox","quantile-uniform","quantile-normal"],metavar='str')
    additionals.add_argument('--threads', help='Number of threads for embedded feature selection method [20]', type=int, default=10, metavar='int')
    additionals.add_argument('--seed', help='Seed for GridSearchCV [None]', metavar="int", type=int, default=None)
    additionals.add_argument('--additional_training', help='TSV containg additional covariates for training', metavar='TSV')
    additionals.add_argument('--additional_validation', help='TSV containg additional covariates for validation', metavar='TSV')    
    additionals.add_argument('--n_splits', help='Splits for CV strategy [3]', metavar="int", type=int, default=3)
    additionals.add_argument('--n_repeats', help='Repeats for CV strategy [5]', metavar="int", type=int, default=5)
    additionals.add_argument('--n-quantiles', help='Number of quantiles for Quantile Transformation',type=int, default=100)
    additionals.add_argument('--robust-parameter', help='Quantile range for robust scaler', type=parse_robust_parameter, metavar='VAL1,VAL2', default=[25.0,75.0])
    additionals.add_argument('-f', '--feature-selection', help='TSV file contained the feature selection results', metavar='TSV')
    additionals.add_argument('--verbose', help='Enable scikit-survival Warnings messages [False]', action='store_true')
    additionals.add_argument('--archetypes', help='Enable archetypes analysis to classify patients in risk classes [False]', action='store_true')
    parser_surv.set_defaults(func=run_subtool)



    #print help if no subcommand nor --help provided

    if len(sys.argv)==1:

        print(fr"""     
    
    ██████╗ ██╗   ██╗██████╗  █████╗ ███╗   ███╗██╗██████╗ 
    ██╔══██╗╚██╗ ██╔╝██╔══██╗██╔══██╗████╗ ████║██║██╔══██╗
    ██████╔╝ ╚████╔╝ ██████╔╝███████║██╔████╔██║██║██║  ██║
    ██╔═══╝   ╚██╔╝  ██╔══██╗██╔══██║██║╚██╔╝██║██║██║  ██║
    ██║        ██║   ██║  ██║██║  ██║██║ ╚═╝ ██║██║██████╔╝
    ╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═════╝  v{__version__}                                                          
    
    """)
        
        parser.print_help(sys.stderr)
        sys.exit(1)

    #case-insensitive submodules
    
    if sys.argv[1].lower() == 'datrax':

        sys.argv[1] = 'DaTrax'

    elif sys.argv[1].lower() == 'slic':

        sys.argv[1] = 'SLIC'

    elif sys.argv[1].lower() == 'fs':

        sys.argv[1] = 'FS'

    elif sys.argv[1].lower() == 'featx':

        sys.argv[1] = 'FeatX'

    elif sys.argv[1].lower() == 'hypertune':

        sys.argv[1] = 'HyPerTune'

    elif sys.argv[1].lower() == 'predict':

        sys.argv[1] = 'PREDICT'

    elif sys.argv[1].lower() == 'ensemble':

        sys.argv[1] = 'ENSEMBLE'

    elif sys.argv[1].lower() == 'surv':

        sys.argv[1] = 'SURV'
    
    args = parser.parse_args()
    args.func(parser, args)


class CustomFormat(HelpFormatter):

    '''
    Customize how help is diplayed
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

def parse_robust_parameter(value):

    '''
    Parse command line when Robust transform is setted
    '''

    try:
        return [float(x) for x in value.split(',')]
    
    except ValueError:
        
        raise argparse.ArgumentTypeError("The --robust-parameter argument must contain two numeric values separated by a comma.")


def run_subtool(parser, args):


    if args.command == 'DaTrax': 

        from .DaTrax import DaTrax as submodule

        if args.transformation != "robust" and 'robust_parameter' in args.__dict__ and args.__dict__['robust_parameter'] != [25.0, 75.0]:
            
            sys.stderr.write('Warning: --robust-parameter is ignored because --transformation is not set to "robust".\n')

        # Check if `--n-quantiles` was explicitly provided by the user
        if args.transformation not in ["quantile-uniform", "quantile-normal"] and 'n_quantiles' in args.__dict__ and args.__dict__['n_quantiles'] != 100:
            
            sys.stderr.write('Warning: --n-quantiles is ignored because --transformation is not set to "quantile-uniform" or "quantile-normal".\n')
    
    elif args.command == 'SLIC': 

        from .SLIC import SLIC as submodule

    elif args.command == 'FS': 

        from .FS import FS as submodule

    elif args.command == 'FeatX':

        from .FeatX import FeatX as submodule

    elif args.command == 'HyPerTune':

        from .HyPerTune import HyPerTune as submodule

    elif args.command == 'PREDICT':

        from .PREDICT import PREDICT as submodule

        if args.permutation and args.n_perm is None:
    
            parser.error("--n_perm is required when --permutation is set.")

    elif args.command == 'ENSEMBLE':

        from .ENSEMBLE import ENSEMBLE as submodule

    elif args.command == 'SURV':

        from .SURV import SURV as submodule

    else:

        parser.print_help()
        return

    submodule.run(parser,args)


if __name__ =='__main__':

    main()