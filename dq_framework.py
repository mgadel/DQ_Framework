"""
General Framework for data quality pipeline
"""

import logging
from datetime import datetime
import pandas as pd
from scr.utils.utils_data_quality import generate_dq_report
from scr.utils.utils_main import save_versioned_dataset
from scr.dq_monitoring import DataDomainMonitoring


def _check_column_presence(func):
    """ Create decorator """
    def wrapper(self,column_name,*args, **kwargs):
        """ check column presence """
        if column_name is not None and column_name not in self.dataset.columns.to_list():
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        return func(self,column_name,*args, **kwargs)
    return wrapper


def _add_to_log(module,step_name,description,n_error,shape_out):
    """ log information from debugging purpose """

    logging.info(
            f"\nModule: {module}\n"
            f"Step: {step_name}\n"
            f"    - Description: {description}\n"
            f"    - Nb line error: {n_error}\n"
            f"    - Shape output: {shape_out}"
            )

    return


class DataQualityFramework():
    """ Framework to manage DQ """
    def __init__(self):
        self.output_name  = ''

        self.dataset = pd.DataFrame()
        self.original_dataset = pd.DataFrame()
        self.final_dataset = pd.DataFrame()
        self.dataset_features = []

        # on log les erreurs qui pourraient amener des erreurs dans le pipeline (NA, duplicate...)
        self.list_error_features = []
        self.corruption_errors = pd.DataFrame()

        # pour le domain monitoring
        self.domain_monitoring = None
        self.data_window = None

        self.dict_pipeline_structure = {
            "Structure_Order": [("Init", "Init")],  # (module, step)
            "Modules":{
                "Init":{
                    "Step":[],
                    "Description":[],
                    "Shape_out": [],
                    "Nb_Error": [],
                },
                'Data_Cleaning':{
                    "Step":[],
                    "Description":[],
                    "Shape_out": [],
                    "Nb_Error": []
                },
                'Create_Feature':{
                    "Step":[],
                    "Description":[],
                    "Shape_out": [],
                    "Nb_Error": []
                },
                'DQ_Rule':{
                    "Step":[],
                    "Description":[],
                    "Shape_out": [],
                    "Nb_Error": []
                },
                "Output":{
                    "Step":[],
                    "Description":[],
                    "Shape_out": [],
                    "Nb_Error": []
                }
            }
        }

        # Set up logging
        logging.basicConfig(
            filename=('reports/processing/data_processing_report_'
                     f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s')


    def init_pipeline(self,dataset,output_name):
        """initialize data pipeline"""
        self.output_name = output_name

        self.original_dataset = dataset.copy()

        self.dataset = dataset.copy()
        self.dataset_features = dataset.columns.to_list()

        self.add_step_to_pipeline("Init","Init","Input raw data")
        print(f"Raw dataset shape: {dataset.shape}")

        return self


    def terminate_pipeline(self, save_db_for_profiling: bool = True, dq_report: bool = True):
        """ 
        close data pipeline with:
        - adding last error field
        - log output info
        """

        self.add_high_level_pipeline_error_flag()

        if save_db_for_profiling:
            self.save_dataset_for_traceability_and_profiling()

        if dq_report:
            generate_dq_report(
                self.dataset,
                self.list_error_features,
                self.dict_pipeline_structure,
                self.output_name
                )

        print(f"Output dataset_all shape: {self.dataset.shape}")
        print(f"Output dataset_clean shape: {self.dataset[~self.dataset['errors']].shape}")

        return self


    def add_high_level_pipeline_error_flag(self):
        """ Create a new error that flag True when at least one error is true"""
        # Create a general "error" flag and save the list of feature related to errors
        self.list_error_features = list(
            self.dict_pipeline_structure["Modules"]["Data_Cleaning"]["Step"] +
            self.dict_pipeline_structure["Modules"]["DQ_Rule"]["Step"])
        self.dataset['errors'] = self.dataset[self.list_error_features].any(axis=1)
        self.list_error_features.append("errors")

        self.add_step_to_pipeline(
            module="Output",
            step_name="errors",
            description=("High level error flag. If error is True,"
                         " at least one errror is true for this row"),
            )

        return self


    def add_step_to_pipeline(self,module,step_name,description):
        """ add step name and step desctiption in each modules of the pipeline """

        if module in ["Init","Data_Cleaning", "Create_Feature", "DQ_Rule","Output"]:

            self.dict_pipeline_structure["Structure_Order"].append((module,step_name))
            self.dict_pipeline_structure["Modules"][module]["Step"].append(step_name)
            self.dict_pipeline_structure["Modules"][module]["Description"].append(description)

            n_error = 0
            if module in ["Init","Create_Feature"]:
                self.dict_pipeline_structure["Modules"][module]["Nb_Error"].append(0)
            else:
                n_error = self.dataset[self.dataset[step_name]].shape[0]
                self.dict_pipeline_structure["Modules"][module]["Nb_Error"].append(n_error)

            shape_out = self.dataset.shape
            self.dict_pipeline_structure["Modules"][module]["Shape_out"].append(shape_out)

            _add_to_log(module,step_name,description,n_error,shape_out)

        else:
            raise ValueError(
                f"Module name is {module} and should be in "
                "[Data_Cleaning, Create_Feature, DQ_Rule]")

        return self


    def expect_row_to_not_be_corrupted(self, drop_values: bool = True):
        """
        Drop lines where values are all NA
        """
        self.dataset['error_corrupted_rows'] = self.dataset.isna().all(axis=1)

        self.add_step_to_pipeline(
            module="Data_Cleaning",
            step_name='error_corrupted_rows',
            description='Drop rows where all values are equal to NA'
            )

        # as we drop value for pipeline integrity security, we log error in DF for debugging
        if drop_values:
            self.corruption_errors = pd.concat(
                [self.corruption_errors,self.dataset[self.dataset['error_corrupted_rows']]],
                axis=0)
            self.dataset = self.dataset.dropna(how = 'all')

        return self


    def expect_rows_to_be_unique(self, drop_values: bool = True):
        """
        on gere les lignes entieres dupliquées
        """
        self.dataset['error_duplicated_rows'] = self.dataset.duplicated()

        self.add_step_to_pipeline(
            module="Data_Cleaning",
            step_name='error_duplicated_rows',
            description='Expect rows to be unique'
            )

        # as we drop value for pipeline integrity security, we log error in DF for debugging
        if drop_values:
            self.corruption_errors = pd.concat(
                [self.corruption_errors,self.dataset[self.dataset['error_duplicated_rows']]],
                axis=0)

            self.dataset = self.dataset.drop_duplicates()

        return self


    @_check_column_presence
    def expect_column_values_to_be_not_na(self, column_name: str, drop_values: bool = True):
        """expect values in a column to be not NA"""
        self.dataset[f'error_{column_name}_is_NA'] = self.dataset[column_name].isna()

        self.add_step_to_pipeline(
            module="Data_Cleaning",
            step_name=f'error_{column_name}_is_NA',
            description=f'Expect {column_name} values not to be NA'
            )

        # as we drop value for pipeline integrity security, we log error in DF for debugging
        if drop_values:
            self.corruption_errors = pd.concat(
                [self.corruption_errors,self.dataset[self.dataset[f'error_{column_name}_is_NA']]],
                axis=0)

            self.dataset = self.dataset.dropna(subset=[column_name])


        return self

    @_check_column_presence
    def expect_column_values_to_be_unique(self,  column_name: str):
        """expect all value in column to be unique"""
        self.dataset[f'error_{column_name}_duplicated'] = (
            self.dataset[column_name].duplicated(keep=False))

        self.add_step_to_pipeline(
            module="Data_Cleaning",
            step_name=f'error_{column_name}_duplicated',
            description=f'Expect {column_name} value to be unique'
            )

        return self

    @_check_column_presence
    def expect_column_value_to_not_be_null(self, column_name: str):
        """ Expect values in a column to be not null """
        self.dataset[f'error_{column_name}_is_null'] = self.dataset[column_name] == 0

        self.add_step_to_pipeline(
            module="Data_Cleaning",
            step_name=f'error_{column_name}_is_null',
            description=f'Expect {column_name} values not to be null'
            )

        return self

    @_check_column_presence
    def expect_column_values_to_be_in_set(self, column_name: str, value_set: list,
                                          drop_values: bool = True):
        """ Expect categorical values of a column to be in a set """

        self.dataset[f'error_{column_name}_outside_set'] = (
            ~ self.dataset[column_name].isin(value_set))

        self.add_step_to_pipeline(
            module="Data_Cleaning",
            step_name=f'error_{column_name}_outside_set',
            description=f'Expect {column_name} values to be in set {value_set}'
            )

        # as we drop value for pipeline integrity security, we log error in DF for debugging
        if drop_values:
            self.corruption_errors = pd.concat(
                [self.corruption_errors,
                 self.dataset[self.dataset[f'error_{column_name}_outside_set']]],
                axis=0)

            self.dataset = self.dataset[self.dataset[column_name].isin(value_set)]

        return self

    @_check_column_presence
    def expect_column_values_to_be_of_type(self, column_name: str, type_: type):
        """ Expect values in a column to be of a given type """

        self.dataset[column_name] = self.dataset[column_name].astype(type_)

        self.dataset[f'error_{column_name}_type'] = (
            ~ self.dataset[column_name].apply(lambda x: isinstance(x, type_)))

        type_str = str(type_).split("'")[1]
        self.add_step_to_pipeline(
            module="Data_Cleaning",
            step_name=f'error_{column_name}_type',
            description=f"Expect {column_name} value to be of type {type_str}"
            )

        return self

    @_check_column_presence
    def expect_column_value_to_be_below(self, column_name: str, max_value:float,
                                        inclusive:bool = True):
        """ Expect values in a column to be below a threshold """
        if inclusive:
            inclusive_string = 'inclusive'
            self.dataset[f'error_{column_name}_value_above_range'] = (
                ~ (self.dataset[column_name] <= max_value))

        else:
            inclusive_string = 'exclusive'
            self.dataset[f'error_{column_name}_value_above_range'] = (
                ~ (self.dataset[column_name] < max_value))

        self.add_step_to_pipeline(
            module="DQ_Rule",
            step_name=f'error_{column_name}_value_above_range',
            description=f'Expect {column_name} values to be less {max_value} {inclusive_string}'
            )

        return self

    @_check_column_presence
    def expect_column_value_to_be_above(self, column_name: str, min_value:float,
                                        inclusive:bool = True):
        """ Expect values in a column to be above a threshold """
        if inclusive:
            inclusive_string = 'inclusive'
            self.dataset[f'error_{column_name}_value_below_range'] = (
                ~ (self.dataset[column_name] >= min_value))
        else:
            inclusive_string = 'exclusive'
            self.dataset[f'error_{column_name}_value_below_range'] = (
                ~ (self.dataset[column_name] > min_value))

        self.add_step_to_pipeline(
            module="DQ_Rule",
            step_name=f'error_{column_name}_value_below_range',
            description=f'Expect {column_name} value to be above {min_value} {inclusive_string}'
            )

        return self

    @_check_column_presence
    def expect_column_value_to_be_between(self, column_name: str, min_value:float,
                                          max_value:float, inclusive:bool = False):
        """ Expect values in a column to be between two threshold """
        if inclusive:
            inclusive_string = 'inclusive'
            self.dataset[f'error_{column_name}_value_outside_range'] = ~ (
                (self.dataset[column_name] <= max_value) &
                (self.dataset[column_name] >= min_value))
        else:
            inclusive_string = 'exclusive'
            self.dataset[f'error_{column_name}_value_outside_range'] = ~ (
                (self.dataset[column_name] < max_value) &
                (self.dataset[column_name] > min_value))

        self.add_step_to_pipeline(
            module="DQ_Rule",
            step_name=f'error_{column_name}_value_outside_range',
            description=f"Expect {column_name} values to be between"
                        f" {min_value} {max_value} {inclusive_string}"
            )

        return self

    @_check_column_presence
    def expect_column_values_to_be_within_n_stdevs(self, column_name: str, n: int,
                                                   inclusive: bool=True):
        """ Expect values in a column to within n standards deviations """
        mean = self.dataset[column_name].mean()
        std = self.dataset[column_name].std()

        lower_bound = mean - n * std
        upper_bound = mean + n * std

        if inclusive:
            inclusive_string = 'inclusive'
            self.dataset[f'error_{column_name}_outside_{n}_std_range'] = ~ (
                (self.dataset[column_name] <= upper_bound) &
                (self.dataset[column_name] >= lower_bound))
        else:
            inclusive_string = 'exclusive'
            self.dataset[f'error_{column_name}_outside_{n}_std_range'] = ~ (
                (self.dataset[column_name] < upper_bound) &
                (self.dataset[column_name] > lower_bound))

        self.add_step_to_pipeline(
            module="DQ_Rule",
            step_name=f'error_{column_name}_outside_{n}_std_range',
            description=f"Expect {column_name} values to be between"
                        f" {lower_bound} {upper_bound} {inclusive_string}"
            )

        return self

    @_check_column_presence
    def expect_column_sum_to_be_between(self, column_name: str, min_value:float , max_value:float):
        """check column sum to be between """
        self.dataset[f'error_{column_name}_sum_outside_range'] = not (
            (self.dataset[column_name].sum() <= max_value) &
            self.dataset[column_name].sum() >= min_value)

        self.add_step_to_pipeline(
            module="DQ_Rule",
            step_name=f'error_{column_name}_sum_outside_range',
            description=f'Expect {column_name} sum to be between {min_value} {max_value}'
            )

        return self

    @_check_column_presence
    def expect_column_mean_to_be_between(self, column_name: str, min_value:float , max_value:float):
        """check column mean to be between """
        self.dataset[f'error_{column_name}_mean_outside_range'] = not (
            (self.dataset[column_name].mean() <= max_value) &
            (self.dataset[column_name].mean() >= min_value))

        self.add_step_to_pipeline(
            module="DQ_Rule",
            step_name=f'error_{column_name}_mean_outside_range',
            description=f'Expect {column_name} mean to be between {min_value} {max_value}'
            )

        return self

    @_check_column_presence
    def expect_column_median_to_be_between(self, column_name: str, min_value:float,
                                           max_value:float):
        """check column median to be between """
        self.dataset[f'error_{column_name}_median_outside_range'] = not (
            (self.dataset[column_name].median() <= max_value) &
            (self.dataset[column_name].median() >= min_value))

        self.add_step_to_pipeline(
            module="DQ_Rule",
            step_name=f'error_{column_name}_median_outside_range',
            description=f'Expect {column_name} median to be between {min_value} {max_value}'
            )

    @_check_column_presence
    def expect_column_stdev_to_be_between(self, column_name: str, min_value:float ,
                                          max_value:float):
        """check column standard deviation to be between """
        self.dataset[f'error_{column_name}_stdev_outside_range'] = not (
            (self.dataset[column_name].std() <= max_value) &
            (self.dataset[column_name].std() >= min_value))

        self.add_step_to_pipeline(
            module="DQ_Rule",
            step_name=f'error_{column_name}_stdev_outside_range',
            description=f'Expect {column_name} stdev to be between {min_value} {max_value}'
            )

        return self

    @_check_column_presence
    def expect_column_to_be_consistant_with_external_source(
        self,
        column_name: str,
        ext_dataset: pd.DataFrame):
        """ 
        cross checking of data between external data source of data
        """

        df_merged = self.dataset.merge(ext_dataset,on=['imonum',column_name],
                                how='outer',
                                suffixes=('_dataset','_external'),
                                indicator=True)

        self.add_step_to_pipeline(
            module="DQ_Rule",
            step_name=f'error_{column_name}_not_consistant_with_external_db',
            description=f'expect {column_name} to be consistant with external db values'
            )

        df_merged.to_csv(
                f'errors/error_logs/error_{column_name}.csv',
                index=False)

        return self


    def add_new_cleaning_step(self, error_name, description, error_value):
        """ Helper function to create cleaning set and log informations """

        # handle definition of step name
        if error_name.startswith("error_"):
            step_name = error_name
        else:
            step_name = "error_" + error_name

        if error_value is not None:
            self.dataset[step_name] = error_value
        else:
            pass

        self.add_step_to_pipeline(
            module="Data_Cleaning",
            step_name=step_name,
            description=description
            )

        return self


    def add_new_dq_rule_step(self, dq_name, description, dq_value):
        """ Helper function to create DQ Rule and log informations """

        # handle definition of step name
        if dq_name.startswith("error_"):
            step_name = dq_name
        else:
            step_name = "error_" + dq_name

        if dq_value is not None:
            self.dataset[step_name] = dq_value
        else:
            pass

        # DQ rules and cleaning steps should begin with Error name
        self.add_step_to_pipeline(
            module="DQ_Rule",
            step_name=step_name,
            description=description
            )

        return self


    def add_new_feature_step(self, feature_name, description, feature_value):
        """ Helper function to create feature and log informations """
        if feature_value is not None:
            self.dataset[feature_name] = feature_value
        else:
            pass

        self.add_step_to_pipeline(
            module="Create_Feature",
            step_name=feature_name,
            description=description
            )

        return self


    def initialize_domain_monitoring(self,reference_dataset,results_path,columns,data_window):
        """ initialize Domain Monotirng for production vs modelling dataset"""
        self.domain_monitoring = DataDomainMonitoring(reference_dataset,results_path)
        self.domain_monitoring.set_columns_for_domain_monitoring(columns)
        self.data_window = data_window

        return self


    def run_domain_analysis(self, production_dataset: pd.DataFrame):
        """ run analysis for production domain """
        if self.domain_monitoring is not None:
            self.domain_monitoring.data_shift_analysis(production_dataset,self.data_window)
        else:
            raise ValueError("Initialize Domain Monitoring first")

        return self




    def save_dataset_for_traceability_and_profiling(self):
        """ Save DB for traceability or for user action (profiling) """

        # save full dataset with track of errors in error column
        save_versioned_dataset(
            self.dataset,'data/intermediate',f'{self.output_name}_full', False)

        self.dataset[~self.dataset['errors']].to_csv(
            path_or_buf = f'errors/error_logs/{self.output_name}_error_flag.csv',
            index=False)

        # save errors for debug
        self.corruption_errors.to_csv(
            path_or_buf = f'errors/error_logs/{self.output_name}_corrupted_error.csv',
            index=False)

        return self
