""" Class That implement domain monitoring and drift control """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon



class DataDomainMonitoring():
    """
    Class that define the Domain Monitoring of input production data. 
    Production data are checked against the dataset that was used to 
    build the model.

    Simple checks are first perform to ensure data are part of the modelling domain:
    - min < DATA < max
    - Model_specific filters

    As we mostly deal with single points, the single points are stored in a 
    production_data.csv with the date of the reporting. 
    That enables to perform data drift monitoring against
    """

    def __init__(self, reference_dataset: pd.DataFrame, results_path):
        self._reference_dataset = pd.DataFrame()
        self.set_reference_dataset(reference_dataset)

        self.production_dataset_all = pd.DataFrame()
        self.production_dataset_window = pd.DataFrame()

        self.numerical_columns= []
        self.categorical_columns = []

        # as we deal mostly with single points,
        # set up the number of point to perform distribution analysis
        self.data_window = None
        self.path_results_save = results_path


    def set_reference_dataset(self, dataset: pd.DataFrame):
        """ setter for reference dataset """
        self._reference_dataset = dataset


    def set_production_dataset(self, dataset: pd.DataFrame):
        """ setter for production dataset """
        self.production_dataset_all = dataset


    def set_columns_for_domain_monitoring(self, columns_analysis: list):
        """ set columns that will be analysed """
        self.categorical_columns = (self._reference_dataset[columns_analysis]
                                    .select_dtypes('object')
                                    .columns.to_list())

        all_numerical_columns = (self._reference_dataset[columns_analysis]
                                  .select_dtypes('number')
                                  .columns.to_list())

        for column in all_numerical_columns:
            if column == 'const':
                pass
            elif column.startswith('log_'):
                self.numerical_columns.append(column)
                if column[4:] in self._reference_dataset.columns.to_list():
                    self.numerical_columns.append(column[4:]) # on ajoute aussi les colones sans log
            else:
                self.numerical_columns.append(column)

        return self


    def plot_numerical_shift(self):
        """ Plots for distribution shift"""

        n_cols = min(5,len(self.numerical_columns))
        # Calculate rows based on the number of features
        n_rows = -(-len(self.numerical_columns) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4 * n_rows))
        fig.tight_layout(pad=3.0)
        fig.subplots_adjust(top=0.90)
        fig.suptitle('Feature Distributions: productuion data vs réference data',size=16)
        axes = axes.flatten()

        for i, feature in enumerate(self.numerical_columns):
            sns.kdeplot(self._reference_dataset[feature], label='Reference Dataset',
                        color='blue', fill=True, alpha=0.4, ax=axes[i],warn_singular=False)
            sns.kdeplot(self.production_dataset_window[feature], label='Production Dataset',
                        color='orange', fill=True, alpha=0.6, ax=axes[i],warn_singular=False)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Density')
            axes[i].legend()

        # Hide unused subplots
        for j in range(len(self.numerical_columns), len(axes)):
            fig.delaxes(axes[j])

        fig.savefig(self.path_results_save + '/feature_distribution.png',
                    dpi=300, bbox_inches='tight')


        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4 * n_rows))
        fig.tight_layout(pad=3.0)
        fig.subplots_adjust(top=0.90)
        fig.suptitle('Cumulative Probability: productuion data vs réference data',size=16)
        axes = axes.flatten()

        for i, feature in enumerate(self.numerical_columns):
            sns.ecdfplot(self._reference_dataset[feature],
                         label='Reference Dataset', ax=axes[i], color='blue')
            sns.ecdfplot(self.production_dataset_window[feature],
                         label='Production Dataset', ax=axes[i], color='orange')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Cumulative Probability')
            axes[i].legend()

        # Hide unused subplots
        for j in range(len(self.numerical_columns), len(axes)):
            fig.delaxes(axes[j])

        fig.savefig(self.path_results_save + '/cumulative_probability.png',
                    dpi=300, bbox_inches='tight')

        return


    def numerical_distribution_shift(self, p_valuethres=0.05):
        """ monitoring of data shift """

        shift_for_feature = []
        features  = []
        p_values_error=[]
        ks_test_results =[]
        js_divergence=[]
        js_interpret = []
        wasser_distance=[]
        # wasser_interpret = []

        # Perform Kolmogorov-Smirnov test for each feature
        for feature in self.numerical_columns:

            feature_reference = self._reference_dataset[feature]
            feature_production = self.production_dataset_window[feature]

            test_results = ks_2samp(feature_reference, feature_production)

            features.append(feature)
            p_values_error.append(round(test_results.pvalue,3))
            ks_test_results.append(round(test_results.statistic,3))

            if test_results.pvalue < p_valuethres:
                shift_for_feature.append(True)
            else:
                shift_for_feature.append(False)

            bins = np.linspace(-4, 4, 50)
            hist_ref, _ = np.histogram(feature_reference, bins=bins, density=True)
            hist_prod, _ = np.histogram(feature_production, bins=bins, density=True)

            hist_ref += 1e-10
            hist_prod += 1e-10

            # Compute Jensen-Shannon Divergence
            js = jensenshannon(hist_ref, hist_prod)
            js_divergence.append(js)

            if js < 0.1:
                js_interpret.append('JS - Very similar')
            elif (js > 0.1) & (js < 0.5):
                js_interpret.append('JS - Moderately different')
            elif js > 0.5:
                js_interpret.append('JS - Very different')
            else:
                js_interpret.append('Not Applicable')

            wasser_distance.append(wasserstein_distance(
                np.ravel(feature_reference), np.ravel(feature_production)))

        results_shift = pd.DataFrame({
            "feature":features,
            "shift_detected_KS":shift_for_feature,
            'p_value':p_values_error,
            'KS Test': ks_test_results,
            'Jensen-Shannon Divergence':js_divergence,
            'JS_Interpret':js_interpret,
            'Wasserstein Distance':wasser_distance
        })

        results_shift.to_csv(self.path_results_save + '/results_shift.csv',index=False)

        return self



    def single_data_shift(self):
        """ perform data shift for one point"""
        percentile_rank = stats.percentileofscore(
            self._reference_dataset,self.production_dataset_window)

        empirical_cumulative_distribution_function = (
            self._reference_dataset <= self.production_dataset_window).mean()

        single_shift_results = pd.DataFrame({
            "percentile rank": percentile_rank,
            "empirical cumulative distribution function":empirical_cumulative_distribution_function 
            }, index = "single data shift")

        print(single_shift_results)

        return self


    def data_shift_analysis(self, production_dataset_all ,data_window = None):
        """ perform data shift analysis """
        if data_window is None:
            self.production_dataset_window = production_dataset_all
            self.production_dataset_all = production_dataset_all
        else:
            self.production_dataset_window = production_dataset_all.tail(data_window).copy()
            self.production_dataset_all = production_dataset_all

        if data_window == 1:
            self.single_data_shift()
        else:
            self.numerical_distribution_shift()
            self.plot_numerical_shift()

        return self
