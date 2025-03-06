""" modelling framework """

import pickle
import numpy as np
import pandas as pd
from statsmodels.api import add_constant

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scr.utils.utils_modelling import (root_mean_squared_error,
                                    mean_absolute_percentage_error,
                                    root_mean_squared_percentage_error)




def transform_function(y: pd.Series , log_model:bool = False):
    """handle log transformed functions"""
    if log_model is False:
        return y
    elif log_model is True:
        return np.exp(y)



class ModellingFramework():
    """ Class that set up modelling framework """
    def __init__(self,path_results):

        self.path_results = path_results

        self.model_repository = None
        self.best_model_name = None

        self.best_model = self.ModelInformation()
        # self.selected_model = self.ModelInformation()

        self.load_modelling_framework()
        self.best_model.load_model(self.model_repository,model_name ='best_model')


    class ModelInformation():
        """ class to score model and model related metadata"""
        def __init__(self):

            self.model_reference_dataset = None

            self.model_name = None
            self.model = None
            self.metadata = None

            self.X_cols = None
            self.y_col = None
            self.per_group = None
            self.model_type = None
            self.log_model = None
            self.y_true_training = None
            self.y_hat_estimation_training = None
            self.res_percents_training = None
            self.model_version = None
            self.dataset_version = None
            self.dataset_name = None
            self.ship_dwt_categories = None
            self.additional_data_domain_limitation = None

            self.model_modelling_features = []


        def load_model(self,model_repository,model_name='best_model'):
            """ load given model from /model """

            if model_name == 'best_model':
                self.model_name = model_repository['best_model_name']
            else:
                self.model_name = model_name

            self.model = model_repository['models'][self.model_name]
            self.metadata = model_repository['metadata'][self.model_name]
            self.model_modelling_features = (
                self.metadata['X_cols'] + [self.metadata['y_col']])

            self.model_reference_dataset = pd.read_csv(self.metadata["dataset_name"])

            for key, value in model_repository['metadata'][self.model_name].items():
                setattr(self,key,value)

            return self


    def load_modelling_framework(self,version='V0'):
        """ load modelling framework from /model """

        print("Version Handling in modelling framework should be coded")

        try:
            with open(f'models/bulks/models_repository_bulks_{version}.pkl', 'rb') as file:
                self.model_repository = pickle.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                'missing modelling framework at models/bulks/models_repository_bulks') from e

        self.best_model_name =self.model_repository['best_model_name']

        return self


    def get_implemented_models_name(self):
        """ getter for models informations"""
        return self.model_repository['models'].keys()


    def summarize(self,production_dataset,return_plots=True,print_metrics=True):
        """ summarize prediction with residuals plots and metrics """

        def summary_metrics(
                y_true,
                y_hat_pred,
                model_name,
                modelling,
                per_group,
                full_name,
                model_features_name,
                y_col):

            rmse = round(root_mean_squared_error(y_true, y_hat_pred),2)
            mae = round(mean_absolute_error(y_true, y_hat_pred),2)
            mape = round(mean_absolute_percentage_error(y_true, y_hat_pred),2)
            rmspe = round(root_mean_squared_percentage_error(y_true, y_hat_pred),2)

            results = pd.DataFrame({
                'Algo Name':model_name,
                'Modelling':modelling,
                'Full Name':full_name,
                'y_name':y_col,
                'features_name':[model_features_name],
                'group model':per_group,
                'RMSPE':rmspe,
                'MAE':mae,
                'RMSE':rmse,
                'MAPE':mape},
            )

            return results


        def df_residuals_init(dataset, cat_production_modelling="production"):

            df_residuals = pd.DataFrame(
                {'IMO':dataset['imonum'].tolist(),
                'MeanConso':dataset['Mean_Consumption_Sea_(kW)'].tolist(),
                'DWT':dataset['dead_wt_tonnage_corrected'].tolist(),
                'V':dataset['Mean_Speed_ReportingPeriod'].tolist(),
                'V**3':dataset['Mean_Speed_ReportingPeriod**3'].tolist(),
                'ReportingTimeMonth':round(
                    (dataset['reporting_period_hours']/24/30.4),1).tolist(),
                'Percentage_time_at_sea':round(
                    (dataset['Percentage_time_at_sea']),1).tolist(),
                'ConsoTotGWh':round(
                    (dataset['Consumption_tot_kWh']/1000000),2).tolist(),
                'Ratio_ConsoSea_ConsoTot':round(
                    dataset['Ratio_ConsoSea_ConsoTot'],2).tolist(),
                'ConsoSeaGWh':round(
                    (dataset['Consumption_Sea_kWh']/1000000),2).tolist(),
                'RatioAuxSea_TotAux':round(
                    dataset['RatioAuxSea_TotAux'],2).tolist(),
                'TotAuxConsoGWh':round(
                    (dataset['Tot_aux_and_boiler_conso_kWh']/1000000),2).tolist(),
                'TotAuxIdleConsoGWh':round(
                    ((dataset['Auxiliary_conso_other_kWh'] + dataset['Boiler_conso_other_kWh'])
                     /1000000),2).tolist(),
                'AuxConsoSeaGWh':round(
                    ((dataset['Auxiliary_conso_sea_kWh'])/1000000),2).tolist(),
                "Group_DWT_BV":dataset['Group_DWT_BV'].tolist(),
                "cat_production_modelling":(
                    [cat_production_modelling] * len(dataset['Group_DWT_BV'])),
                }, index=dataset.index)

            return df_residuals


        def plot_residuals(df_residuals):
            """ plot function residuals """

            df_residuals = df_residuals.sort_values(by="cat_production_modelling")

            f, ax = plt.subplots(2, 4, figsize=(20, 10))
            f.tight_layout(pad=3.0)
            f.subplots_adjust(top=0.90)
            f.suptitle('Residual plot: productuion data vs réference data in %',size=16)

            axi = ax[0,0]
            sns.scatterplot(x=df_residuals.DWT, y=df_residuals.res_percent, 
                            ax=axi,hue=df_residuals["cat_production_modelling"], alpha=0.8)
            axi.set_xlabel("DWT")
            axi.set_ylabel("Résidus %")

            axi = ax[0,1]
            sns.scatterplot(x=df_residuals.V, y=df_residuals.res_percent, ax=axi,
                            hue=df_residuals["cat_production_modelling"],alpha=0.8)
            axi.set_xlabel("V")
            axi.set_ylabel("Résidus %")

            axi = ax[0,2]
            sns.scatterplot(x=df_residuals['V**3'], y=df_residuals.res_percent, ax=axi,
                            hue=df_residuals["cat_production_modelling"],alpha=0.8)
            axi.set_xlabel("V**3")
            axi.set_ylabel("Résidus %")

            axi = ax[0,3]
            sns.scatterplot(x=df_residuals.Percentage_time_at_sea, y=df_residuals.res_percent,
                            ax=axi, hue=df_residuals["cat_production_modelling"],alpha=0.8)
            axi.set_xlabel("Percentage_time_at_sea")
            axi.set_ylabel("Résidus %")

            axi = ax[1,0]
            sns.scatterplot(x=df_residuals.Ratio_ConsoSea_ConsoTot, y=df_residuals.res_percent,
                            ax=axi, hue=df_residuals["cat_production_modelling"],alpha=0.8)
            axi.set_xlabel("RatioConsoSea_ConsoTot")
            axi.set_ylabel("Résidus %")

            axi = ax[1,1]
            sns.scatterplot(x=df_residuals.RatioAuxSea_TotAux, y=df_residuals.res_percent, ax=axi,
                            hue=df_residuals["cat_production_modelling"],alpha=0.8)
            axi.set_xlabel("RatioAuxSea_TotAux")
            axi.set_ylabel("Résidus %")

            axi = ax[1,2]
            sns.scatterplot(x=df_residuals.ConsoSeaGWh, y=df_residuals.res_percent, ax=axi,
                            hue=df_residuals["cat_production_modelling"],alpha=0.8)
            axi.set_xlabel("ConsoSeaGWh")
            axi.set_ylabel("Résidus %")

            axi = ax[1,3]
            sns.scatterplot(x=df_residuals.TotAuxConsoGWh, y=df_residuals.res_percent, ax=axi,
                            hue=df_residuals["cat_production_modelling"],alpha=0.8)
            axi.set_xlabel("TotAuxConsoGWh")
            axi.set_ylabel("Résidus %")

            f.savefig(self.path_results + '/residuals.png',dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

            return


        if self.best_model.model_type not in ['OLS_L2','Perso_RMSPE']:
            raise ValueError('model_type error, should be either OLS_L2 or Perso_RMSPE')

        df_residuals_prod = df_residuals_init(production_dataset)

        df_residuals_prod['y_pred'] = production_dataset['y_pred']

        df_residuals_prod['y_true'] = transform_function(production_dataset[self.best_model.y_col],
                                                         self.best_model.log_model)


        df_residuals_prod['res_percent'] = (
            (df_residuals_prod['y_true'] - df_residuals_prod['y_pred'])
            / df_residuals_prod['y_true']*100)


        df_residuals_modelling = df_residuals_init(self.best_model.model_reference_dataset,
                                                   "modelling")
        df_residuals_modelling['res_percent'] = self.best_model.res_percents_training

        if return_plots is True:
            all_residuals = pd.concat([df_residuals_prod,df_residuals_modelling],axis=0)
            plot_residuals(all_residuals)

        if print_metrics is True:

            results_summary_metrics = summary_metrics(
                                            y_true = df_residuals_prod['y_true'] ,
                                            y_hat_pred = df_residuals_prod['y_pred'],
                                            model_name = self.best_model.model_name ,
                                            modelling = self.best_model.model_type,
                                            full_name = (
                                                self.best_model.model_name + ' - ' +
                                                self.best_model.model_type),
                                            per_group = self.best_model.per_group,
                                            model_features_name = self.best_model.X_cols,
                                            y_col = self.best_model.y_col
                                            )

            print("TO CODE: Ajouter les performances moyennes de l'algo selected")
            print("TO CODE: Performances Monitoring for ALGO")

            print(results_summary_metrics[['Full Name','RMSPE','MAE','RMSE','MAPE']])
            results_summary_metrics.to_csv(self.path_results + '/summary_metrics.csv',index=False)

        return df_residuals_prod


    def predict(self, production_dataset: pd.DataFrame,
                model:str = 'best_model',version:str ='V0',summarize=True):
        """
        make prediction
        """

        if (model != self.best_model.model_name) or (version != self.best_model.model_version):
            # self.best_model.load_model(self.model_repository,model_name ='best_model')
            print("model versionning is not coded yet")

        if self.best_model.model_modelling_features not in production_dataset.columns.to_list():
            print("predictions features not in input dataset. Check pipeline")

        if self.best_model.X_cols is None:
            raise ValueError('X_cols are missing')

        if (self.best_model.log_model is True) and ('const' not in self.best_model.X_cols):
            raise ValueError('const should be part of model features for Product model')

        if (len(self.best_model.model) == 1) and (self.best_model.per_group is True):
            raise ValueError('per_group error, should be False ?')

        elif (len(self.best_model.model) != 1) and (self.best_model.per_group is False):
            raise ValueError('per_group error, should be True ?')

        production_dataset['y_pred'] = 0.0

        if (('const' in self.best_model.X_cols) and 
            ('const' not in production_dataset.columns.to_list())):
            production_dataset = add_constant(production_dataset)

        if self.best_model.per_group is False:
            model_all = list(self.best_model.model.values())[0]
            production_dataset['y_pred'] = (
                transform_function(model_all.predict(production_dataset[self.best_model.X_cols]),
                                   self.best_model.log_model))

        elif self.best_model.per_group is True:
            for _, group in enumerate(self.best_model.ship_dwt_categories):

                models_i = self.best_model.model[group]
                production_dataset.loc[production_dataset['Group_DWT_BV']==group,'y_pred'] = (
                    transform_function(models_i.predict(
                        production_dataset.loc[production_dataset['Group_DWT_BV']==group,
                                               self.best_model.X_cols]),
                        self.best_model.log_model))

        df_residuals = pd.DataFrame()

        if summarize is True:
            print("Warning !  This is true when we have access to target data. "
                  "For pure production dataset, implement code so that " 
                  "if self.best_model.y_col not in production_dataset on ne perform pas l'analyse")
            df_residuals = self.summarize(production_dataset)

        return production_dataset, df_residuals
