################################################################
# Automatic Parameter Tuning: Optimizer
#   1. Class "ParameterOptimizer"
#Last Update: 16-Dec-2024
#Environment: Stryker Dev3
################################################################


from helpers import process_and_export, final_report


import pandas as pd
from itertools import product
import logging
import os

       
class ParameterOptimizer:
    """
    A class to optimize forecasting parameters and causal factors to improve accuracy metrics.

    This class provides functionality to:
    - Tune forecasting parameters or causal factors based on selected modes (Cartesian or Regular).
    - Evaluate different parameter values or CF kinds for their impact on forecasting accuracy.
    - Generate and save results, including best configurations, WMAPE comparisons, and detailed reports.
    - Configure optimized kinds in ODMC.

    Attributes:
        client (object): The API client used for communication with the external system.
        profile (object): The profile containing forecasting parameters.
        plan (object): The plan.
        base_url (str): The base URL for API communication.
        accuracy_table_id (str): The ID of the table for accuracy metrics.
        top_offenders_table_id (str): The ID of the table for top offenders.
        year_over_year_table_id (str): The ID of the table for year-over-year comparisons.
        number_of_top_offenders (int): Number of top offenders to track during optimization.
        tune_by_wmape (On / Off): Whether to optimize based on WMAPE.
        base_wmape (float): The initial WMAPE value before optimization.
        base_bias (float): The initial bias value before optimization.
        parameters_tuning_mode (str): The tuning mode, "Cartesian" or "Regular."
        best_wmape (float): The best WMAPE value achieved during optimization.
        updated_bias (float): The bias value corresponding to the best WMAPE.
        permutations (list): A list of permutations generated during Cartesian tuning.
        metrics_dict (dict): A dictionary storing WMAPE and Bias for parameter combinations or CF kinds.
        all_parameters_results (.DataFrame): DataFrame summarizing results for all parameter evaluations.
        best_values_dict_params (dict): Dictionary storing the best parameter values and their metrics.
        best_values_df (DataFrame): DataFrame summarizing the best parameter values.
        best_value (object): The current best parameter value or CF kind.
        what_to_tune (str): Target for tuning, "Parameters" or "CFs."
        CF_kinds_table_id (str): The ID of the CF kinds table for tuning CFs.
        kind_values_lst (list): A list of CF kind values to evaluate.
        all_kinds_results (dict): Dictionary storing WMAPE and Bias results for all CF kinds.
        best_values_dict_CF_kinds (dict): Dictionary storing the best CF kind values and their metrics.
        merged_dfs_lst (dict): Dictionary of DataFrames containing results for each kind value.
        best_kinds (DataFrame): DataFrame summarizing the best kind for each combination.
        Configure_best_kind_IDs_on_ODMC (On / Off): Whether to configure the best kinds in ODMC.
        before_and_after_accuracy_df (DataFrame): DataFrame comparing WMAPE and Bias before and after optimization.
    """
    def __init__(self, client, profile, plan, base_url, accuracy_table_id, top_offenders_table_id, year_over_year_table_id, number_of_top_offenders, tune_by_wmape, base_wmape, base_bias,
                 parameters_tuning_mode, what_to_tune = None,CF_kinds_table_id = None, kind_values_lst = None, Configure_best_kind_IDs_on_ODMC = None):
        self.client = client
        self.profile = profile
        self.plan = plan
        self.base_url = base_url
        self.accuracy_table_id = accuracy_table_id
        self.top_offenders_table_id = top_offenders_table_id
        self.year_over_year_table_id = year_over_year_table_id
        self.number_of_top_offenders = number_of_top_offenders
        self.tune_by_wmape = tune_by_wmape
        self.base_wmape = base_wmape
        self.base_bias = base_bias
        self.parameters_tuning_mode = parameters_tuning_mode
        self.best_wmape = self.base_wmape
        self.updated_bias = self.base_bias
        self.permutations = []
        self.metrics_dict = {} 
        self.all_parameters_results = None
        self.best_values_dict_params = {}
        self.best_values_df = None
        self.best_value = None
        self.what_to_tune = what_to_tune
        self.CF_kinds_table_id = CF_kinds_table_id
        self.kind_values_lst = kind_values_lst
        self.all_kinds_results = {}
        self.best_values_dict_CF_kinds = {}
        self.merged_dfs_lst = {}
        self.best_kinds = None
        self.Configure_best_kind_IDs_on_ODMC = None
        self.before_and_after_accuracy_df = None
        self.Top_offenders_per_run_dict = {}
        self.is_dummy = False

    def run_process(self, what_to_tune, parameter_id, parameter_name, parameter_value, permutation, ind,CF_kinds_table_id, kind_value, Top_offenders_per_run_dict):
        '''
        Run the process for each parameter-value pair / Kind value and return results
        '''
        return process_and_export(
            self.profile,
            self.plan,
            self.client,
            self.base_url,
            self.accuracy_table_id,
            self.top_offenders_table_id,
            self.year_over_year_table_id,
            self.number_of_top_offenders,
            self.parameters_tuning_mode,
            what_to_tune,
            parameter_id,
            parameter_name,
            parameter_value,
            permutation, 
            ind,
            CF_kinds_table_id,
            kind_value,
            Top_offenders_per_run_dict
        )

    def find_best_value(self, parameter_id = None, parameter_name = None, parameter_values = None, permutation = None, ind = None):
        """
        Find the best parameter value or permutation based on WMAPE.

        This method evaluates parameter values or Cartesian permutations to determine the best configuration
        based on WMAPE and Bias. Updates metrics and tracks the best results.

        Args:
            parameter_id (str, optional): The ID of the parameter being tuned (for Regular tuning mode).
            parameter_name (str, optional): The name of the parameter being tuned (for Regular tuning mode).
            parameter_values (list, optional): A list of possible values for the parameter (for Regular tuning mode).
            permutation (dict, optional): A dictionary representing a permutation of parameter values (for Cartesian mode).
            ind (int, optional): Index of the current permutation (for Cartesian mode).

        Returns:
            tuple: The best WMAPE and the corresponding best value or permutation.
        """

        if self.parameters_tuning_mode == "Cartesian":
            # Tuning mode: Cartesian
            logging.info(f'Cartesian')

            # Evaluate the current permutation
            wmape, bias, self.Top_offenders_per_run_dict = self.run_process(self.what_to_tune, None, None, None, permutation = permutation, ind = ind, CF_kinds_table_id = None, kind_value = None, Top_offenders_per_run_dict = self.Top_offenders_per_run_dict)

            logging.info(f'Current WMAPE: {wmape}, current Bias: {bias} ')

            # Update metrics dictionary for each parameter in the permutation
            for ((perm_id, perm_type), value) in permutation.items():
                key_0 = ind
                key_1 = perm_type
                key_2 = value
                self.metrics_dict[(key_0, key_1, key_2)] = [wmape, bias]
                print(print(f'self.metrics_dict: {self.metrics_dict}, type: {type(self.metrics_dict)}'))

            # Update best WMAPE and corresponding values if the current WMAPE is better    
            if wmape < self.best_wmape:
                logging.info(f'Best Wmape so far')
                print("in wmape < best_wmape")
                self.best_wmape = wmape
                self.best_value = permutation
                self.metrics_dict[(key_0, key_1, key_2)].append('Best')
                self.updated_bias = bias
                print(self.best_value)
            else:
                self.metrics_dict[(key_0, key_1, key_2)].append('NOT Best')
        else:
            # Tuning mode: Regular
            for value in parameter_values:
                print(f'----------------')
                # Evaluate the current parameter value
                wmape, bias, self.Top_offenders_per_run_dict  = self.run_process(self.what_to_tune, parameter_id = parameter_id, parameter_name = parameter_name,
                                                                                 parameter_value = value, permutation = None, ind = None, CF_kinds_table_id = None,
                                                                                 kind_value = None, Top_offenders_per_run_dict = self.Top_offenders_per_run_dict)                       
                print(f'----------------')
                logging.info(f'Current WMAPE: {wmape}, current Bias: {bias} ')

                # Update metrics dictionary for the parameter and its value
                self.metrics_dict[(parameter_name, value)] = [wmape, bias]

                # Update best WMAPE and corresponding value if the current WMAPE is better
                if wmape < self.best_wmape:
                    logging.info(f'Best Wmape for {parameter_name} so far')
                    print("in wmape < best_wmape")
                    logging.info(f'self.best_wmape: {self.best_wmape}')
                    logging.info(f'current wmape: {wmape}')
                    self.best_wmape = wmape
                    self.best_value = value
                    self.metrics_dict[(parameter_name, value)].append('Best')
                    self.updated_bias = bias
                else:
                    self.metrics_dict[(parameter_name, value)].append('NOT Best')

        # Return the best WMAPE and corresponding best value or permutation            
        return self.best_wmape, self.best_value
    
    def find_cartesian_permutations(self):
        """
        Generate all possible Cartesian permutations of parameter values and store them as a list of dictionaries.

        This method uses the keys and values from the profile's parameters dictionary to compute
        all combinations of parameter values and stores them in `self.permutations`.
        """
        # Prepare keys and values for the Cartesian product
        keys = list(self.profile.parameters_dict.keys())
        values = list(self.profile.parameters_dict.values())

        # Generate all combinations of the values
        permutations = list(product(*values))

        # Store all permutations as a list of dictionaries
        self.permutations = []
        for perm in permutations:
            result_dict = {keys[i]: value for i, value in enumerate(perm)}
            self.permutations.append(result_dict)

    def create_all_parameters_results(self):
        """
        Create a DataFrame summarizing the tuning results for all parameters or permutations.

        This method processes the `self.metrics_dict` to generate a structured DataFrame with tuning results.
        The format of the DataFrame varies based on the tuning mode (Cartesian or Regular).
        """
        if self.parameters_tuning_mode == "Cartesian":
            self.all_parameters_results = pd.DataFrame(
            [
                {"Permutation Number":key[0], "parameter_name": key[1], "value": key[2], "WMAPE": val[0], "Bias": val[1]}
                for key, val in self.metrics_dict.items()
            ]
        )
        else:
            self.all_parameters_results = pd.DataFrame(
                [
                    {"parameter_name": key[0], "value": key[1], "wmape": val[0], "bias": val[1], "Best?": val[2]}
                    for key, val in self.metrics_dict.items()
                ]
            )
  


    def create_best_values_df(self):
        """
        Create a DataFrame summarizing the best parameter values and their corresponding WMAPE.

        This method processes `self.best_values_dict_params` to generate a structured DataFrame,
        providing an overview of the best value for each parameter and its associated WMAPE.
        """

        self.best_values_df = pd.DataFrame(
            [
                {"Parameter Name": key, "Value": val[0], "Wmape": val[1]}
                for key, val in self.best_values_dict_params.items()
            ]
        )



    def create_before_and_after_accuracy_df(self):
        """
        Create a DataFrame to compare accuracy metrics (WMAPE and Bias) 
        before and after the parameter optimization process.

        This method structures the initial and final WMAPE and Bias values 
        into a table format for easy comparison.
        """
        # Prepare data for the DataFrame
        data = {
                "Type": ["Initial", "Final"],
                "WMAPE": [self.base_wmape, self.best_wmape],
                "Bias": [self.base_bias, self.updated_bias]
            }

        # Create the DataFrame from the prepared data
        self.before_and_after_accuracy_df = pd.DataFrame(data)


    def create_top_offenders_comparison_dfs(self):
        # Directory to store exported DataFrames
        output_dir = "exported_dfs"
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        # Dictionary to store the resulting DataFrames
        top_offenders_comparison_dfs = {}

        # Iterate over each combination in the dictionary
        for combination, data_list in self.Top_offenders_per_run_dict.items():
            combined_df = pd.DataFrame()  # Start with an empty DataFrame
            if self.what_to_tune == "Parameters":
                if  self.parameters_tuning_mode == 'Cartesian':
                    for data in data_list:
                        for ind_key, subset_df in data.items():
                            # Add a suffix to the column names
                            suffix = f"_permutation_{ind_key}"
                            subset_df_with_suffix = subset_df.add_suffix(suffix)
                            
                            # Concatenate the DataFrame with the current combined DataFrame
                            combined_df = pd.concat([combined_df, subset_df_with_suffix], axis=1)
                else:
                    for data in data_list:
                        for (parameter_name, parameter_value), subset_df in data.items():
                            # Add a suffix to the column names
                            suffix = f"_{parameter_name}_{parameter_value}"
                            subset_df_with_suffix = subset_df.add_suffix(suffix)
                            
                            # Concatenate the DataFrame with the current combined DataFrame
                            combined_df = pd.concat([combined_df, subset_df_with_suffix], axis=1)

            elif self.what_to_tune == "CFs":
                for data in data_list:
                    for kind_id_key, subset_df in data.items():
                        # Add a suffix to the column names
                        suffix = f"_Kind_{kind_id_key}"
                        subset_df_with_suffix = subset_df.add_suffix(suffix)
                        
                        # Concatenate the DataFrame with the current combined DataFrame
                        combined_df = pd.concat([combined_df, subset_df_with_suffix], axis=1)
            
            # Store the resulting DataFrame in the dictionary
            top_offenders_comparison_dfs[combination] = combined_df
            
            # Export to Excel in the specified directory
            filename = os.path.join(output_dir, f"{combination}_combined.xlsx")
            combined_df.to_excel(filename, index=False)
            print(f"Saved: {filename}")


            # Step 1: Group columns by their prefix and exclude the output measure columns
            unique_columns = {}
            for col in combined_df.columns:
                prefix = col.split('_')[0]  # Extract prefix before first underscore
                if not col.startswith(self.plan.output_measure):  # Ignore output measure columns
                    if prefix not in unique_columns:
                        unique_columns[prefix] = col  # Keep the first occurrence
                else:
                    unique_columns[col] = col  # Always keep output measure columns

            # Step 2: Create a new DataFrame with the selected columns
            new_columns = list(unique_columns.values())
            df_cleaned = combined_df[new_columns]

            # Step 3: Rename columns for duplicates to clean prefixes
            df_cleaned.columns = [col.split('_')[0] if not col.startswith(self.plan.output_measure) else col for col in df_cleaned.columns]


            # Move "OOS" and "End of History" to the last two positions
            columns = [col for col in df_cleaned.columns if col not in ["End of History", "OOS"]]
            columns += ["End of History", "OOS"]  # Append the desired columns at the end

            # Reorder DataFrame
            df_cleaned = df_cleaned[columns]
            df_cleaned.to_excel(filename, index=False)
        

    def optimize_parameters(self):
        """
        Optimize parameters based on the selected tuning mode (Cartesian or Regular).

        This method evaluates all parameter combinations or individual parameter values to find the best configuration
        based on WMAPE and Bias. Results are recorded, and a final report is generated.
        """

        # Check if the tuning mode is Cartesian
        if self.parameters_tuning_mode == "Cartesian":
            logging.info(f"Cartesian Mode:")

            # Generate all Cartesian permutations of parameter values
            self.find_cartesian_permutations()

            logging.info("All permutations to tune: ")
            logging.info(self.permutations)

            # Iterate through each permutation dictionary
            for i, permutation in enumerate(self.permutations):

                logging.info("__________________________________________________________________________________________________________________________________________________")
                logging.info(f'Premutation number {i+1}: {permutation}')

                # Evaluate the current permutation
                best_wmape, best_value = self.find_best_value(None, None, None, permutation=permutation, ind = i+1)
                
            print(print(f'best_value: {best_value}, type: {type(best_value)}'))

            # Update the best values and configure the profile based on the results
            for ((parameter_id, parameter_name), parameter_value) in best_value.items():
                self.best_values_dict_params[parameter_name] = [parameter_value, best_wmape]
                
                 # If no permutation improves WMAPE, reset to initial parameter values
                if  best_value is None:
                    logging.info(f'No permutation improved the base WMAPE. initializing the profile to have initial parameters values: {parameter_name} ')
                    original_parameter_value = self.profile.initial_parameters.loc[self.profile.initial_parameters['ForecastingParameterName'] == parameter_name, 'ForecastingParameterValue'].values
                    original_parameter_value_int = float(original_parameter_value[0])
                    #self.profile.change_parameter(parameter_id, original_parameter_value_int)  # Update the profile

                else:
                    # Configure the profile with the best permutation values
                    logging.info(f'Configuring best permutation values in the profile: {parameter_name}: value: {parameter_value} ')
                    #self.profile.change_parameter(parameter_id, parameter_value)

        else:
            logging.info(f'profile.parameters_dict: {self.profile.parameters_dict} ')
             # Regular tuning mode: iterate through each parameter and its possible values
            for (parameter_id, parameter_name), parameter_values in self.profile.parameters_dict.items():
                 # Evaluate each parameter value
                self.best_value = None
                best_wmape, best_value = self.find_best_value(parameter_id, parameter_name, parameter_values)

                # Store the best value and WMAPE for the parameter
                self.best_values_dict_params[parameter_name] = [best_value, best_wmape]
                if not self.is_dummy:
                    logging.info(f'best_value is: {best_value}, self.best_value is: {self.best_value} ')

                if self.tune_by_wmape != "On" or best_value is None:
                    print("Best Value is None")
                    # If no improvement, reset to the initial parameter value
                    if not self.is_dummy:
                        logging.info(f'Initializing the profile to have initial parameters values: {parameter_name} ')
                    original_parameter_value = self.profile.initial_parameters.loc[self.profile.initial_parameters['ForecastingParameterName'] == parameter_name, 'ForecastingParameterValue'].values
                    original_parameter_value_int = float(original_parameter_value[0])
                    self.best_values_dict_params[parameter_name] = [original_parameter_value, best_wmape]
                    if not self.is_dummy:
                        self.profile.change_parameter(parameter_id, original_parameter_value_int)  # Update the profile
                        logging.info(f'best_value is: {original_parameter_value}')
                        print(f"Updated {parameter_name} to the initial value:{original_parameter_value_int} as no other tested value had better WMAPE")
                else:
                    # Store the best value and WMAPE for the parameter
                    self.best_values_dict_params[parameter_name] = [best_value, best_wmape]
                    if not self.is_dummy:
                        logging.info(f'best_value is: {best_value}')
                        # Update the profile with the best value found
                        logging.info(f'Configuring best value in the profile: {parameter_name}: value: {best_value} ')
                        self.profile.change_parameter(parameter_id, best_value)  # Update the profile
                        print(f"Updated {parameter_name} to {best_value} with WMAPE: {best_wmape}")


        # Create summary results and reports
        self.create_all_parameters_results()
        self.create_best_values_df()
        self.create_before_and_after_accuracy_df()
        if not self.is_dummy:
            # Generate the final report
            final_report(self.profile, self.all_parameters_results, self.before_and_after_accuracy_df,
                        self.best_values_df,best_CF_Kind_value_df = None, all_kinds_results_df = None, 
                        what_to_tune = "Parameters", best_kinds_df = None, file_path = "Final Report Parameters.xlsx")
            if self.Top_offenders_per_run_dict:
                self.create_top_offenders_comparison_dfs()



    def find_best_kind_per_comb(self):
        """
        Identify the best 'kind ID' for each combination based on the minimum ABS Error'
        and save the results to a CSV file.

        This method processes a dictionary of DataFrames (`self.merged_dfs_lst`), flattens it 
        into a single DataFrame, and identifies the optimal kind ID for each combination.
        """
        # Step 1: Flatten the dictionary into a single DataFrame
        df_list = []
        for kind_id, df in self.merged_dfs_lst.items():
            df = df.copy()
            df['kind ID'] = kind_id  # Add the kind id as a column
            df_list.append(df)

        combined_df = pd.concat(df_list, ignore_index=True)

        # Step 2: Group by 'Combination' and find the row with the minimum 'ABS Error'
        self.best_kinds = combined_df.loc[combined_df.groupby('Combination')['ABS Error'].idxmin()]

        # Step 3: Resulting DataFrame with the best 'kind id' for each 'Combination'
        self.best_kinds = self.best_kinds.reset_index(drop=True)
        # Step 4: Move 'Combination' column to the end
        combination_col = self.best_kinds.pop('Combination')  # Remove 'Combination' column
        self.best_kinds['Combination'] = combination_col  # Add it back as the last column

        self.best_kinds.to_csv("best_kinds.csv", index=False)



    def configure_best_kinds_per_comb_ODMC(self, kinds_table):
        """
        Configure the best 'kind ID' for each combination in ODMC.

        This method updates the `kinds_table` with the best kinds for each combination,
        excluding unnecessary columns like 'ABS Error', and triggers the update process in ODMC.
        
        Args:
            kinds_table: The table object representing kinds data that needs to be modified.
        """
        # Exclude the "ABS Error" column from `self.best_kinds`
        filtered_best_kinds = self.best_kinds.drop(columns=['ABS Error'])

        # Update the `modified_kinds_df` attribute of the `kinds_table` with the filtered DataFrame
        kinds_table.modified_kinds_df = filtered_best_kinds


        #Trigger the update of kinds in ODMC with the modified DataFrame
        kinds_table.change_kinds_on_ODMC(None, "Local")



    def optimize_CFs_kinds(self):
        """
        Optimize CFs kinds by evaluating WMAPE and Bias for each kind value.

        This method iterates over a list of kind values, evaluates their performance using WMAPE and Bias,
        identifies the best-performing kind, and prepares summary results and a final report.
        """
        # Iterate over each kind value in the provided list
        for kind_value in self.kind_values_lst:
            # Run the tuning process for the current kind value
            print(f'in optimize CFs: kind value = {kind_value}')
            wmape, bias, merged_df, kinds_table, self.Top_offenders_per_run_dict  = self.run_process(self.what_to_tune, None, None, None, permutation = None, ind = None, 
                                                                   CF_kinds_table_id = self.CF_kinds_table_id, kind_value = kind_value,
                                                                   Top_offenders_per_run_dict = self.Top_offenders_per_run_dict)
            # Store WMAPE and Bias results for the current kind value
            self.all_kinds_results[kind_value] = [wmape, bias]

            # Dynamically select columns based on their position
            self.merged_dfs_lst[kind_value] = merged_df.iloc[:, [
                0,                      # First column (product level)
                1,                      # Second column (location level)
                2,                      # Third column (combination)
                -2                      # Column one before the last  (abs error)
            ]]

             # Check if the current WMAPE is better than the best WMAPE so far
            if wmape < self.best_wmape:
                logging.info(f'Best Wmape for CF kinds tuning so far (kind ID = {kind_value})')
                print("in wmape < best_wmape")
                logging.info(f'old self.best_wmape: {self.best_wmape}')
                logging.info(f'current wmape: {wmape}')
                self.best_wmape = wmape
                self.best_value = kind_value
                self.updated_bias = bias

        # Store the best kind value and its WMAPE in a dictionary        
        self.best_values_dict_CF_kinds['Kind'] = [self.best_value, self.best_wmape]

        # Create a DataFrame summarizing the best CF kind values
        self.best_CF_Kind_value_df = pd.DataFrame(
            [
                {"Name": key, "Value": val[0], "Wmape": val[1]}
                for key, val in self.best_values_dict_CF_kinds.items()
            ]
        )

        # Create a DataFrame summarizing all kinds results
        self.all_kinds_results_df = pd.DataFrame(
            [
                {"Kind ID Value": key, "Wmape": val[0], "Bias": val[1]}
                for key, val in self.all_kinds_results.items()
            ]
            
        )
        
               
        logging.info(f'self.best_value: {self.best_value}')

        # Find the best kind for each combination based on the minimum ABS Error
        self.find_best_kind_per_comb()

        # Create a comparison DataFrame for before-and-after accuracy metrics
        self.create_before_and_after_accuracy_df()

        # Generate the final report for CF tuning
        final_report(self.profile, all_parameters_results = None, before_and_after_accuracy_df = self.before_and_after_accuracy_df,
                     best_values_dict_params = None,best_CF_Kind_value_df = self.best_CF_Kind_value_df,
                     all_kinds_results_df = self.all_kinds_results_df, what_to_tune = "CFs",
                     best_kinds_df =self.best_kinds, file_path = "Final Report CFs.xlsx")

        # If configuration on ODMC is enabled, apply the best kind IDs for each combination
        if self.Configure_best_kind_IDs_on_ODMC == "On":
            logging.info(f'Configuring best kind ID per combination on ODMC')
            self.configure_best_kinds_per_comb_ODMC(kinds_table)

        if self.Top_offenders_per_run_dict:
            self.create_top_offenders_comparison_dfs()


    def functions_navigator(self):
        """
        Navigate to the appropriate optimization function based on the target to tune.

        This method decides whether to tune Causal Factors (CFs) or Parameters
        based on the value of `self.what_to_tune`. Logs and prints the selected
        tuning mode before proceeding with the respective optimization process.
        """
        if self.what_to_tune == "CFs":
            # If the target to tune is Causal Factors, log and proceed with CF tuning
            logging.info('Going CFs Tuning')
            print("Going CFs Tuning")
            self.optimize_CFs_kinds()
        elif self.what_to_tune == "Parameters":
            logging.info('Going Parameters Tuning')
            print("Going Parameters tuning")
            self.optimize_parameters()
        elif self.what_to_tune == "Dummy":
            logging.info('Dummy mode. Not launching plan runs, only generating reports.')
            self.parameters_tuning_mode = "Regular"
            self.is_dummy = True
            self.profile.parameters_dict = {(0,'FitValidationSensitivity'): [1]}
            self.optimize_parameters()
            
        

     