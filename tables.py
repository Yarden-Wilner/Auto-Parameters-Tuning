################################################################
# Automatic Parameter Tuning: Tables
#   1. Class "Table"
#   2. Class "Accuracy_Table"
#   3. Class "Top_Offenders_Table"
#   4. Class "Yoy_Table"
#   5. Class "Kinds_Table"
#Last Update: 16-Dec-2024
#Environment: Stryker Dev3
################################################################

import pandas as pd
import json
import base64
from io import StringIO
import logging
import sys


class Table:
    '''
    Initialize the Table class.

    :param client (APIClient): The API client used for sending requests to the server
    :param table_url (str): The base URL for plan-related API operations
    '''
    def __init__(self, client, plan_id, input_measure, output_measure):
        self.client = client
        self.plan_id = plan_id
        self.input_measure = input_measure
        self.output_measure = output_measure


    def fetch_data(self):
        '''
        Get the json response that holds plans attributes (offset, end of history date, etc.)
        '''
        return self.client.send_get_request(self.table_url)

class Accuracy_Table(Table):
    def __init__(self,client, plan_id, input_measure, output_measure, base_url, accuracy_table_id, number_of_top_offender):
        '''
        Initialize the Table class.

        :param client (APIClient): The API client used for sending requests to the server
        :param table_url (str): The base URL for plan-related API operations
        :param number_of_top_offender (int): Number of top offenders to find (user input)
        :param item_level (str): The item level name of the table (e.g., item, HW Option)
        :param location_level (str): The location level name of the table (e.g., organization, Region)
        :param time_level (str): The time level name of the table (e.g., week, month)
        :param accuracy_df (DataFrame): DataFrame version of the publishe ODMC accuracy table (for the current run)
        :param total_history_sum (int): sum of historical demand across all the table (for the current run)
        :param total_forecast_sum (int): sum of forecast across all the table (for the current run)
        :param total_abs_error_sum (int): sum of abs error across all the table (for the current run)
        :param wmape (int): wmape of the current run
        :param bias (int): wmape of the current run
        :param merged_df (DataFrame): columns: combination, exported history (few columns), exported forecast (few columns), total history, total forecast, abs error, mape (per combination)
        :param accuracy_df_for_export (DataFrame): holds metrics- overall total history, overall total forecast, overall total abs error, overall mape, overall bias
        :param grouped_accuracy (DataFrame): holds wmape and bias per group (group=locations)
        :param top_offenders (DataFrame): subset of merged_df, with only top offenders combinations
        '''
        super().__init__(client, plan_id, input_measure, output_measure)
        self.table_url  = f"{client.base_url}/supplyChainPlans/{plan_id}/child/PlanningTables/{accuracy_table_id}/child/Data"
        self.number_of_top_offender = number_of_top_offender
        self.item_level= None
        self.location_level = None
        self.time_level = None
        self.accuracy_df = None
        self.total_history_sum = None
        self.total_forecast_sum = None
        self.total_abs_error_sum = None
        self.wmape = None
        self.bias = None
        self.merged_df = None
        self.accuracy_df_for_export = None
        self.grouped_accuracy = None
        self.top_offenders = None
        # Initialize placeholders for the optional columns
        self.forecast_methods_col_indx = None
        self.forecast_levels_col_indx = None
        self.item_level_indx = None         # Index of item level column
        self.location_level_indx = None,     # Index of location level column
        self.time_level_indx = None,         # Index of time level column
        self.input_measure_indx = None,      # Index of input measure column
        self.output_measure_indx = None      # Index of output measure column
        self.methods_and_levels_dfs = {"Methods":None, "Levels":None}

        self.initialize_funcs()

    def fetch_data(self):
        '''
        Get the json response that holds plans attributes (offset, end of history date, etc.)
        '''
        return self.client.send_get_request(self.table_url)

    def get_table_levels(self,dict):
        '''
        Get the levels names of the accuracy table (item level, location level, time level)
        '''
        table_levels_str = dict[0]['TableDataHeader']
        table_levels_lst = table_levels_str.split(',')
        self.item_level= table_levels_lst[0]
        self.location_level = table_levels_lst[1]
        self.time_level = table_levels_lst[2]

        # Normalize self.output_measure to lowercase for consistent comparisons
        normalized_output_measure = self.output_measure.lower()

        for idx, header in enumerate(table_levels_lst):
            normalized_header = header.lower()  # Normalize the header to lowercase
            if f"{normalized_output_measure} forecast method" in normalized_header:
                self.forecast_methods_col_indx = idx  # Store the index of the column
            elif f"{normalized_output_measure} forecast level" in normalized_header:
                self.forecast_levels_col_indx = idx  # Store the index of the column

    
    def create_accuracy_table_data_df(self, json_table):
        '''
        convert json table data into a DataFrame (self.accuracy_df)
        steps:
        1.convert str to list, each row is a combination
        2. create a dictionary: 
            keys: headers of the table (e.g., date, input/output measures, item level, location level)
            values: exported table values
        3. convert dictionary to DataFrame
            change data types
        '''
        table_data_str = json_table['items'][0]['TableData']
        table_data_list = table_data_str.split('\r\n')  #each row is a combination
        table_data_dict = {self.item_level: [], self.location_level: [], self.time_level: [], self.input_measure: [], self.output_measure: []}

        if self.forecast_methods_col_indx is not None and self.forecast_levels_col_indx is not None:
            if self.forecast_methods_col_indx < self.forecast_levels_col_indx:
                table_data_dict["Methods"] = []
                table_data_dict["Levels"] = []
            else:
                table_data_dict["Levels"] = []
                table_data_dict["Methods"] = []
        elif self.forecast_methods_col_indx is not None:
                table_data_dict["Methods"] = []
        elif self.forecast_levels_col_indx is not None:
                table_data_dict["Levels"] = []

        self.create_accuracy_table_data_df_helper(table_data_dict, table_data_list)
        

    def create_accuracy_table_data_df_helper(self, table_data_dict, table_data_list):
        '''
        execute steps 2 and 3 of the function create_accuracy_table_data_df
        '''
        # Define column indices (correctly matching column names to indices)
        column_mapping = {
            self.item_level: 0,          # Index of item level column
            self.location_level: 1,     # Index of location level column
            self.time_level: 2,         # Index of time level column
            self.input_measure: 3,      # Index of input measure column
            self.output_measure: 4      # Index of output measure column
        }

        # Optionally add "Methods" and "Levels" if their indices are set
        if self.forecast_methods_col_indx is not None:
            column_mapping["Methods"] = self.forecast_methods_col_indx
        if self.forecast_levels_col_indx is not None:
            column_mapping["Levels"] = self.forecast_levels_col_indx

    # Populate the dictionary with values
        for i in range(len(table_data_list)):
            # Split each element by the ',' delimiter
            values = table_data_list[i].split(',')  #each element is a different column on the table
            if values == [""]: 
                continue
            else:  
                #populate dictionary with values from the 
                table_data_dict[self.item_level].append(values[0])  #item names
                table_data_dict[self.location_level].append(values[1])  #location names
                table_data_dict[self.time_level].append(values[2])  #date
                table_data_dict[self.input_measure].append(values[3])   #historical demand
                table_data_dict[self.output_measure].append(values[4])  #forecast
                if "Methods" in table_data_dict:
                    table_data_dict["Methods"].append(values[self.forecast_methods_col_indx])
                if "Levels" in table_data_dict:
                    table_data_dict["Levels"].append(values[self.forecast_levels_col_indx])

        self.accuracy_df = pd.DataFrame(table_data_dict)
        self.accuracy_df['Combination'] = self.accuracy_df[self.item_level] + " " + self.accuracy_df[self.location_level]
        self.accuracy_df[self.time_level] = pd.to_datetime(self.accuracy_df[self.time_level]).dt.date 
        # Converting input (history) and output (forecast) columns to numeric
        self.accuracy_df[self.input_measure] = pd.to_numeric(self.accuracy_df[self.input_measure])
        self.accuracy_df[self.output_measure] = pd.to_numeric(self.accuracy_df[self.output_measure])
        # Add "Methods" column if it exists in table_data_dict
        if "Methods" in table_data_dict:
            self.accuracy_df["Methods"] = table_data_dict["Methods"]

        # Add "Levels" column if it exists in table_data_dict
        if "Levels" in table_data_dict:
            self.accuracy_df["Levels"] = table_data_dict["Levels"]

    def create_Pivots(self, df):
        '''
        Input: self.accuracy_df
        Output: 3 Dataframes: 
                pivot_forecast - sums forecast for all time periods per combination
                pivot_history - sums historical for all time periods per combination
                pivot_diff - abs(history-forecast)
        '''

        # Creating the pivot table for self.output_measure
        pivot_forecast = df.pivot_table(index='Combination', columns=self.time_level, values=self.output_measure, aggfunc='sum').fillna(0)
        pivot_forecast["Total Forecast"]=pivot_forecast.sum(axis=1)

        # Creating the pivot table for self.input_measure
        pivot_history = df.pivot_table(index='Combination', columns=self.time_level, values=self.input_measure, aggfunc='sum').fillna(0)
        pivot_history["Total History"]=pivot_history.sum(axis=1)

        # Calculating the absolute difference for each date column
        pivot_diff = abs(pivot_forecast - pivot_history)
        #pivot_diff.columns = [str(col) + '_abs_difference' for col in pivot_diff.columns]
        pivot_diff['ABS Error'] = pivot_diff.sum(axis=1)
        pivot_diff = pivot_diff.rename(
            columns={
                self.input_measure: f"{self.input_measure}_diff",
                self.output_measure: f"{self.output_measure}_diff"
            }
        )

        # Drop unnecessary columns
        if 'Total Forecast' in pivot_diff.columns:
            pivot_diff = pivot_diff.drop(columns=['Total Forecast'])
        if 'Total History' in pivot_diff.columns:
            pivot_diff = pivot_diff.drop(columns=['Total History'])

        # Remove duplicates based on the 'Combination' column or indices
        pivot_diff = pivot_diff.reset_index().drop_duplicates(subset=['Combination']).set_index('Combination')


        # Resetting the index for merging
        pivot_forecast = pivot_forecast.reset_index()
        pivot_history = pivot_history.reset_index()
        pivot_diff = pivot_diff.reset_index()

        return pivot_forecast, pivot_history, pivot_diff
    
    def Create_Merge_df(self,pivot_forecast, pivot_history, pivot_diff):
        '''
        merge pivot tables into one table.
        Output: df with all original demand and forecast, woth total forcast and total history columns, along with abs error and mape per combiantion
        (each row is a combination)
        '''

        # Drop duplicates to avoid unexpected Cartesian products
        pivot_forecast = pivot_forecast.drop_duplicates(subset=['Combination'])
        pivot_history = pivot_history.drop_duplicates(subset=['Combination'])
        pivot_diff = pivot_diff.drop_duplicates(subset=['Combination'])
        self.accuracy_df = self.accuracy_df.drop_duplicates(subset=['Combination'])

        # Merging the pivot tables
        self.merged_df = pd.merge(pivot_history, pivot_forecast, on='Combination', suffixes=( '_History', '_Forecast'))
        self.merged_df = pd.merge(self.merged_df, pivot_diff[['ABS Error', 'Combination']], on='Combination')
        #merge with self.accuracy_df to get the Methods and Levels (if exist in the table)
        # Identify columns to include in the merge
        merge_columns = ['Combination']  # Always include 'Combination'
        optional_columns = ['Methods', 'Levels'] #sometimes only one exists in the accuracy table, sometimes both, sometimes none

        # Add optional columns only if they exist in self.accuracy_df
        merge_columns += [col for col in optional_columns if col in self.accuracy_df.columns]

        # Perform the merge with the dynamically determined columns
        self.merged_df = pd.merge(self.merged_df, self.accuracy_df[merge_columns], on='Combination')


            # Calculating MAPE
        self.merged_df['MAPE'] = self.merged_df['ABS Error'] / self.merged_df['Total History']
        self.merged_df['MAPE'] = self.merged_df['MAPE'].replace([float('inf'), float('-inf')], 1)  # Handle division by zero by replacing inf with 1
        self.merged_df['MAPE'] = self.merged_df['MAPE'].fillna(1)  # Handle NaN resulting from division by zero

        #self.merged_df = self.merged_df.drop(columns=['Total Forecast_abs_difference', 'Total History_abs_difference'])
        # Split 'comb' into two columns: 'item' and 'Location'
        self.merged_df[[self.item_level, self.location_level]] = self.merged_df['Combination'].str.split(' ', n=1, expand=True).fillna("Missing")
        # Insert 'item' and 'Location' as the first two columns
        self.merged_df.insert(0, self.item_level, self.merged_df.pop(self.item_level))
        self.merged_df.insert(1, self.location_level, self.merged_df.pop(self.location_level))

        required_columns = ["Total History", "Total Forecast", "ABS Error"]
        for col in required_columns:
            if col not in self.merged_df.columns:
                raise ValueError(f"Required column '{col}' is missing from the DataFrame.")

        # Separate the final column (e.g., "MAPE")
        final_col = self.merged_df.columns[-1]

        # Reorder columns
        reordered_columns = (
            [col for col in self.merged_df.columns if col not in required_columns + [final_col]]  # Existing columns except specified
            + ["Total History", "Total Forecast", "ABS Error"]  # Place these columns in order
            + [final_col]  # Place the final column last
        )
        
        self.merged_df = self.merged_df[reordered_columns]


    
    def accuracy_calc(self, merged_df,df):
        '''
        calculate:
        overall total history, overall total forecast, overall total abs error, overall wmape, overall bias, wmape and bias splitted by location

        (df = self.accuracy_df)
        '''
        # Calculate the sums of specified columns
        self.total_history_sum = self.merged_df['Total History'].sum()
        self.total_forecast_sum = self.merged_df['Total Forecast'].sum()
        self.total_abs_error_sum = self.merged_df['ABS Error'].sum()



        # Calculate WMAPE
        if self.total_history_sum == 0:
            self.wmape = 100
        else:
            self.wmape = self.total_abs_error_sum / self.total_history_sum

        # Calculate Bias
        if self.total_history_sum == 0:
            self.bias = 100
        else:
            self.bias = (self.total_history_sum - self.total_forecast_sum) / self.total_history_sum


        # Prepare the accuracy metrics DataFrame
        accuracy_data_for_export = {
            "Metric": ["Total History Sum", "Total Forecast Sum", "ABS Error Sum", "WMAPE", "Bias"],
            "Value": [self.total_history_sum, self.total_forecast_sum, self.total_abs_error_sum, self.wmape, self.bias]
        }
        self.accuracy_df_for_export = pd.DataFrame(accuracy_data_for_export)


        # Exclude self.time_level column from the groupby operation
        grouped = self.merged_df[[self.item_level, self.location_level, 'Total History', 'Total Forecast', 'ABS Error']].groupby(self.location_level).sum().reset_index()
        grouped['WMAPE'] = grouped['ABS Error'] / grouped['Total History'].replace(0, pd.NA)
        grouped['Bias'] = (grouped['Total History'] - grouped['Total Forecast']) / grouped['Total History'].replace(0, pd.NA)

        # Handle division by zero cases for WMAPE and Bias
        grouped['WMAPE'] = grouped['WMAPE'].replace([pd.NA, float('inf'), float('-inf')], 100).fillna(100)
        grouped['Bias'] = grouped['Bias'].replace([pd.NA, float('inf'), float('-inf')], 100).fillna(100)

        # Select only the relevant columns
        self.grouped_accuracy = grouped[[self.location_level, 'WMAPE', 'Bias']]

    def create_methods_pivot(self, column):

        
        pivot = (
            self.merged_df[column]
            .value_counts()
            .reset_index()
            .rename(columns={"index": column, column: column})
        )
        pivot["%"] = (pivot["count"] / pivot["count"].sum()) * 100
        self.methods_and_levels_dfs[column] = pivot



    def find_top_offenders(self):
        if self.number_of_top_offender > 0:
            self.top_offenders = self.merged_df.nlargest(self.number_of_top_offender, "ABS Error")


    def initialize_funcs(self):
        '''
        call class functions
        '''
        table_data = self.fetch_data()
        if table_data is None:
            logging.error("Get request for the accuracy table URL failed. Please ensure the accuracy table URL is properly configured (Table ID needs to be vaild)")
            sys.exit(1)
        self.get_table_levels(table_data['items']) 
        self.create_accuracy_table_data_df(table_data)
        pivot_forecast, pivot_history, pivot_diff = self.create_Pivots(self.accuracy_df)
        self.Create_Merge_df(pivot_forecast, pivot_history, pivot_diff)
        self.accuracy_calc(self.merged_df, self.accuracy_df)
        if "Methods" in self.merged_df.columns:
            self.create_methods_pivot("Methods")
        if "Levels" in self.merged_df.columns:
            self.create_methods_pivot("Levels")
        self.find_top_offenders()



class Top_Offenders_Table(Table):
    def __init__(self,client, plan_id, input_measure, output_measure, base_url, top_offenders_table_id , top_offenders, item_level, location_level, time_level, is_valid):
        '''
        Initialize the Table class.

        :param client (APIClient): The API client used for sending requests to the server
        :param top_offenders_table_id(str): id of the Top Offenders table (user input)
        ::param item_level (str): The item level name of the table (e.g., item, HW Option)
        :param location_level (str): The location level name of the table (e.g., organization, Region)
        :param time_level (str): The time level name of the table (e.g., week, month)
        :param table_url(str): url for the Get request (filtering the ODMC top offenders table)
        :param encoded_filter(str): encoded version of the top offenders items list and localtions lists
        :param
        :param
        :param
        :param
        '''
        super().__init__(client, plan_id, input_measure, output_measure)
        self.base_url = base_url
        self.top_offenders_table_id = top_offenders_table_id
        self.top_offenders = top_offenders
        self.item_level= item_level
        self.location_level = location_level
        self.time_level = time_level
        self.is_valid = is_valid
        self.table_url  = None
        self.encoded_filter = None
        self.unique_items = None
        self.unique_locations = None
        self.encoded_filter = None
        self.top_offenders_for_export = None
        self.initialize_funcs()
    

    def fetch_data(self):
        '''
        Get the json response that holds plans attributes (offset, end of history date, etc.)
        '''
        return self.client.send_get_request(self.table_url)
    
    def json_to_df(self, json_data, Data_Key):
        if isinstance(json_data, dict) and "items" in json_data:
            if isinstance(json_data["items"], list):
                return pd.DataFrame(json_data["items"])  # Convert items list to DataFrame
            else:
                print("Error: 'items' is not a list in the JSON data")
                return None
        print("Error: Unexpected JSON structure or missing 'items' key")
    
    def find_unique_items_locations(self):
        '''
        Get lists of unique items and unique locations of the Top Offenders
        (needed for filtering table)
        '''
        self.unique_items = self.top_offenders[self.item_level].unique().tolist()
        self.unique_locations = self.top_offenders[self.location_level].unique().tolist()

    def create_filtered_url(self):
        '''
        Create Table URL.
        Filter only the top offenders records (like filtering on ODMC: members --> product filter = self.unique_items, location filter =  self.unique_locations)
        '''
        # Construct the decoded filter
        decoded_filter = [
            {"Level": self.item_level, "Value": self.unique_items},
            {"Level": self.location_level, "Value": self.unique_locations}
        ]
        
        # Convert the filter to a JSON string
        decoded_filter_json = json.dumps(decoded_filter)
        
        # Encode the JSON string in Base64
        self.encoded_filter = base64.b64encode(decoded_filter_json.encode('utf-8')).decode('utf-8')
        self.table_url = f"{self.base_url}/supplyChainPlans/{self.plan_id}/child/PlanningTables/{self.top_offenders_table_id}/child/Data?finder=FindByFilter;Filter={self.encoded_filter}"

    def create_top_offenders_full_data_df(self):
        top_offenders_json = self.fetch_data()
        headers_str = top_offenders_json['items'][0]['TableDataHeader']
        headers_lst = headers_str.split(',')


        table_data_str = top_offenders_json['items'][0]['TableData']

        # Convert string to DataFrame
        self.top_offenders_for_export = pd.read_csv(StringIO(table_data_str), header=None, names=headers_lst, parse_dates=[self.time_level] )   ##########

        #add a "combiantion" column
        self.top_offenders_for_export['Combination'] = self.top_offenders_for_export[self.item_level].astype(str) + " " + self.top_offenders_for_export[self.location_level].astype(str)
        #reorder columns
        self.top_offenders_for_export.insert(2, 'Combination', self.top_offenders_for_export.pop('Combination'))

    
    def initialize_funcs(self):
        '''
        call class functions
        '''
        if self.is_valid == "Valid":
            self.find_unique_items_locations()
            self.create_filtered_url()
            print(self.table_url)
            Top_Offenders_Table_Data = self.client.send_get_request(self.table_url)
            top_offenders_json = self.fetch_data()
            if top_offenders_json is None:
                logging.warning("Get request failed for top offenders table URL. Please check the URL.")
                self.is_valid = "Not Valid"
                return  # Exit the function without continuing
            self.create_top_offenders_full_data_df()


class Yoy_Table(Table):
    def __init__(self,client, plan_id, input_measure, output_measure, base_url, year_over_year_table_id , end_of_history_date, is_valid):
        '''
        Initialize the Table class.

        :param client (APIClient): The API client used for sending requests to the server
        :param top_offenders_table_id(str): id of the Top Offenders table (user input)
        ::param item_level (str): The item level name of the table (e.g., item, HW Option)
        :param location_level (str): The location level name of the table (e.g., organization, Region)
        :param time_level (str): The time level name of the table (e.g., week, month)
        :param table_url(str): url for the Get request (filtering the ODMC top offenders table)
        :param encoded_filter(str): encoded version of the top offenders items list and localtions lists
        :param
        :param
        :param
        :param
        '''
        super().__init__(client, plan_id, input_measure, output_measure)
        self.year_over_year_table_id = year_over_year_table_id
        self.end_of_history_date = end_of_history_date
        self.table_url = f"{client.base_url}/supplyChainPlans/{plan_id}/child/PlanningTables/{year_over_year_table_id}/child/Data"
        self.is_valid = is_valid
        self.yoy_df = None

    def fetch_data(self):
        '''
        Get the json response that holds plans attributes (offset, end of history date, etc.)
        '''
        return self.client.send_get_request(self.table_url)
    
    def json_to_df(self):
        yoy_json = self.fetch_data()
        if yoy_json is None:
                logging.warning("Get request failed for Year Over Year table URL. Please check the URL (table ID needs to be vaild).")
                self.is_valid = "Not Valid"
                return  # Exit the function without continuing
        headers_str = yoy_json['items'][0]['TableDataHeader']
        headers_lst = headers_str.split(',')


        table_data_str = yoy_json['items'][0]['TableData']
    

        # Convert string to DataFrame
        self.yoy_df = pd.read_csv(StringIO(table_data_str), header=None, names=headers_lst)


class Kinds_Table(Table):
    def __init__(self,client, plan_id, input_measure, output_measure, CF_kinds_table_id):
        '''
        Initialize the Table class.

        :param client (APIClient): The API client used for sending requests to the server
        :param top_offenders_table_id(str): id of the Top Offenders table (user input)
        ::param item_level (str): The item level name of the table (e.g., item, HW Option)
        :param location_level (str): The location level name of the table (e.g., organization, Region)
        :param time_level (str): The time level name of the table (e.g., week, month)
        :param table_url(str): url for the Get request (filtering the ODMC top offenders table)
        :param encoded_filter(str): encoded version of the top offenders items list and localtions lists
        :param
        :param
        :param
        :param
        '''
        super().__init__(client, plan_id, input_measure, output_measure)
        self.table_url  = f"{client.base_url}/supplyChainPlans/{plan_id}/child/PlanningTables/{CF_kinds_table_id}/child/Data"
        print(self.table_url)
        self.item_level= None
        self.location_level = None
        self.kind_measure = None
        self.initial_kinds_df = None
        self.table_hierarchies_str = None
        self.table_levels_str = None
        self.modified_kinds_df = None
        self.initialize_funcs() 

    def fetch_data(self):
        '''
        Get the json response that holds plans attributes (offset, end of history date, etc.)
        '''
        return self.client.send_get_request(self.table_url)

    def get_table_levels(self,dict):
        '''
        Get the levels names of the accuracy table (item level, location level, time level)
        '''
        self.table_hierarchies_str = dict[0]['TableHierarchies']
        self.table_levels_str = dict[0]['TableDataHeader']
        table_levels_lst = self.table_levels_str.split(',')
        self.item_level= table_levels_lst[0]
        self.location_level = table_levels_lst[1]
        self.time_level = table_levels_lst[2]

    def create_kinds_table_data_df_helper(self, table_data_dict, table_data_list):
        '''
        execute steps 2 and 3 of the function create_accuracy_table_data_df
        '''
        # Define column indices (correctly matching column names to indices)
        column_mapping = {
            self.item_level: 0,          # Index of item level column
            self.location_level: 1,     # Index of location level column
            self.kind_measure: 3,         # Index of kind level column
        }


        # Populate the dictionary with values
        for i in range(len(table_data_list)):
            # Split each element by the ',' delimiter
            values = table_data_list[i].split(',')  #each element is a different column on the table
            if values == [""]: 
                continue
            else:  
                #populate dictionary with values from the 
                table_data_dict[self.item_level].append(values[0])  #item names
                table_data_dict[self.location_level].append(values[1])  #location names
                table_data_dict[self.kind_measure].append(values[2])  #date

                self.initial_kinds_df = pd.DataFrame(table_data_dict, columns=[self.item_level, self.location_level, 'Kinds'])
                self.initial_kinds_df['Combination'] = self.initial_kinds_df[self.item_level] + " " + self.initial_kinds_df[self.location_level]
                # Converting input (history) and output (forecast) columns to numeric


    
    def create_kinds_table_data_df(self, json_table):
        '''
        convert json table data into a DataFrame
        steps:
        1.convert str to list, each row is a combination
        2. create a dictionary: 
            keys: headers of the table (e.g., date, input/output measures, item level, location level)
            values: exported table values
        3. convert dictionary to DataFrame
            change data types
        '''
        table_data_str = json_table['items'][0]['TableData']
        table_data_list = table_data_str.split('\r\n')  #each row is a combination
        table_data_dict = {self.item_level: [], self.location_level: [], self.kind_measure: []}

        self.create_kinds_table_data_df_helper(table_data_dict, table_data_list)

    def change_kind_in_df(self, value):
         # Create a copy of the original DataFrame
        self.modified_kinds_df = self.initial_kinds_df.copy()

        # Modify the new DataFrame
        self.modified_kinds_df['Kinds'] = value


    def change_kinds_on_ODMC(self, value, mode):
        '''
        mode= "Global" or "Local"
        '''
        if mode == "Global":
            self.change_kind_in_df(value)
        # Convert DataFrame rows to a formatted string
        self.modified_kinds_str= "\r\n".join([f"{row[0]},{row[1]},{row[2]}" for row in self.modified_kinds_df.itertuples(index=False)])

        # Add the final \r\n if required
        self.modified_kinds_str += "\r\n"
        # Write the result to a text file
        with open("kinds_modified_str.txt", "w") as file:
            file.write(self.modified_kinds_str)

        payload = {
                "TableHierarchies": self.table_hierarchies_str,
                "TableDataHeader":self.table_levels_str,
                "TableData": self.modified_kinds_str
        }
        logging.info("start post request: uploading kind values to a table")
        post_result = self.client.send_post_request(self.table_url, payload)
        logging.info("completed post request")
        if post_result is None:
                logging.error("CFs kind table update failed")
                sys.exit(1)


    def initialize_funcs(self):
        '''
        call class functions
        '''
        logging.info(f"URL: {self.table_url}")
        table_data = self.fetch_data()
        if table_data is None:
            logging.error("Get request for the CF Kinds table URL failed. Please ensure the CF Kinds URL is properly configured (Table ID needs to be vaild)")
            sys.exit(1)
        self.get_table_levels(table_data['items']) 
        self.create_kinds_table_data_df(table_data)
                    

 