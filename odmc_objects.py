################################################################
# Automatic Parameter Tuning: ODMC Objects
#   1. Class "Profile"
#   2. Class "Plan"
#Last Update: 16-Dec-2024
#Environment: Stryker Dev3
################################################################

import pandas as pd
import time

import logging
import sys


class Profile:
    def __init__(self, client, base_url, profile_id, parameters_input):
        '''
        Initialize the Profile class.

        :param client (APIClient): API client for handling HTTP requests
        :param profile_id (str): The unique identifier for the plan
        :param profile_url (str): URL for the profile's API endpoint. (300000049953003)
        :param parameters_input (dict): keys= parameters to tune (by name) as recieved by user input. values= parameters values to tune as recieved by user input
        :param profile_items_df(DataFrame): DataFrame version of parameters_input["items"]
        :param name(str): Name of the profile (default is None)
        'param parameters_dict(dict): keys=parameter id, value = parameter optional values list (as reciened as user input)
        '''
        self.client = client
        self.profile_id = profile_id
        self.profile_url = f"{base_url}/forecastingProfiles/{profile_id}/child/forecastingParameters"
        self.parameters_input  = parameters_input
        self.profile_items_df = None
        self.name = None
        self.parameters_dict = {}
        self.initial_parameters = None
        self.current_parameters_configuration = None
        self.current_parameters = None
        self.initialize_funcs()


    def get_profile_json(self):
        """
        Sends a GET request to fetch the profile JSON.

        Returns:
            dict: JSON response containing profile data.
        """
        return self.client.send_get_request(self.profile_url)   


    def json_to_df(self, json_data):
        """
        Converts JSON response into a DataFrame and extracts the profile name.

        Args:
            json_data (dict): JSON response containing profile data.

        Returns:
            pd.DataFrame: DataFrame of profile items or None if JSON is invalid.
        """
        if isinstance(json_data, dict) and "items" in json_data:
            if isinstance(json_data["items"], list):
                self.profile_items_df = pd.DataFrame(json_data["items"])    #create a df
                self.name = self.profile_items_df['ForecastingProfileName'].unique()[0]     #find the name of the profile
                return pd.DataFrame(json_data["items"])  # Convert items list to DataFrame
            else:
                print("Error: 'items' is not a list in the JSON data")
                return None
        print("Error: Unexpected JSON structure or missing 'items' key")
        return None
    

    def find_parameter_id(self, parameter_name):
        """
        Finds the parameter ID for a given parameter name.

        Args:
            parameter_name (str): Name of the parameter.

        Returns:
            int: Parameter ID.

        Raises:
            ValueError: If the parameter name is not found in the profile.
        """
        # extract the first match for parameter_name
        match = self.profile_items_df.loc[self.profile_items_df['ForecastingParameterName'] == parameter_name, 'ForecastingParameterId']
        if not match.empty:
            return match.iloc[0]  # Return the first matching ID
        else:
            raise ValueError(f"Parameter '{parameter_name}' not found in the the profile. Please ensure the parameter name is properly configured")
        
    
    def create_parameters_dict(self):   #keys=parameter id, values=parameters optional values
        '''
        Creates a dictionary:
        Keys = parameters ids
        Values = parameters optional values
        '''
        for parameter_name in self.parameters_input.keys():
            parameter_id = self.find_parameter_id(parameter_name)
            #self.parameters_dict[int(parameter_id)] = self.parameters_input.get(parameter_name)
            self.parameters_dict[(int(parameter_id), parameter_name)] = self.parameters_input.get(parameter_name)
        


    def initialize_funcs(self):
        """
        Initializes class attributes: name, profile_items_df, and parameters_dict.
        """
        profile_json = self.get_profile_json()
        if profile_json is None:
            logging.error("Get request for the profile URL failed. Please ensure the profile URL is properly configured (Profile ID needs to be vaild)")
            sys.exit(1)
        profile_items_df = self.json_to_df(profile_json)
        self.initial_parameters = profile_items_df[['ForecastingParameterName', 'ForecastingParameterValue']]
        self.create_parameters_dict()

    


    def get_current_profile_parameters_values(self):
        """
        Updates the current profile parameter values by sending a GET request.
        """
        profile_json = self.get_profile_json()
        if profile_json is None:
            logging.error("Get request for the profile URL failed. Please ensure the profile URL is properly configured(Profile ID needs to be vaild)")
            sys.exit(1)
        profile_items_df = self.json_to_df(profile_json)
        self.current_parameters = profile_items_df[['ForecastingParameterName', 'ForecastingParameterValue', 'ForecastingParameterId']]

    def change_parameter(self, parameter_id, value):
        """
        Updates the value of a specific parameter in the profile.

        Args:
            parameter_id (int): ID of the parameter to change.
            value (float): New value for the parameter.

        Raises:
            SystemExit: If the parameter update fails.
        """
        self.get_current_profile_parameters_values()
        current_value = self.current_parameters.loc[self.current_parameters['ForecastingParameterId'] == parameter_id,'ForecastingParameterValue'].values

                # Extract the value from the DataFrame
        current_value_array = self.current_parameters.loc[self.current_parameters['ForecastingParameterId'] == parameter_id,'ForecastingParameterValue'].values
        # extract a single element and convert it to float
        if current_value_array.size > 0:  # Ensure the array is not empty
            current_value = float(current_value_array[0])  # Extract the first element and convert to float
            print("-----------------------------")
            print(f'current_value: {current_value}')
            print(f'float(value): {float(value)}')
            print("-----------------------------")
            self.get_current_profile_parameters_values()
        else:
            current_value = None  # Handle cases where no match is found


        if current_value != float(value):
            parameter_url = f"{self.profile_url}/{parameter_id}"
            payload = {"ForecastingParameterValue": value}
            
            post_result = self.client.send_patch_request(parameter_url, payload)

            if post_result is None:
                logging.error("Parameter update failed. Please ensure the following are properly configured: profile URL, parameters names, parameters values")
                sys.exit(1)



class Plan:
    def __init__(self, client, base_url, plan_id, profile_name, profile_id, max_duration=None):
        '''
        Initialize the Plan class.

        :param client (APIClient): The API client used for sending requests to the server
        :param plan_id (str): The unique identifier for the plan.
        :param plan_url (str): The base URL for plan-related API operations.
        :param profile_id (int): The id of the forecasting profile used for forecasting
        :param profile_name (str): The Name of the forecasting profile used for forecasting.
        :param max_duration (int): The maximum allowed duration (in seconds) for waiting on operations.
        :param name (str): Plan name
        :param input_measure (str): The input Measure as specified in Edit Plan Options (such as "Final Bookings History")
        :param output_measure (str): The output Measure as specified in Edit Plan Options (such as "Bookings Forecast")
        :param historical_buckets (str): The number of historical buckets used for forecasting
        :param start_of_history_date (str): The first date of historical data seen by the engine
        :param end_of_history_date (str): The last date of historical data seen by the engine
        :param offset (str): The number of offset periods as specified in Edit Plan Options
        :param curr_plan_status (int): The current status of the plan run (values- 0: completed, 1: processing, 2: error, or 3: warning)
        :param Parameter_overrides (str): parameters overrides as specified in Edit Plan Options (example: "#BLOCKING=2#RESRESH=TRUE")
        :param job_id (str): process id of the last run
        :param plan_details_df (DataFrame): holds: Plan ID, Plan Name,Profile Name, Input Measure, Output Measure, Historical Buckets, Start of History Date, End of History Date, Offset,Parameter Overrides, Job I
        '''
        self.client = client
        self.plan_id = plan_id
        self.plan_url = f"{base_url}/demandPlans/{plan_id}"
        self.profile_id = profile_id
        self.profile_name = profile_name
        self.max_duration = max_duration

        # Initialize attributes
        self.name = None
        self.input_measure = None
        self.output_measure = None
        self.historical_buckets = None
        self.start_of_history_date = None
        self.end_of_history_date = None
        self.offset = None
        self.curr_plan_status = None
        self.Parameter_overrides = None
        self.job_id = None
        self.plan_details_df = None

         # Initialize plan attributes by fetching data
        self.initialize_funcs()

    def get_plan_json(self):
            """
            Fetches JSON response with plan attributes such as offset, end of history date, etc.

            Returns:
                dict: JSON response containing plan details.
            """
            return self.client.send_get_request(self.plan_url)

    def get_profile_details(self, json_data):
        """
        Populates class attributes from the JSON response.

        Args:
            json_data (dict): JSON response containing plan details.
        """
        self.name = json_data["PlanName"]
        self.curr_plan_status = int(json_data["PlanStatus"])    #status type=int
        self.Parameter_overrides = json_data["ParameterOverrides"]
        
        if isinstance(json_data, dict) and "ForecastProfiles" in json_data:
                forecast_profiles_lst = json_data["ForecastProfiles"].split(") (")  #split string to a list

                # Parse the forecast profiles to extract relevant details
                data_dict = {}  #Keys=profile ids, values= profile configuration element in Edit Plan Options (offset, end of history date, etc.)
                for item in forecast_profiles_lst:
                    values = item.strip('(').split(',')
                    profile_name = values[0]  # Assuming the 4th value is profile_id
                    data_dict[profile_name] = values
                

                # Look up details for the specified profile name
                target_profile_name = self.profile_name
                if target_profile_name in data_dict:
                    result = data_dict[target_profile_name]
                    self.input_measure = result[1]
                    self.output_measure = result[2]
                    self.historical_buckets = result[4]
                    self.start_of_history_date = result[5]
                    self.end_of_history_date = result[6]
                    self.offset = result[8]

                else:
                    print(f"Profile ID {target_profile_name} not found.")

    def initialize_funcs(self):
        """
        Initializes plan details by fetching JSON data and populating attributes.
        """
        plan_json = self.get_plan_json()
        if plan_json is None:
            logging.error("Get request for the plan URL failed. Please ensure the plan URL is properly configured (Plan ID needs to be vaild)")
            sys.exit(1)
        self.get_profile_details(plan_json)

    def create_plan_detaild_df(self):
        """
        Creates a DataFrame summarizing plan details for export.

        Attributes added:
            self.plan_details_df: A DataFrame containing plan details.
        """
        data = {
            "Plan ID": self.plan_id,
            "Plan Name": self.name,
            "Profile Name": self.profile_name,
            "Input Measure": self.input_measure,
            "Output Measure": self.output_measure,
            "Historical Buckets": self.historical_buckets,
            "Start of History Date": self.start_of_history_date,
            "End of History Date": self.end_of_history_date,
            "Offset": self.offset,
            "Parameter Overrides": self.Parameter_overrides,
            "Job ID": self.job_id,
        }

        # Convert the dictionary to a pandas DataFrame
        self.plan_details_df = pd.DataFrame([data])  # Wrap `data` in a list to create a single-row DataFrame

    
    
    def get_status(self):
        """
        Updates the current status of the plan.

        Status codes:
            0 - Completed
            1 - Processing
            2 - Error
            3 - Warning
        """
        Plan_URL_Response = self.client.send_get_request(self.plan_url)
        if Plan_URL_Response is None:
            logging.error("Get request for the plan URL failed. Please ensure the plan URL is properly configured (Plan ID needs to be vaild). Error when checking plan's status.")
            sys.exit(1)
        self.curr_plan_status = int(Plan_URL_Response['PlanStatus'])
    
    def wait_for_completion(self):
        """
        Waits until the plan execution is completed or an error/warning occurs.

        Raises:
            SystemExit: If the plan execution exceeds the max duration or encounters an error/warning.
        """
        start_time = time.time()
        while self.curr_plan_status == 1:
            elapsed_time = time.time() - start_time
            if time.time() - start_time > self.max_duration:
                error_message = (
                    f"Plan execution exceeded maximum duration.\n"
                    f"Elapsed Time: {elapsed_time:.2f} seconds\n"
                    f"Max Duration Allowed: {self.max_duration} seconds\n"
                    f"Check logs or retry the operation for more details."
                )
                logging.error(error_message)
                sys.exit(1)
            
            time.sleep(120)
            self.get_status()
            if self.curr_plan_status == 3:
                logging.error("Plan run finished with warnings")
                sys.exit(1)
            elif self.curr_plan_status == 2:
                logging.error("Plan run finished with errors")
                sys.exit(1)
            print("time: ", elapsed_time)
            print("status: ", self.curr_plan_status)
            if (self.curr_plan_status == 1):
                logging.info(f"Current status of the plan run: {self.curr_plan_status} (Running).")

    
    def launch(self):
        """
        Launches the plan (no collections) and waits for its completion.

        Returns:
            dict: Response from the POST request initiating the plan run.
        """

        launch_plan_url = f"{self.plan_url}/child/Runs"
        payload = {
            "Mode": "3",
            "ForecastProfiles": self.profile_name,
            "ArchivePlanFlag": False,
            "ForecastMethodsFlag": False,
            "CausalFactorsFlag": False
        }
        self.curr_plan_status = 1   # Set status to "Processing"
        launch_run_response = self.client.send_post_request(launch_plan_url, payload)
        if launch_run_response is None:
            logging.error("Post request for the plan URL failed. Please ensure the plan URL is properly configured (Plan ID needs to be vaild). Error when initiating a run.")
            sys.exit(1)
        self.wait_for_completion()
        self.job_id = launch_run_response['JobId']
        self.create_plan_detaild_df()
        return launch_run_response
    

    