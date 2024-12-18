################################################################
# Automatic Parameter Tuning: Valudations
#   1. function: validate_parameter_values
#   2. function: get_valid_float
#Last Update: 16-Dec-2024
#Environment: Stryker Dev3
################################################################


import pandas as pd
import logging
import sys

def validate_parameter_values(parameters_input):
    """
    Validate the parameter values provided in the input against predefined constraints.

    This function ensures that:
    - Parameter values are non-negative.
    - Parameter values do not exceed their maximum allowed values.
    - Binary and categorical parameters only contain allowed values.

    Args:
        parameters_input (dict): A dictionary where keys are parameter names and values are lists of values for validation.

    Raises:
        SystemExit: If any parameter value violates the defined constraints.
    """

    # Define the limitations for parameters, including their maximum allowed value and type
    limitastion_df = pd.DataFrame([
        {'name': 'EnableNonNegRegr', 'max': 1, 'type': 'binary'},
        {'name': 'GlobalAllocationPeriods', 'max': float('inf'), 'type': 'num'},
        {'name': 'PeriodsUntilActive', 'max': float('inf'), 'type': 'num'},
        {'name': 'PeriodsUntilInactive', 'max': float('inf'), 'type': 'num'},
        {'name': 'IntermitCriterion', 'max': 99, 'type': 'num' }, 
        {'name': 'TooFew', 'max': float('inf'), 'type': 'num'},
        {'name': 'WriteFit', 'max': 1, 'type': 'binary'},
        {'name': 'DetectOutlier	', 'max': 1, 'type': 'binary'},
        {'name': 'OutlierSensitivity', 'max': float('inf'), 'type': 'num'},
        {'name': 'OutlierStdErr', 'max': float('inf'), 'type': 'num'},
        {'name': 'OutliersPercent', 'max': 100, 'type': 'num'},
        {'name': 'RemoveExtremeOutlier', 'max': 1, 'type': 'binary'},
        {'name': 'EnableFitValidation', 'max': 1, 'type': 'binary'},
        {'name': 'EnableForecastValidation', 'max': 1, 'type': 'binary'},
        {'name': 'FitTestPeriod', 'max': float('inf'), 'type': 'num'},
        {'name': 'FitValidationSensitivity', 'max': float('inf'), 'type': 'num'},
        {'name': 'ForecastTestPeriod', 'max': float('inf'), 'type': 'num'},
        {'name': 'ForecastValidationSensitivity', 'max': float('inf'), 'type': 'num'},
        {'name': 'SmoothCroston', 'max': 1, 'type': 'binary'},
        {'name': 'SmoothIntermittent', 'max': 1, 'type': 'binary'},
        {'name': 'CutLeadingZeros', 'max': 1, 'type': 'binary'},
        {'name': 'FillMissingMethod', 'max': 1, 'type': '0-1-2'},
        {'name': 'AllowNegativeForecast', 'max': 1, 'type': 'binary'},
    ])

    # Iterate through the input parameters to validate their values    
    for parameter_name, values_lst in parameters_input.items():
        # Locate the corresponding record in the limitations DataFrame
        record = limitastion_df[limitastion_df['name'] == parameter_name]


        if record.empty:
            logging.error(f"Parameter {parameter_name} is not defined in the limitations.")
            sys.exit(1)


        # Extract the maximum allowed value and type of the parameter
        max_limit = float(record['max'].iloc[0])
        param_type = record['type'].iloc[0]

        # Validate each value for the parameter
        for value in values_lst:
            # Check for non-negative values
            if value < 0:
                logging.error(f"Invalid value for {parameter_name}: {value}. Parameter's value must be non-negative.")
                sys.exit(1)

            # Check for values exceeding the maximum limit
            if value > max_limit:
                logging.error(f"Invalid value for {parameter_name}: {value}. Parameter's value must not exceed the maximum of {max_limit}.")
                sys.exit(1)

            # Validate binary parameters
            if param_type == 'binary' and value not in [0, 1]:
                logging.error(f"Invalid value for {parameter_name}: {value}. Binary parameters can only be 0 or 1.")
                sys.exit(1)

            # Validate 0-1-2 categorical parameters
            elif param_type == '0-1-2' and value not in [0, 1, 2]:
                logging.error(f"Invalid value for {parameter_name}: {value}. Allowed values are 0, 1, or 2.")
                sys.exit(1)



def get_valid_float(config, key, default=0.999):
    """
    Retrieve a valid float value from the configuration.

    This function attempts to extract a value from the cconfig`,
    convert it to a float, and scale it down by dividing by 100. If the value is invalid (non-numeric or missing),
    it logs an error and returns a default value.

    Returns:
        float: The valid float value scaled down by dividing by 100, or the default value.
    """

    try:
        # Retrieve the value associated with the key, convert it to float, and scale it down
        return float(config.get(key))/100
    except (ValueError, TypeError):
        # Log an error if the value cannot be converted to float or is None
        logging.error(f"Invalid value for {key}: {config.get(key)}. Defaulting to {default}.")
        return default



 
def transform_parameters(param_list):
    """
    Function to transform a list of parameters into a dictionary of tuning values.
    Each parameter in the input list contains a name, a minimum value, a maximum value,
    and a step size. The function generates a list of values for each parameter by 
    incrementing from the minimum value to the maximum value in steps and formats them.

    Args:
        param_list (list of dict): A list where each dictionary represents a parameter
                                   with keys: 'name', 'min', 'max', and 'step'.
    Returns:
        dict: A dictionary where the keys are parameter names, and the values are lists
              of generated values based on the min, max, and step size.
    """
    # Initialize an empty dictionary to store transformed parameters
    transformed = {}

    # Iterate through each parameter in the input list
    for param in param_list:
        # Extract the parameter's name, min value, max value, and step size
        name = param['name']
        min_val = param['min']
        max_val = param['max']
        step = param['step']

        # Generate a list of values starting from min_val to max_val with step size
        # Round each value to two decimal places for consistent formatting
        values = [round(min_val + i * step, 2) for i in range(int((max_val - min_val) / step) + 1)]

        # Map the generated values to the parameter's name in the dictionary
        transformed[name] = values
    return transformed


def validate_tune_by_wmape(tune_by_wmape):
    """
    Validates the input value for the 'Tune_by_wmape' parameter.
    The parameter must be either 'On' or 'Off'. If the value is invalid,
    an error is logged, and a ValueError is raised.

    Args:
        tune_by_wmape (str): The input value to validate. Expected values are 'On' or 'Off'.

    Raises:
        ValueError: If the input value is not 'On' or 'Off'.
    """
    # Check if the input value is not in the accepted list of values
    if tune_by_wmape not in ["On", "Off"]:
        logging.error(f"Invalid value for Tune_by_wmape: {tune_by_wmape}")
        raise ValueError("Tune_by_wmape must be 'On' or 'Off'")



def validate_tune_by_step(tune_by_step):
    """
    Validates the input value for the 'tune_by_step' parameter.
    The parameter must be either 'On' or 'Off'. If the value is invalid,
    an error is logged, and a ValueError is raised.

    Args:
        tune_by_step (str): The input value to validate. Expected values are 'On' or 'Off'.

    Raises:
        ValueError: If the input value is not 'On' or 'Off'.
    """
    # Check if the input value is not in the accepted list of values
    if tune_by_step not in ["On", "Off"]:
        logging.error(f"Invalid value for tune_by_step: {tune_by_step}")
        raise ValueError("tune_by_step must be 'On' or 'Off'")


# Validate and handle `number_of_top_offenders`
def validate_number_of_top_offenders(number_of_top_offenders_raw):
    """
    Validates and converts the input for the 'number_of_top_offenders' parameter.
    If the input is invalid, logs a warning and defaults the value to 0.

    Args:
        number_of_top_offenders_raw: The raw input value to validate and convert (user input). 
                                     Expected to be convertible to an integer.

    Returns:
        int: A valid integer value for 'number_of_top_offenders'. Defaults to 0 if the input is invalid.
    """
    try:
        # Attempt to convert the raw input to an integer
        number_of_top_offenders = int(number_of_top_offenders_raw)
        return number_of_top_offenders
    except (ValueError, TypeError):
        logging.warning(f"Invalid value for number_of_top_offenders: {number_of_top_offenders_raw}. Defaulting to 0.")

        # Default to 0 in case of invalid input
        number_of_top_offenders = 0
        return number_of_top_offenders

def validate_parameters_tuning_mode(parameters_tuning_mode):
    """
    Validates the input value for the 'parameters_tuning_mode' parameter.
    Ensures that the input is either 'Regular' or 'Cartesian'. If the value is invalid,
    an error is logged, and a ValueError is raised.

    Args:
        parameters_tuning_mode (str): The input value to validate. Expected values are 'Regular' or 'Cartesian'.

    Raises:
        ValueError: If the input value is not 'Regular' or 'Cartesian'.
    """
    # Check if the input value is not in the list of accepted value
    if parameters_tuning_mode not in ["Regular", "Cartesian"]:
        logging.error(f"Invalid value for Parameters_tuning_mode: {parameters_tuning_mode}")
        raise ValueError("Parameters_tuning_mode must be 'Regular' or 'Cartesian'")
    

def validate_run_mode(run_mode):
     if run_mode not in ["Dummy", "Parameters", "CFs", "Parameters then CFs", "CFs then Parameters"]:
        logging.error(f"Invalid value for run_mod: {run_mode}")
        raise ValueError("Parameters_tuning_mode must be 'Regular' or 'Cartesian'")
    