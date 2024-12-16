################################################################
# Automatic Parameter Tuning: Main
#   1. Read config parameters
#   2. Initialize class instance: client, profile plan, optimizer
#   3. start process by calling optimizer.functions_navigator()
#Last Update: 16-Dec-2024
#Environment: Stryker Dev3
################################################################


from configuration_handler import ConfigYaml, APIClient
from odmc_objects import Profile, Plan
from optimizer import ParameterOptimizer
from validation import validate_parameter_values, get_valid_float, transform_parameters, validate_tune_by_wmape, validate_number_of_top_offenders, validate_parameters_tuning_mode, validate_tune_by_step
import logging



if __name__ == "__main__":
    # Get user input from the config file
    config_loader = ConfigYaml("config.yaml")
    config_loader.load_config()
    username = config_loader.get("username")
    password = config_loader.get("password")
    servername = config_loader.get("servername")
    profile_id = config_loader.get("profile_id")
    plan_id = config_loader.get("plan_id")
    accuracy_table_id = config_loader.get("accuracy_table_id")
    top_offenders_table_id = config_loader.get("top_offenders_table_id")
    year_over_year_table_id = config_loader.get("year_over_year_table_id")
    parameters_input = config_loader.get("parameters", {})
    parameters_by_step = config_loader.get("parameters by step")  
    CF_kinds_table_id =config_loader.get("CF_kinds_table_id")
    kind_values_lst = config_loader.get("Kind Values")
    Configure_best_kind_IDs_on_ODMC = config_loader.get("Configure best kind IDs on ODMC")



    base_wmape = round(get_valid_float(config_loader, "Base_WMAPE"), 5)
    base_bias = round(get_valid_float(config_loader, "Base_Bias"), 5)
    validate_parameter_values(parameters_input)

    tune_by_wmape = config_loader.get("Tune_by_wmape")
    validate_tune_by_wmape(tune_by_wmape)

    tune_by_step = config_loader.get("Tune_by_step")
    validate_tune_by_step(tune_by_step)
    # Transform the parameters
    if tune_by_step == "On":
        parameters_input = transform_parameters(parameters_by_step)

    
    # Validate and handle `number_of_top_offenders`
    number_of_top_offenders_raw = config_loader.get("Number_of_top_offenders")
    number_of_top_offenders = validate_number_of_top_offenders(number_of_top_offenders_raw)


    # Validate and handle `parameters_tuning_mode`
    parameters_tuning_mode = config_loader.get("Parameters_tuning_mode")
    validate_parameters_tuning_mode(parameters_tuning_mode)


    # Dictionary of variable names and their values
    config_values = {
        "username": username,
        "password": password,
        "servername": servername,
        "profile_id": profile_id,
        "plan_id": plan_id,
        "accuracy_table_id": accuracy_table_id,
        "top_offenders_table_id": top_offenders_table_id,
        "year_over_year_table_id": year_over_year_table_id,
        "parameters_input": parameters_input,
        "number_of_top_offenders": number_of_top_offenders,
        "tune_by_wmape": tune_by_wmape,
        "base_wmape": base_wmape,
        "base_bias": base_bias,
        "parameters_tuning_mode": parameters_tuning_mode,
        "tune_by_step": tune_by_step, 
        "CF_kinds_table_id": CF_kinds_table_id,
        "kind_values_lst": kind_values_lst,
        "Configure_best_kind_IDs_on_ODMC": Configure_best_kind_IDs_on_ODMC
    }

    # Log each configuration value
    logging.info("Loaded configuration values:")
    for key, value in config_values.items():
        logging.info(f"{key}: {value}")
    
    logging.info("__________________________________________________________________________________________________________________________________________________")


    #basic url setup
    base_url = f"https://{servername}:443/fscmRestApi/resources/11.13.18.05"
    
    # API Client setup
    client = APIClient(base_url, username, password)

    # Profile setup
    profile = Profile(client, base_url, profile_id, parameters_input)

    #Plan setup
    plan = Plan(client, base_url, plan_id, profile.name,profile.profile_id, max_duration=2000)

    #Optimizer setup
    optimizer = ParameterOptimizer(client, profile, plan, base_url, accuracy_table_id, top_offenders_table_id, year_over_year_table_id, number_of_top_offenders, tune_by_wmape, base_wmape, base_bias, parameters_tuning_mode,what_to_tune = "CFs",CF_kinds_table_id = CF_kinds_table_id, kind_values_lst = kind_values_lst, Configure_best_kind_IDs_on_ODMC = Configure_best_kind_IDs_on_ODMC )
    #Start process
    optimizer.functions_navigator()

    #optimizer = ParameterOptimizer(client, profile, plan, base_url, accuracy_table_id, top_offenders_table_id, year_over_year_table_id, number_of_top_offenders, tune_by_wmape, base_wmape, base_bias, parameters_tuning_mode,what_to_tune = "Parameters",CF_kinds_table_id = CF_kinds_table_id, kind_values_lst = kind_values_lst, Configure_best_kind_IDs_on_ODMC = Configure_best_kind_IDs_on_ODMC)

    #optimizer.functions_navigator()
