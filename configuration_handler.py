################################################################
# Automatic Parameter Tuning: Configuration Handler
#   1. Class "ConfigLoaderJson"
#   2. Class "ConfigYaml"
#   3. Class "APIClient"
#   4. class "PlanExecutionError"
#Last Update: 16-Dec-2024
#Environment: Stryker Dev3
################################################################

import requests
from requests.auth import HTTPBasicAuth
import json
import yaml
import logging
import sys

# Configure logging
logging.basicConfig(
    filename='Logs.log',
    filemode='w',  # Overwrite the log file on each run
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ConfigLoaderJson:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = None
        
    def load_config(self):
        """
        Load configuration from the specified JSON file.
        """
        try:
            with open(self.config_file, "r") as file:
                self.config = json.load(file)
        except FileNotFoundError:
            logging.error(f"Configuration file '{self.config_file}' not found.")
            sys.exit(1)  # Exit the program
        except json.JSONDecodeError:
            logging.error(f"Error decoding the configuration file '{self.config_file}'.")
            sys.exit(1)  # Exit the program

    def get(self, key, default=None):
        """
        Get a value from the configuration, or return a default value if not found.
        """
        if self.config is None:
            logging.error("Configuration not loaded. Call load_config() first.")
            sys.exit(1)  # Exit the program
        return self.config.get(key, default)


class ConfigYaml:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = None



    def load_config(self):
        """
        Load configuration from the specified YAML file.
        """
        try:
            with open(self.config_file, "r") as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            logging.error(f"Configuration file '{self.config_file}' not found.")
            sys.exit(1)  # Exit the program
        except yaml.YAMLError as e:
            logging.error(f"Error parsing the configuration file '{self.config_file}': {e}")
            sys.exit(1)  # Exit the program

    def get(self, key, default=None):
        """
        Get a value from the configuration, or return a default value if not found.
        """
        if self.config is None:
            logging.error("Configuration not loaded. Call load_config() first.")
            sys.exit(1)  # Exit the program
        return self.config.get(key, default)


class APIClient:
    """
    A client for making API requests with basic authentication.

    Attributes:
        base_url (str): The base URL for the API.
        auth (HTTPBasicAuth): Basic authentication credentials.
    """
    def __init__(self, base_url, username, password):
        """
        Initializes the APIClient with the base URL and authentication credentials.

        Args:
            base_url (str): The base URL for the API.
            username (str): The username for basic authentication.
            password (str): The password for basic authentication.
        """
        self.base_url = base_url
        self.auth = HTTPBasicAuth(username, password)

    def send_get_request(self, url):
        """
        Sends a GET request to the specified URL.

        Args:
            url (str): The endpoint URL for the GET request.

        Returns:
            dict or None: The JSON response from the server if successful, None otherwise.
        """
        try:
            # Send a GET request with authentication
            response = requests.get(url, auth=self.auth)
            response.raise_for_status() # Raise HTTPError for bad responses
            return response.json()  # Return the JSON response
        except requests.exceptions.RequestException as e:
            logging.error(f"GET request error: {e}")
            return None

    def send_post_request(self, url, payload):
        """
        Sends a POST request to the specified URL with a JSON payload.

        Args:
            url (str): The endpoint URL for the POST request.
            payload (dict): The JSON payload to send with the request.

        Returns:
            dict or None: The JSON response from the server if successful, None otherwise.
        """
        headers = {"Content-Type": "application/json"}
        try:
            # Send a POST request with authentication and headers
            response = requests.post(url, auth=self.auth, headers=headers, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses
            return response.json()  # Return the JSON response
        except requests.exceptions.RequestException as e:
            logging.error(f"POST request error: {e}")
            return None

    def send_patch_request(self, url, payload):
        """
        Sends a PATCH request to the specified URL with a JSON payload.

        Args:
            url (str): The endpoint URL for the PATCH request.
            payload (dict): The JSON payload to send with the request.

        Returns:
            str: "Good" if the request is successful.
        """
        headers = {"Content-Type": "application/json"}
        try:
            # Send a PATCH request with authentication and headers
            response = requests.patch(url, auth=self.auth, headers=headers, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses
            return "Good"   # Indicate success
        except requests.exceptions.RequestException as e:
            logging.error(f"PATCH request error: {e}")
            return None

class PlanExecutionError(Exception):
    """Custom exception to indicate plan execution issues. Used when plan's running time exceeds limitations"""
    pass

