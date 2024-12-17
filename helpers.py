################################################################
# Automatic Parameter Tuning
#   1. Change parameters values
#   2. Change CFs value
#Last Update: 16-Dec-2024
#Environment: Stryker Dev3
################################################################

from configuration_handler import PlanExecutionError
from tables import Table, Accuracy_Table, Top_Offenders_Table, Yoy_Table, Kinds_Table
import sys
from datetime import timedelta
import numpy as np
import pandas as pd
import logging


plan_id = "300000049960080"

class ExcelReportGenerator:
    """
    A utility class for generating and formatting Excel reports using pandas and xlsxwriter.
    
    Attributes:
        file_path (str): Path to the Excel file to be generated.
        writer (pd.ExcelWriter): The Excel writer object.
        workbook (xlsxwriter.Workbook): The workbook object for applying formats.
    """
    def __init__(self, file_path):
        """
        Initializes the ExcelReportGenerator with the file path and prepares the writer.

        Args:
            file_path (str): Path to the Excel file to be created or modified.
        """
        self.file_path = file_path
        self.writer = pd.ExcelWriter(file_path, engine="xlsxwriter")
        self.workbook = None

    def _set_workbook(self):
        """
        Ensures the workbook object is set for formatting operations.
        """
        if self.workbook is None:
            self.workbook = self.writer.book

    def set_column_width(self, sheet_name, columns, width):
        """
        Sets the width of specified columns in a worksheet.

        Args:
            sheet_name (str): Name of the worksheet.
            columns (str): Column range (e.g., "A:D").
            width (int): Desired column width.
        """
        self._set_workbook()
        worksheet = self.writer.sheets[sheet_name]
        worksheet.set_column(columns, width)

    def write_dataframe(self, dataframe, sheet_name, startrow=0, bold_text=None, header=True, index=False):
        """
        Writes a pandas DataFrame to an Excel worksheet and optionally adds a bold header.

        Args:
            dataframe (DataFrame): The DataFrame to write.
            sheet_name (str): Name of the worksheet.
            startrow (int, optional): Row index to start writing the DataFrame. Defaults to 0.
            bold_text (str, optional): Text to write in bold format above the DataFrame. Defaults to None.
            header (bool, optional): Whether to include DataFrame column headers. Defaults to True.
            index (bool, optional): Whether to include the DataFrame index. Defaults to False.
        """
        self._set_workbook()
        # Write the DataFrame to the Excel sheet
        dataframe.to_excel(
            self.writer, sheet_name=sheet_name, startrow=startrow, header=header, index=index)
        
        # Apply bold text above the DataFrame if specified
        worksheet = self.writer.sheets[sheet_name]
        if bold_text:
            bold_format = self.workbook.add_format({"bold": True})
            worksheet.write(startrow - 1, 0, bold_text, bold_format)

    def apply_format(self, worksheet_name, columns=None, cells=None, values = None, format_type=None):
        """
        Applies custom formatting to specified columns or cells in a worksheet.

        Args:
            worksheet_name (str): Name of the worksheet.
            columns (str, optional): Column range to format (e.g., "A:D"). Defaults to None.
            cells (tuple, optional): Cell coordinates to format (e.g., (row, col)). Defaults to None.
            values (numeric, optional): The value to write at the specified cell. Defaults to None.
            format_type (str, optional): The type of format to apply ('comma_format', 'percentage_format'). Defaults to None.
        """
        # Define custom formats
        self._set_workbook()
        if format_type == "comma_format":
            fmt = self.workbook.add_format({'num_format': '#,##0'})
        elif format_type == "percentage_format":
            fmt = self.workbook.add_format({'num_format': '0.00%'})

        # Apply formatting
        worksheet = self.writer.sheets[worksheet_name]
        if columns:
            worksheet.set_column(columns, None, fmt)    # Apply format to columns
        if cells:
            worksheet.write_number(cells, values, fmt)  # Apply format to specific cell
        



    def highlight_rows(self, sheet_name, dataframe, startrow, highlight_condition):
        """
        Highlights rows in an Excel worksheet based on a condition.

        Args:
            sheet_name (str): Name of the worksheet.
            dataframe (DataFrame): The DataFrame used to determine row indices for highlighting.
            startrow (int): The row index in the worksheet where the DataFrame starts.
            highlight_condition (callable): A function that takes a row (as a pandas Series) 
                                            and returns True if the row should be highlighted.
        """
        self._set_workbook()    # Ensure the workbook object is initialized
        worksheet = self.writer.sheets[sheet_name]
        yellow_format = self.workbook.add_format({"bg_color": "#FFFF00"})    # Define yellow background format

        # Iterate over the rows of the DataFrame
        for row in range(len(dataframe)):
            if highlight_condition(dataframe.iloc[row]):    # Check if the row meets the highlight condition
                worksheet.set_row(startrow + row, None, yellow_format)  # Apply yellow highlight to the row

    def save(self):
        """
        Saves and closes the Excel file.
        """
        self.writer.close()



def Export_Accuracy_to_Excel(
    plan_details_df, current_parameters, merged_df, accuracy_df_for_export, grouped_accuracy,
    methods_and_levels_dfs, top_offenders_oos, top_offenders_table, y_axis_cols, 
    is_valid_top_offenders, yoy_df, is_valid_yoy, time_level, input_measure, output_measure,
    parameter_name, parameter_value, parameters_tuning_mode, ind, Top_offenders_per_run_dict, what_to_tune, kind_value, file_path):
    """
    Main function to generate an Excel report with multiple sheets.
    """
    generator = ExcelReportGenerator(file_path)
    # Call individual functions for different sections
    write_plan_details(generator, plan_details_df, current_parameters)
    write_data_sheet(generator, merged_df)
    write_accuracy_sheet(generator, accuracy_df_for_export, grouped_accuracy)
    write_methods_and_levels(generator, methods_and_levels_dfs)
    if is_valid_yoy == "Valid":
        write_year_over_year(generator, yoy_df, plan_details_df, time_level, input_measure, output_measure)

    Top_offenders_per_run_dict_updated = write_top_offenders(generator, top_offenders_oos, top_offenders_table, y_axis_cols, is_valid_top_offenders, plan_details_df,
                        time_level, parameter_name, parameter_value, parameters_tuning_mode, ind, Top_offenders_per_run_dict, output_measure, what_to_tune, kind_value)

    # Save the report
    generator.save()

    return Top_offenders_per_run_dict_updated


def write_plan_details(generator, plan_details_df, current_parameters):
    """Writes the plan details to the Excel sheet."""
    generator.write_dataframe(
        plan_details_df.T, "Plan Details", bold_text="Plan Details:", header=False, index=True
    )
    generator.write_dataframe(
        current_parameters, "Plan Details", startrow=len(plan_details_df.T) + 4
    )
    generator.set_column_width("Plan Details", "A:A", 33)
    generator.set_column_width("Plan Details", "B:B", 25)


def write_data_sheet(generator, merged_df):
    """Writes the data sheet with merged data."""
    generator.write_dataframe(merged_df, "Data", bold_text="Merged Data:")
    generator.set_column_width("Data", "B:P", 15)


def write_accuracy_sheet(generator, accuracy_df_for_export, grouped_accuracy):
    """Writes the accuracy results and grouped accuracy to the Excel sheet."""
    generator.write_dataframe(
        accuracy_df_for_export, "Accuracy", bold_text="Accuracy Results:"
    )
    generator.write_dataframe(
        grouped_accuracy,
        "Accuracy",
        startrow=len(accuracy_df_for_export) + 4,
        bold_text="Grouped Accuracy Results:"
    )
    generator.set_column_width("Accuracy", "A:A", 20)
    generator.set_column_width("Accuracy", "B:C", 12)
    generator.apply_format("Accuracy", columns="B:C", cells=None, values=accuracy_df_for_export.iloc[0, 1], format_type="percentage_format")
    for row in range(2, 5):
        generator.apply_format("Accuracy", columns=None, cells=f"B{row}", values=accuracy_df_for_export.iloc[row - 2, 1], format_type="comma_format")


def write_methods_and_levels(generator, methods_and_levels_dfs):
    """Writes the methods and levels pivot tables."""
    startrow = 1
    for pivot_name, df in methods_and_levels_dfs.items():
        if df is not None and not df.empty:
            generator.write_dataframe(
                df,
                "Methods-Levels Pivots",
                startrow=startrow,
                bold_text=f"{pivot_name} Pivot:"
            )
            startrow += len(df) + 4

def yoy_calculations(yoy_df, closest_date, time_level, input_measure, output_measure):

        # Calculate target dates
        date_before_target = closest_date - pd.Timedelta(days=365)
        # Ensure date_before_target is a pandas Timestamp
        date_before_target = pd.Timestamp(date_before_target)

        date_after_target = closest_date + pd.Timedelta(days=365)
        date_after_target = pd.Timestamp(date_after_target)
        # Find the closest valid date AFTER date_before_target
        closest_date_before = yoy_df.loc[yoy_df[time_level] >= date_before_target, "Month"].min()
        # Find the closest valid date AFTER date_after_target
        closest_date_after = yoy_df.loc[yoy_df[time_level] >= date_after_target, "Month"].min()

        
        # Create the two subsets
        subset_df_history = yoy_df[(yoy_df[time_level] >= closest_date_before) & (yoy_df[time_level] <= closest_date)].copy()
        subset_df_forecast = yoy_df[(yoy_df[time_level] > closest_date) & (yoy_df[time_level] <= closest_date_after)].copy()

        # Ensure the two subsets have the same number of records
        len_history, len_forecast = len(subset_df_history), len(subset_df_forecast)

        if len_history > len_forecast:
            # Drop oldest record(s) from the first subset
            subset_df_history = subset_df_history.iloc[-len_forecast:]  # Keep the most recent 'len_forecast' records
        elif len_forecast > len_history:
            # Drop newest record(s) from the second subset
            subset_df_forecast = subset_df_forecast.iloc[:len_history]  # Keep the oldest 'len_history' records

       

        # Calculate Total History 1 year for Subset 1
        total_history = subset_df_history[input_measure].sum()

        # Calculate Total History 1 year for Subset 2
        total_forecast = subset_df_forecast[output_measure].sum()

        # Calculate % Difference
        if total_history != 0:
            percentage_difference = (total_history - total_forecast) / total_history
        else:
            percentage_difference = np.nan  # Set to null if denominator is 0


        return total_history, total_forecast, percentage_difference


def write_year_over_year(generator, yoy_df, plan_details_df, time_level, input_measure, output_measure):
    """
    Writes the Year-over-Year sheet and creates a chart with a vertical line for the closest date.

    Args:
        generator (ExcelReportGenerator): The Excel generator object.
        yoy_df (DataFrame): Year-over-Year data DataFrame.
        plan_details_df (DataFrame): Plan details DataFrame.
    """
    sheet_name = "Year over year"

    # Write the Year-over-Year data
    


    yoy_df[time_level] = pd.to_datetime(yoy_df[time_level], errors="coerce")
    # Extract the End of History date
    end_of_history_date_str = plan_details_df.loc[0, "End of History Date"]
    end_of_history_date = pd.to_datetime(end_of_history_date_str)
    closest_date = yoy_df[yoy_df[time_level] >= end_of_history_date][time_level].min()


    # Add "End of History" column to yoy_df
    yoy_df["End of History"] = yoy_df[time_level].apply(
        lambda x: 1 if x == closest_date else None
    )
    
    total_history, total_forecast, percentage_difference = yoy_calculations(yoy_df, closest_date, time_level, input_measure, output_measure)


    yoy_df[time_level] = yoy_df[time_level].dt.date

    generator.write_dataframe(yoy_df, sheet_name)    

    # Create a line chart
    chart = generator.workbook.add_chart({'type': 'line'})

    # Add data series for columns B and C (assumes these are the main series)
    chart.add_series({
        'name':       [sheet_name, 0, 1],
        'categories': [sheet_name, 1, 0, len(yoy_df), 0],
        'values':     [sheet_name, 1, 1, len(yoy_df), 1],
    })
    chart.add_series({
        'name':       [sheet_name, 0, 2],
        'categories': [sheet_name, 1, 0, len(yoy_df), 0],
        'values':     [sheet_name, 1, 2, len(yoy_df), 2],
    })

    # Add a vertical line using a clustered column chart
    chart_vertical = generator.workbook.add_chart({'type': 'column'})
    chart_vertical.add_series({
        'name':       'End of History',
        'categories': [sheet_name, 1, 0, len(yoy_df), 0],
        'values':     [sheet_name, 1, yoy_df.columns.get_loc("End of History"), len(yoy_df), yoy_df.columns.get_loc("End of History")],
        'y2_axis': True,  # Assign to secondary axis
        'fill': {'color': 'red'},
        'border': {'none': True},
    })

    # Combine the main chart with the vertical line chart
    chart.combine(chart_vertical)

    # Configure chart properties
    chart.set_title({'name': 'Year-over-Year'})
    chart.set_x_axis({'name': 'Date', 'num_format': 'dd/mm/yyyy'})
    chart.set_y_axis({'name': 'Values', 'major_gridlines': {'visible': False}})
    chart.set_legend({'position': 'bottom'})

    # Configure the secondary y-axis for the vertical line
    chart_vertical.set_y2_axis({
        'max': 1,
        'label_position': 'none',
        'major_gridlines': {'visible': False},
    })

    # Insert the chart into the worksheet
    generator.writer.sheets[sheet_name].insert_chart('J4', chart, {
        'x_offset': 0,
        'y_offset': 0,
        'x_scale': 2.2,
        'y_scale': 1.5
    })
    chart.set_y_axis({'num_format': '#,##0'})

        # Write total_history, total_forecast, and percentage_difference to specific cells
    worksheet = generator.writer.sheets[sheet_name]

    # Define number and percentage formats
    numeric_format = generator.workbook.add_format({'num_format': '#,##0'})       # Numeric format: 100,000
    percentage_format = generator.workbook.add_format({'num_format': '0.00%'})    # Percentage format: 12.82%

    # Write the values to cells K28, K29, K30
    worksheet.write('K28', total_history, numeric_format)          # Total history
    worksheet.write('K29', total_forecast, numeric_format)         # Total forecast
    worksheet.write('K30', percentage_difference, percentage_format)  # % Difference

    # Optional: Add labels for clarity in cells J28, J29, J30
    worksheet.write('J28', 'Total History 1 year before End of History:', generator.workbook.add_format({'bold': True}))
    worksheet.write('J29', 'Total Forecast 1 year after End of History:', generator.workbook.add_format({'bold': True}))
    worksheet.write('J30', 'Percentage Difference:', generator.workbook.add_format({'bold': True}))

    


def add_vertical_line_to_chart(generator, sheet_name, subset_df, x_axis_range, column_name, color, label, combine_chart=None):
    """
    Adds a vertical line to the chart using a clustered column chart.

    Args:
        generator (ExcelReportGenerator): The Excel generator object.
        sheet_name (str): The name of the Excel sheet.
        subset_df (DataFrame): The DataFrame containing the data.
        x_axis_range (str): The range for the X-axis categories.
        column_name (str): The column in the DataFrame used for the vertical line.
        color (str): The fill color for the vertical line.
        label (str): The label for the series.
        combine_chart (xlsxwriter.chart.Chart, optional): The chart to combine the vertical line with.
    
    Returns:
        xlsxwriter.chart.Chart: The updated chart object.
    """
    # Create the vertical line chart
    chart_column = generator.workbook.add_chart({'type': 'column'})
    column_index = subset_df.columns.get_loc(column_name) + 1
    value_range = f"='{sheet_name}'!${chr(64 + column_index)}$2:${chr(64 + column_index)}${len(subset_df) + 1}"
    chart_column.add_series({
        'name': label,
        'categories': x_axis_range,
        'values': value_range,
        'y2_axis': True,  # Assign to secondary axis
        'fill': {'color': color},
        'border': {'none': True},
    })

    # Configure the secondary Y-axis
    chart_column.set_y2_axis({
        'max': 1,
        'label_position': 'none',
        'major_gridlines': {'visible': False},
    })

    # Combine the chart if provided
    if combine_chart:
        combine_chart.combine(chart_column)
    return chart_column


def process_combination_chart(generator, subset_df, x_axis_range, y_axis_cols, sheet_name):
    """
    Processes the chart creation for each combination, including OOS and End of History lines.

    Args:
        generator (ExcelReportGenerator): The Excel generator object.
        subset_df (DataFrame): The subset DataFrame for the combination.
        x_axis_range (str): The range for the X-axis categories.
        y_axis_cols (list): The list of Y-axis columns to plot.
        sheet_name (str): The name of the Excel sheet for the combination.
    """
    # Create the line chart
    chart = generator.workbook.add_chart({"type": "line"})

    # Add Y-axis data series
    for y_axis_col in y_axis_cols:
        y_axis_col_index = subset_df.columns.get_loc(y_axis_col) + 1
        y_axis_range = f"='{sheet_name}'!${chr(64 + y_axis_col_index)}$2:${chr(64 + y_axis_col_index)}${len(subset_df) + 1}"
        chart.add_series({
            "name": f"='{sheet_name}'!${chr(64 + y_axis_col_index)}$1",
            "categories": x_axis_range,
            "values": y_axis_range,
        })

    # Add OOS vertical line
    chart_OOS = add_vertical_line_to_chart(
        generator, sheet_name, subset_df, x_axis_range, column_name="OOS", color="green", label="OOS", combine_chart=chart
    )

    # Add End of History vertical line
    chart_End_of_History = add_vertical_line_to_chart(
        generator, sheet_name, subset_df, x_axis_range, column_name="End of History", color="red", label="End of History", combine_chart=chart
    )

    # Combine all charts
    #chart.combine(chart_End_of_History)
    chart.combine(chart_OOS)

    # Configure chart properties
    chart.set_title({'name': sheet_name})
    chart.set_x_axis({'name': 'Date', 'major_gridlines': {'visible': False}})
    chart.set_y_axis({'num_format': '#,##0'})




    # Insert the chart into the worksheet
    generator.writer.sheets[sheet_name].insert_chart("J4", chart, {
        "x_offset": 0,
        "y_offset": 0,
        "x_scale": 2.2,
        "y_scale": 1.5
    })


def process_top_offenders(generator, top_offenders_table, plan_details_df, y_axis_cols, top_offenders_oos, time_level, Top_offenders_per_run_dict,
                          parameter_name, parameter_value, parameters_tuning_mode, ind, output_measure, what_to_tune, kind_value):
    """
    Processes and generates charts for each combination in the top offenders table.

    Args:
        generator (ExcelReportGenerator): The Excel generator object.
        top_offenders_table (DataFrame): The top offenders data table.
        plan_details_df (DataFrame): Plan details DataFrame.
        y_axis_cols (list): List of Y-axis columns for charting.
    """
    print(f'in process_top_offenders: Top_offenders_per_run_dict = {Top_offenders_per_run_dict}')
    for combination in top_offenders_oos["Combination"].unique():
        # Filter the DataFrame for the current combination
        subset_df = top_offenders_table[top_offenders_table["Combination"] == combination].copy()
        subset_df[time_level] = pd.to_datetime(subset_df[time_level], errors="coerce")

        # Extract End of History date and populate the relevant columns
        end_of_history_date_str = plan_details_df.loc[0, "End of History Date"]
        target_date = pd.to_datetime(end_of_history_date_str)
        closest_date = subset_df[subset_df[time_level] >= target_date][time_level].min()

        if pd.notnull(closest_date):
            subset_df["End of History"] = subset_df[time_level].apply(lambda x: 1 if x == closest_date else None)


                    
        # Step 1: Filter columns that end with "_History"
        history_columns = [col for col in top_offenders_oos.columns if col.endswith("_History")]

        # Step 2: Remove the suffix "_History"
        history_dates = [col.replace("_History", "") for col in history_columns]

        # Step 3: Convert the dates to timestamps
        history_timestamps = pd.to_datetime(history_dates, format="%Y-%m-%d")

        # Step 4: Find the min and max dates
        min_OOS_date = history_timestamps.min()
        max_OOS_date = history_timestamps.max()
        OOS_Dates = [min_OOS_date, max_OOS_date]

        
        # Add OOS column based on history dates


        subset_df["OOS"] = subset_df[time_level].apply(
            lambda x: 1 if x in OOS_Dates else None
        )



        subset_df["OOS"] = subset_df[time_level].apply(lambda x: 1 if x in [min_OOS_date, max_OOS_date] else None)
        # Convert "Month" to date type
        subset_df[time_level] = subset_df[time_level].dt.date

        #for Top Offenders comparison by run:
        # Append the dictionary to the list within the dictionary `Top_offenders_per_run_dict`
        print(f'Before if: Top_offenders_per_run_dict = {Top_offenders_per_run_dict}')
        if combination not in Top_offenders_per_run_dict:
            Top_offenders_per_run_dict[combination] = []  # Initialize the list if not present

        # Append the dictionary
        if what_to_tune == "Parameters":
            if  parameters_tuning_mode == 'Cartesian':
                Top_offenders_per_run_dict[combination].append({
                ind: subset_df
            })

            else:
                Top_offenders_per_run_dict[combination].append({
                    (parameter_name, parameter_value): subset_df
                })

        elif what_to_tune == "CFs": 
            Top_offenders_per_run_dict[combination].append({
                kind_value: subset_df
            })
        # Write the subset DataFrame to a sheet
        sheet_name = str(combination)[:31]

        subset_df.to_csv(f'{sheet_name}.csv')
        

        generator.write_dataframe(subset_df, sheet_name, bold_text=f"Combination: {combination}")

        # Define the X-axis range
        x_axis_col_index = subset_df.columns.get_loc("Month") + 1
        x_axis_range = f"='{sheet_name}'!${chr(64 + x_axis_col_index)}$2:${chr(64 + x_axis_col_index)}${len(subset_df) + 1}"

        # Process the combination chart
        process_combination_chart(generator, subset_df, x_axis_range, y_axis_cols, sheet_name)

    return Top_offenders_per_run_dict


def write_top_offenders(generator, top_offenders_oos, top_offenders_table, y_axis_cols, is_valid_top_offenders, plan_details_df,
                        time_level, parameter_name, parameter_value, parameters_tuning_mode, ind, Top_offenders_per_run_dict, output_measure, what_to_tune, kind_value):
    """
    Write the Top Offenders analysis to the Excel sheet.
    """
    generator.write_dataframe(top_offenders_oos, "Top Offenders", bold_text="Top Offenders:")
    generator.set_column_width("Top Offenders", "A:P", 15)
    print(f'is_valid_top_offenders: {is_valid_top_offenders}')

    if is_valid_top_offenders == "Valid":
        print(f'in write_top_offenders -- valid: Top_offenders_per_run_dict = {Top_offenders_per_run_dict}')
        Top_offenders_per_run_dict_updated = process_top_offenders(generator, top_offenders_table, plan_details_df, y_axis_cols, top_offenders_oos, time_level, 
                                                                   Top_offenders_per_run_dict, parameter_name, parameter_value, parameters_tuning_mode, ind,
                                                                    output_measure, what_to_tune, kind_value)
    else:
        print('in else')
        Top_offenders_per_run_dict_updated = Top_offenders_per_run_dict
    return Top_offenders_per_run_dict_updated
        

def process_and_export(profile, plan, client, base_url, accuracy_table_id, top_offenders_table_id, year_over_year_table_id, number_of_top_offenders,
                        parameters_tuning_mode, what_to_tune, parameter_id = None, parameter_name = None, parameter_value = None, permutation = None,
                        ind = None,CF_kinds_table_id = None,kind_value = None, Top_offenders_per_run_dict = None ):
    """
    Processes a specific parameter key-value pair or kind value, launches the plan, and exports results to an Excel file.

    Args:
        profile (object): The profile object to modify parameters.
        plan (object): The plan object for launching and analyzing results.
        client (object): API client for interacting with the data service.
        base_url (str): Base URL for API requests.
        accuracy_table_id (str): ID of the accuracy table.
        top_offenders_table_id (str): ID of the top offenders table.
        year_over_year_table_id (str): ID of the Year-over-Year table.
        number_of_top_offenders (int): Number of top offenders to analyze.
        parameters_tuning_mode (str): Mode for tuning parameters ("Regular" or "Cartesian").
        what_to_tune (str): Indicates what to tune ("Parameters" or "CFs").
        parameter_id (str, optional): ID of the parameter to tune. Defaults to None.
        parameter_name (str, optional): Name of the parameter to tune. Defaults to None.
        parameter_value (any, optional): Value of the parameter to tune. Defaults to None.
        permutation (dict, optional): Permutations for Cartesian tuning. Defaults to None.
        ind (int, optional): Index of the permutation in Cartesian tuning. Defaults to None.
        CF_kinds_table_id (str, optional): ID of the CF kinds table. Defaults to None.
        kind_value (any, optional): Value of the CF kind to modify. Defaults to None.

    Returns:
        tuple: Results including WMAPE, bias, and optionally merged DataFrame and kinds table.
    """
    try:
        # Handle tuning for parameters
        if what_to_tune == "Parameters":
            # Update the profile with the current parameter value
            if parameters_tuning_mode == 'Cartesian':
                # Apply permutations for Cartesian tuning
                for (parameter_id, parameter_name), parameter_value in permutation.items():
                    # Call the function with the key and value
                    #profile.change_parameter(parameter_id, parameter_value)
                    pass
                logging.info(f'Launched plan: {plan.name}:  permutation Number: {ind} ')
            else:
                # Apply individual parameter tuning
                # Call the function with the key and value
                #profile.change_parameter(parameter_id, parameter_value)
                logging.info(f'Launched plan: {plan.name}:  {parameter_name} value: {parameter_value} ')
                pass

         # Handle tuning for CFs
        elif what_to_tune == "CFs":
            kinds_table = Kinds_Table(client, plan_id, plan.input_measure, plan.output_measure, CF_kinds_table_id)
            kinds_table.change_kinds_on_ODMC(kind_value, "Global")
            logging.info(f'Launched plan: {plan.name}:  kind value: {kind_value} ')
            

        # Launch and monitor the plan
        #launch_response = plan.launch()
        logging.info(f'Run is completed')


        # Accuracy table setup
        accuracy_table = Accuracy_Table(client, plan.plan_id, plan.input_measure, plan.output_measure, base_url, accuracy_table_id, number_of_top_offenders)

        if top_offenders_table_id is None:
            is_valid = "Not Valid"
            logging.info(f'top_offenders_table_id is invalid. Not exporting top offenders graphs')
        elif accuracy_table.number_of_top_offender <= 0:
            is_valid = "Not Valid"
            logging.info(f'Number of top offenders: {profile.number_of_top_offender}. Not exporting top offenders graphs ')
        else:
            is_valid = "Valid"
        
        
        # Top Offenders table setup
        top_offenders_table = Top_Offenders_Table(
            client, plan.plan_id, plan.input_measure, plan.output_measure, base_url, 
            top_offenders_table_id, accuracy_table.top_offenders, 
            accuracy_table.item_level, accuracy_table.location_level, 
            accuracy_table.time_level,is_valid
        )

        # Determine columns for top offenders charts
        x_axis_col = top_offenders_table.time_level if top_offenders_table.is_valid == "Valid" else None   
        y_axis_cols = [top_offenders_table.input_measure, top_offenders_table.output_measure] if top_offenders_table.is_valid == "Valid" else None



        # Year-over-Year table setup
        if year_over_year_table_id is not None:
            is_valid_yoy = "Valid"
            print("initializing Yoy Object")
            yoy_table = Yoy_Table(client, plan.plan_id, plan.input_measure, plan.output_measure, base_url, year_over_year_table_id, plan.end_of_history_date, is_valid_yoy)
            yoy_table.json_to_df()
        else:
            logging.info(f'year_over_year_table_id is invalid. Not exporting year over year graph')
            is_valid_yoy = "Not Valid"
            yoy_table = Yoy_Table(client, plan.plan_id, plan.input_measure, plan.output_measure, base_url, year_over_year_table_id, plan.end_of_history_date, is_valid_yoy)

        # Update profile and plan details   
        profile.get_current_profile_parameters_values()
        plan.create_plan_detaild_df()

        # Determine file path for the report
        if what_to_tune == "Parameters": 
            if parameters_tuning_mode == 'Cartesian':
                file_path = f"Permutation_{ind}_Dec012.xlsx"
            else:
                file_path = f"{parameter_name}_{parameter_value}_Dec012.xlsx"
        elif what_to_tune == "CFs":
            file_path = f"Kind_{kind_value}_Dec012.xlsx"

        # Export results to Excel
        Top_offenders_per_run_dict_updated = Export_Accuracy_to_Excel(
            plan.plan_details_df,
            profile.current_parameters[['ForecastingParameterName', 'ForecastingParameterValue']],
            accuracy_table.merged_df, 
            accuracy_table.accuracy_df_for_export, 
            accuracy_table.grouped_accuracy, 
            accuracy_table.methods_and_levels_dfs,
            accuracy_table.top_offenders,
            top_offenders_table.top_offenders_for_export,
            y_axis_cols,
            top_offenders_table.is_valid,
            yoy_table.yoy_df, 
            yoy_table.is_valid,
            accuracy_table.time_level,
            plan.input_measure,
            plan.output_measure,
            parameter_name,
            parameter_value,
            parameters_tuning_mode,
            ind,
            Top_offenders_per_run_dict,
            what_to_tune,
            kind_value,
            file_path
        )

         # Log success and return results
        if what_to_tune == "Parameters": 
            if what_to_tune == "Parameters": 
                print(f"Exported results to {file_path} for parameter {parameter_name} = {parameter_value}")
                logging.info(f'Exported results to {file_path} for parameter {parameter_name} = {parameter_value} ')
                return accuracy_table.wmape, accuracy_table.bias, Top_offenders_per_run_dict_updated
            else:
                print(f"Exported results to {file_path} for permutation {ind}")
                logging.info(f'Exported results to {file_path} for permutation {ind} ')
        elif what_to_tune == "CFs":
            print(f"Exported results to {file_path} for kind value {kind_value} ")
            logging.info(f'Exported results to {file_path} for kind value {kind_value} ')
            return accuracy_table.wmape, accuracy_table.bias, accuracy_table.merged_df, kinds_table, Top_offenders_per_run_dict_updated

    except PlanExecutionError as e:
        # Handle plan execution errors
        print(f"Error for parameter {parameter_name} = {parameter_value}: {e}")
        if what_to_tune == "Parameters": 
            if parameters_tuning_mode == 'Cartesian':
                logging.info(f'Error for permutation {permutation} : {e} ')
            else:
                logging.info(f'Error for parameter {parameter_name} = {parameter_value}: {e} ')
        elif what_to_tune == "CFs":
            logging.info(f'Error for CF Kind: {kind_value}: {e} ')


def highlight_differences(worksheet, data, start_row, formats): 
    """
    Highlight rows where 'Best?' column is 'Best' or where 'Initial' and 'Final' columns differ.
    """
    for row in range(len(data)):
        highlight = False
        
        # Check if "Best?" column exists and the value is "Best"
        if "Best?" in data.columns:
            if data.iloc[row]["Best?"] == "Best":
                highlight = True
        
        # Check if 'ForecastingParameterValue Initial' and 'Final' differ (if those columns exist)
        elif "ForecastingParameterValue Initial" in data.columns and "ForecastingParameterValue Final" in data.columns:
            if data.iloc[row]["ForecastingParameterValue Initial"] != data.iloc[row]["ForecastingParameterValue Final"]:
                highlight = True

        # Write the row, applying highlight if necessary
        for col in range(data.shape[1]):  # Iterate through all columns (A, B, C, D, E, etc.)
            cell_value = data.iloc[row, col]
            if highlight:
                worksheet.write(start_row + 1 + row, col, cell_value, formats['yellow'])
            else:
                worksheet.write(start_row + 1 + row, col, cell_value)



def write_data_to_excel(writer, sheet_name, dataframes, headers, formats):
    """
    Write multiple dataframes with headers to a specified sheet in the Excel file.
    """
    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet
    
    # Set column widths
    worksheet.set_column('A:D', 33)  # Adjust as needed
    current_row = 0

    for i, (df, header) in enumerate(zip(dataframes, headers)):
        if header:
            worksheet.write(current_row, 0, header, formats['bold'])
            current_row += 1
       
        df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, header=True, index=False)
        

        # Calculate the range for formatting percentages
        data_start_row = current_row + 1  # Data starts one row below the header
        data_end_row = current_row + len(df)  # Ends after the last record
        current_row += len(df) + 3  # Move to the next section, add 2 blank rows for spacing

        columns_to_format = {
                            'Tuned Parameters Report': [2, 3],  # Columns C and D
                            'Tuned CF kinds Report': {0: [2], 1: [1, 2]}  # Format C for i=0, B and C for i=1
                                    }

        if sheet_name in columns_to_format:
            columns_inds = columns_to_format.get(sheet_name)
            
            # Select the correct columns to format based on `i`
            if isinstance(columns_inds, dict):
                columns_inds = columns_inds.get(i, [])  # Default to an empty list if `i` is not found

            # Iterate over rows in the range
            for row in range(data_start_row, data_end_row + 1):
                # Check if the row is highlighted (yellow)
                highlight = False
                if "Best?" in df.columns and df.iloc[row - data_start_row]["Best?"] == "Best":
                    highlight = True

                for col in range(len(df.columns)):
                    cell_value = df.iloc[row - data_start_row, col]
            
                    # Combine formatting
                    if col in columns_inds:  # Apply formatting only for the specified columns
                        if highlight:
                            # Create a combined format for yellow and percentage
                            combined_format = workbook.add_format({
                                'bg_color': 'yellow',
                                'num_format': '0.00%'  # Percentage formatting
                            })
                            worksheet.write(row, col, cell_value, combined_format)
                        else:
                            worksheet.write(row, col, cell_value, formats['percentage'])
                    else:
                        if highlight:
                            worksheet.write(row, col, cell_value, formats['yellow'])
                        else:
                            worksheet.write(row, col, cell_value)

                    
    return worksheet, current_row




def final_report(profile, all_parameters_results = None, before_and_after_accuracy_df = None, best_values_dict_params = None, best_CF_Kind_value_df = None, 
                 all_kinds_results_df = None, what_to_tune = None, best_kinds_df = None, file_path = None):

    """
    Generates a final report in Excel summarizing the results of the tuning process for either parameters or CF kinds.

    Args:
        profile (object): Profile object containing initial and current parameter values.
        all_parameters_results (DataFrame, optional): WMAPE and bias results for all parameter tuning runs.
        before_and_after_accuracy_df (DataFrame, optional): Accuracy comparison before and after tuning.
        best_values_dict_params (ataFrame, optional): Best values for each parameter.
        best_CF_Kind_value_df (DataFrame, optional): Best global CF kind values.
        all_kinds_results_df (DataFrame, optional): WMAPE and bias results for all CF kind tuning runs.
        what_to_tune (str, optional): Indicates the target of tuning ("Parameters" or "CFs").
        best_kinds_df (DataFrame, optional): Kind IDs per combination for CF tuning.
        file_path (str): Path to save the Excel report.
    """
    logging.info(f'In final report. what_to_tune = {what_to_tune}')
    
    # Create an Excel writer with xlsxwriter engine
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        # Define formatting options
        formats = {
            'bold': workbook.add_format({'bold': True}),
            'percentage': workbook.add_format({'num_format': '0.00%'}),
            'yellow': workbook.add_format({'bg_color': '#FFFF00'}),
        }

        if what_to_tune == "Parameters":
            # Dataframes and headers for the 'Parameters' sheet
            dataframes = [best_values_dict_params, all_parameters_results]
            headers = ["Best Values per parameter:", "WMAPE and bias for all runs:"]
            sheet_name = "Tuned Parameters Report"

            # Write the data
            worksheet, current_row = write_data_to_excel(writer, sheet_name, dataframes, headers, formats)

            # write the Initial VS Final parameters of the profile
            profile.get_current_profile_parameters_values()
            before_and_after = profile.initial_parameters[['ForecastingParameterName', 'ForecastingParameterValue']].merge(
                profile.current_parameters[['ForecastingParameterName', 'ForecastingParameterValue']],
                on="ForecastingParameterName",
                suffixes=(' Initial', ' Final')
            )
            worksheet.write(current_row, 0, "Profile parameters: Before and after", formats['bold'])
            before_and_after.to_excel(writer, sheet_name = sheet_name, startrow = current_row + 1, header = True, index = False)
            


            # Highlight differences
            highlight_differences(worksheet, before_and_after, current_row + 1, formats)

            # Write the before_and_after_accuracy_df DataFrame starting at G2
            before_and_after_accuracy_df.to_excel(writer, sheet_name = sheet_name, startrow = 1, startcol = 6, header = True, index = False)
            # Apply percentage format to columns H:I (0-based indices 7:8)
            worksheet.set_column(7, 8, None, formats['percentage'])

        elif what_to_tune == "CFs":
            # Dataframes and headers for the 'CFs' sheet
            dataframes = [best_CF_Kind_value_df, all_kinds_results_df]
            headers = ["Best Global Kind ID:", "WMAPE and bias for all runs:"]
            sheet_name = "Tuned CF kinds Report"
            write_data_to_excel(writer, sheet_name, dataframes, headers, formats)

            # Write the before_and_after_accuracy_df DataFrame starting at E2
            before_and_after_accuracy_df.to_excel(writer, sheet_name = sheet_name, startrow = 1, startcol = 4, header = True, index = False)
            # Apply percentage format to columns F:G (0-based indices 5:6)
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column(5, 6, None, formats['percentage'])
            
            # Add a new worksheet for "kind id per comb" and write best_kinds_df
            best_kinds_sheet = "Kind ID per comb"
            best_kinds_df.to_excel(writer, sheet_name=best_kinds_sheet, index=False)

            # Write the before_and_after_accuracy_df DataFrame starting at E2
            before_and_after_accuracy_df.to_excel(writer, sheet_name = sheet_name, startrow = 1, startcol = 4, header = True, index = False)
            # Apply percentage format to columns F:G (0-based indices 5:6)
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column(5, 6, None, formats['percentage'])

