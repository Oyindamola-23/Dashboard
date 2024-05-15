import json
import logging
from flask import Flask, make_response, request, render_template, jsonify, redirect, session, url_for
from flask import redirect
from flask_session import Session 
import numpy as np
import requests
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta, date
from collections import defaultdict
import warnings
from flask import send_file
import io

app = Flask(__name__)
app.secret_key = '089724'

# Configure session storage
app.config['SESSION_TYPE'] = 'filesystem'  # You can choose other session types

# Initialize Flask-Session
Session(app)

class AuthenticationError(Exception):
    pass

def authenticate(base_url, developer_key, store_id, username, password):
    try:
        endpoint = f"{base_url}/cloudapi/app/access/login?storeid={store_id}"
        payload = {
            "userToken": username,
            "passToken": password
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {developer_key}'
        }
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json(), store_id  # Return authentication response and store_id

    except requests.exceptions.RequestException as e:
        raise AuthenticationError(f"Authentication request failed: {e}")

    except Exception as e:
        raise AuthenticationError(f"Authentication failed: {e}")

def get_data(base_url, access_token, session_id, store_id, start_date, end_date, analysis_type):
    try:
        # Construct endpoint based on analysis type and date range
        if analysis_type == "1":
            endpoint = f"{base_url}/cloudapi/app/{store_id}/reports/sales?startdate={start_date}&enddate={end_date}"
        elif analysis_type == "2":
            endpoint = f"{base_url}/cloudapi/app/{store_id}/reports/salesdetails?startdate={start_date}&enddate={end_date}"
        else:
            raise ValueError("Invalid analysis type. Please select either 1 or 2.")

        # Set headers for API request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}',
            'sessiontoken': f'{session_id}'
        }

        # Make GET request to API endpoint
        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Extract all data from the response
        all_data = data['data']

        return all_data

    except Exception as e:
        raise Exception(f"Failed to fetch data: {e}")

def convert_to_dict(raw_data):
    """
    Convert raw data into a list of dictionaries with specific columns.
    """
    dict_list = []
    for row in raw_data:
        dict_row = {
            "Sales": row[0],
            "Date": row[3],  # Assuming 'Date of Sales' is the fourth column (index 3)
            "Quantity Sold": row[2]  # Assuming 'Quantity Sold' is the third column (index 2)
        }
        dict_list.append(dict_row)
    return dict_list

def calculate_totals(raw_data):
    """
    Calculate total sales and total quantity sold from raw data.
    """
    try:
        # Check if data is not empty
        if raw_data:
            # Initialize variables
            total_sales = sum(item["Sales"] for item in raw_data) / 100  # Divided by 100 to get total sales
            total_quantity_sold = sum(item["Quantity Sold"] for item in raw_data)

            # Print total sales and total quantity sold for debugging
            print("Total Sales:", total_sales)
            print("Total Quantity Sold:", total_quantity_sold)

            return {
                "total_sales": total_sales, 
                "total_quantity_sold": total_quantity_sold
            }
        else:
            raise ValueError("No sales data available.")

    except Exception as e:
        logging.error("Error calculating totals: %s", e)
        raise

def forecast_sales(data, period=None):
    try:
        # Ensure data is a list of dictionaries
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("Data must be a list of dictionaries.")

        # Convert the list of dictionaries into a DataFrame
        sales_df = pd.DataFrame(data)

        # Convert 'Date' to datetime
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])

        # Handle duplicate dates by aggregating sales
        sales_df = sales_df.groupby('Date').sum().reset_index()

        # Ensure the frequency is set in the index
        sales_df.set_index('Date', inplace=True)
        sales_df = sales_df.asfreq('D', fill_value=0)  # Fill missing values with zeros

        # Add 'Year', 'Month', and 'Quarter' columns
        sales_df['Year'] = sales_df.index.year
        sales_df['Month'] = sales_df.index.month
        sales_df['Quarter'] = sales_df.index.quarter

        if period == 'daily':
            # Get the start date and end date for the forecast period
            start_date = sales_df.index[-1] + pd.Timedelta(days=1)
            end_date = start_date + pd.Timedelta(days=4)  # Forecast for 5 days
            future = pd.date_range(start=start_date, end=end_date, freq='D')

        elif period == 'monthly':
            # Get the start month for the forecast period
            start_month = sales_df.index[-1].month
            # Forecast for the remaining months in the year
            future = pd.date_range(start=sales_df.index[-1], periods=12-start_month+1, freq='M')

            # Calculate weekly forecasts for the remaining weeks in the current month
            current_month = sales_df.index[-1].month
            current_year = sales_df.index[-1].year
            current_week = sales_df.index[-1].week
            current_weekday = sales_df.index[-1].weekday()
            remaining_weeks = (pd.Timestamp(year=current_year, month=current_month, day=1) + pd.offsets.MonthEnd(0)).week - current_week

            for i in range(1, remaining_weeks + 1):
                start_week = sales_df.index[-1] + pd.Timedelta(days=7 * i)
                end_week = start_week + pd.Timedelta(days=6)
                future = future.append(pd.date_range(start=start_week, end=end_week, freq='D'))

        elif period == 'quarterly':
            # Get the start quarter for the forecast period
            start_quarter = sales_df.index[-1].quarter
            # Forecast for the remaining quarters in the year
            future = pd.date_range(start=sales_df.index[-1], periods=4-start_quarter+1, freq='Q')

        elif period == 'yearly':
            # Forecast for the next 5 years
            future = pd.date_range(start=sales_df.index[-1], periods=5, freq='Y')

        elif period == 'weekly':  # Adjusted handling for weekly period
            # Get the start week for the forecast period
            start_week = sales_df.index[-1].week
            # Forecast for the next 5 weeks
            future = pd.date_range(start=sales_df.index[-1], periods=5, freq='W')

        else:
            raise ValueError("Invalid period. Please choose 'daily', 'monthly', 'quarterly', 'yearly', or 'weekly'.")

        future_df = pd.DataFrame({'Date': future, 'Sales': 0})  # Initialize sales to 0

        # Feature engineering
        future_df['Year'] = future_df['Date'].dt.year
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Quarter'] = future_df['Date'].dt.quarter

        # Split data into features and target
        X = sales_df[['Year', 'Month', 'Quarter', 'Sales']]  # Include sales column
        y = sales_df['Sales']

        # Divide sales by 100
        X.loc[:, 'Sales'] = X['Sales'] / 100

        # Initialize and train XGBoost regressor
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(X, y)

        # Forecast
        forecast = model.predict(future_df[['Year', 'Month', 'Quarter', 'Sales']])

        # Calculate upper and lower bounds (assuming normal distribution for simplicity)
        std_dev = np.std(y)
        upper_bound = forecast + 1.96 * std_dev  # 95% confidence interval
        lower_bound = forecast - 1.96 * std_dev

        # Create forecast dataframe
        forecast_df = pd.DataFrame({'Date': future, 'Forecasted Sales': forecast, 'Upper Bound': upper_bound,
                                    'Lower Bound': lower_bound})

        return forecast_df
    
    except Exception as e:
        print("An error occurred:", e)
        return None

# Function to convert raw data into a DataFrame
def convert_to_dataframe(raw_data):
    try:
        df = pd.DataFrame(raw_data, columns=["Sales", "Date", "Time", "Quantity Sold"])
        print("DataFrame created from raw data:")
        print(df.head())  # Print the first few rows of the DataFrame
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        print("DataFrame after adding DateTime column:")
        print(df.head())  # Print the first few rows of the DataFrame with DateTime column
        return df[['DateTime', 'Sales', 'Quantity Sold']]
    except Exception as e:
        print(f"An error occurred during data conversion: {e}")
        return None

def forecast_peak_sales_period(sales_df, period):
    try:
        # Resample data based on the selected period
        if period == 'daily':
            period_offset = pd.DateOffset(days=1)
            future_periods = 1  # Forecasting for the next day
        elif period == 'weekly':
            period_offset = pd.DateOffset(weeks=1)
            future_periods = 1  # Forecasting for the next week
        elif period == 'monthly':
            period_offset = pd.DateOffset(months=1)
            future_periods = 12  # Forecasting for the next 12 months
        elif period == 'quarterly':
            period_offset = pd.DateOffset(months=3)
            future_periods = 4  # Forecasting for the next 4 quarters
        else:
            raise ValueError("Invalid period. Please choose 'daily', 'weekly', 'monthly', or 'quarterly'.")

        # Extract hour from the DateTime column
        sales_df['Hour'] = sales_df['DateTime'].dt.hour

        # Group by hour and calculate total sales and quantity sold for each hour
        hourly_sales = sales_df.groupby('Hour').agg({'Sales': 'sum', 'Quantity Sold': 'sum'})

        # Find the hour with maximum sales
        peak_hour = hourly_sales['Sales'].idxmax()

        # Extract date of the last data point
        last_date = sales_df['DateTime'].max()

        # Generate future dates for the forecast
        future_dates = pd.date_range(start=last_date + period_offset, periods=future_periods, freq=period)

        # Calculate the mean sales and quantity sold for the peak hour
        mean_sales = hourly_sales.loc[peak_hour, 'Sales']
        mean_quantity_sold = hourly_sales.loc[peak_hour, 'Quantity Sold']

        # Create a DataFrame for the forecasted sales and quantity
        forecasted_sales = pd.DataFrame({
            'DateTime': future_dates,
            'Peak Hour': [peak_hour] * len(future_dates),
            'Sales': [mean_sales] * len(future_dates),
            'Quantity Sold': [mean_quantity_sold] * len(future_dates)
        })

        # Print peak hour and forecasted sales and quantity data
        print("Peak sales hour for", period, ":", peak_hour)
        print("Peak sales period forecast:")
        print(forecasted_sales)

        return forecasted_sales

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return None

def convert_product_data_to_dict(product_data):
    """
    Convert product data into a list of dictionaries with specific columns.
    """
    dict_list = []
    for row in product_data:
        dict_row = {
            "Product ID": row[0],
            "Customer ID": row[1],  # Include customer ID
            "Product Title": row[2],
            "Date of Sales": row[7],  # Assuming 'Date of Sales' is the eighth column (index 7)
            "Total Sales": row[5],  # Assuming 'Total Sales' is the sixth column (index 5)
            "Quantity Sold": row[6]  # Assuming 'Quantity Sold' is the seventh column (index 6)
        }
        dict_list.append(dict_row)
    return dict_list

def aggregate_product_data(raw_data):
    """
    Aggregate raw sales data by product ID, product title, customer ID, and date of sales.
    """
    aggregated_data = {}
    for entry in raw_data:
        product_id = entry['Product ID']
        customer_id = entry['Customer ID']  # Added customer ID
        product_title = entry['Product Title']
        date_of_sales = entry['Date of Sales']
        total_sales = entry['Total Sales']
        qty_sold = entry['Quantity Sold']
        
        # Parse the date string into a datetime object
        date_obj = datetime.strptime(date_of_sales, '%Y-%m-%d').date()
        
        key = (product_id, customer_id, product_title, date_obj)  # Include customer ID in the key
        if key in aggregated_data:
            # If the key already exists, update the aggregated sales and quantity
            aggregated_data[key]['Total Sales'] += total_sales
            aggregated_data[key]['Quantity Sold'] += qty_sold
        else:
            # If the key doesn't exist, create a new entry
            aggregated_data[key] = {
                'Product ID': product_id,
                'Customer ID': customer_id,  # Include customer ID in the aggregated data
                'Product Title': product_title,
                'Date of Sales': date_obj,
                'Total Sales': total_sales,
                'Quantity Sold': qty_sold
            }
    
    # Convert aggregated data to a list of dictionaries
    product_data = list(aggregated_data.values())
    
    return product_data

def forecast_product(product_data, period=None):
    try:
        # Ensure data is a list of dictionaries
        if not isinstance(product_data, list) or not all(isinstance(item, dict) for item in product_data):
            raise ValueError("Data must be a list of dictionaries.")

        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(product_data)

        # Convert 'Date of Sales' to datetime
        df['Date of Sales'] = pd.to_datetime(df['Date of Sales'])

        # Aggregate sales for each day to handle duplicates
        df = df.groupby(['Product ID', 'Product Title', 'Date of Sales']).agg({'Quantity Sold': 'sum', 'Total Sales': 'sum'}).reset_index()

        # Initialize an empty DataFrame to store forecasts
        forecast_df = pd.DataFrame(columns=['Product ID', 'Product Title', 'Date', 'Forecasted Quantity', 'Forecasted Sales'])

        for product_id, product_title in zip(df['Product ID'].unique(), df['Product Title'].unique()):
            # Subset data for the current product
            product_df = df[df['Product ID'] == product_id].copy()  # Ensure we're working with a copy

            if len(product_df) < 1:
                print(f"Not enough data available for product: {product_title}. Skipping.")
                continue

            # Feature engineering
            product_df['Year'] = product_df['Date of Sales'].dt.year
            product_df['Month'] = product_df['Date of Sales'].dt.month
            product_df['Quarter'] = product_df['Date of Sales'].dt.quarter

            # Ensure each date for each product is unique
            product_df = product_df.groupby(['Date of Sales']).sum().reset_index()

            # Split data into features and target for sales prediction
            X_sales = product_df[['Year', 'Month', 'Quarter', 'Total Sales']]  # Features for sales prediction
            y_sales = product_df['Total Sales']  # Target variable for sales prediction

            # Initialize and train XGBoost regressor for sales prediction
            model_sales = xgb.XGBRegressor(objective='reg:squarederror')
            model_sales.fit(X_sales, y_sales)

            # Split data into features and target for quantity prediction
            X_quantity = product_df[['Year', 'Month', 'Quarter', 'Quantity Sold']]  # Features for quantity prediction
            y_quantity = product_df['Quantity Sold']  # Target variable for quantity prediction

            # Initialize and train XGBoost regressor for quantity prediction
            model_quantity = xgb.XGBRegressor(objective='reg:squarederror')
            model_quantity.fit(X_quantity, y_quantity)

            # Initialize future DataFrame
            future_df = pd.DataFrame()

            # Forecast
            if period == 'daily':
                forecast_steps = 5
                freq = 'D'
            elif period == 'weekly':
                # Find the next Saturday after the last sales date
                next_saturday = product_df['Date of Sales'].max() + pd.offsets.Week(weekday=5)
                # Forecast for the next week starting from the next Saturday
                future = pd.date_range(start=next_saturday, periods=7, freq='D')
                freq = 'W'
                forecast_steps = 1
            elif period == 'monthly':
                remaining_months = 12 - product_df['Date of Sales'].dt.month.max()
                forecast_steps = remaining_months + 1  # Include the current month
                freq = 'M'
            elif period == 'quarterly':
                remaining_quarters = 4 - product_df['Date of Sales'].dt.quarter.max()
                forecast_steps = remaining_quarters + 1  # Include the current quarter
                freq = 'Q'
            elif period == 'yearly':
                forecast_steps = 5 * 12  # 5 years
                freq = 'M'
            else:
                raise ValueError("Invalid period. Please choose 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'.")

            future = pd.date_range(start=product_df['Date of Sales'].iloc[-1], periods=forecast_steps, freq=freq)
            future_df = pd.DataFrame({'Date': future, 'Quantity Sold': 0, 'Total Sales': 0})  # Remove initialization of 'Forecasted Quantity' to 0
            future_df['Year'] = future_df['Date'].dt.year
            future_df['Month'] = future_df['Date'].dt.month
            future_df['Quarter'] = future_df['Date'].dt.quarter
            
            # Predict forecasted sales
            forecast_sales = model_sales.predict(future_df[['Year', 'Month', 'Quarter', 'Total Sales']])

            # Predict forecasted quantities
            forecast_quantity = model_quantity.predict(future_df[['Year', 'Month', 'Quarter', 'Quantity Sold']])
            
            # Round forecasted quantity to the nearest integer
            forecast_quantity = np.round(forecast_quantity)

            # Create forecast DataFrame for the current product
            forecast_product_df = pd.DataFrame({'Product ID': product_id,
                                                'Product Title': product_title,
                                                'Date': future,
                                                'Forecasted Quantity': forecast_quantity,
                                                'Forecasted Sales': forecast_sales})

            # Check for any NA or empty entries in forecast_product_df
            if forecast_product_df.isnull().values.any() or forecast_product_df.empty:
                print(f"Empty or NA entries found in the forecast for product: {product_title}. Skipping.")
                print(f"DataFrame causing the issue: {forecast_product_df}")
                continue

            # Divide forecasted sales by 100
            forecast_product_df['Forecasted Sales'] /= 100

            # Append to the main forecast DataFrame
            forecast_df = pd.concat([forecast_df, forecast_product_df], ignore_index=True)

        return forecast_df

    except Exception as e:
        print(f"An error occurred in the forecast_product function: {e}")
        return None

# Suppress FutureWarning for DataFrame concatenation
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.")

def download_product_forecast_csv(forecast_data):
    try:
        # Create a StringIO object to store CSV data
        csv_buffer = io.StringIO()

        # Write forecast data to the StringIO buffer as CSV
        forecast_data.to_csv(csv_buffer, index=False)

        # Set response headers for CSV download
        headers = {
            "Content-Disposition": "attachment; filename=product_forecast.csv",
            "Content-Type": "text/csv",
        }

        # Create response object with CSV data
        response = make_response(csv_buffer.getvalue())
        response.headers = headers

        return response
    except Exception as e:
        print("An error occurred while generating CSV:", e)
        return None

def calculate_product_trend(product_data, product_name, period):
    print("Calculating product trend...")
    # Filter product data for the selected product name
    product_data_filtered = [record for record in product_data if record['Product Title'] == product_name]
    
    # Convert date strings to datetime.date objects
    for record in product_data_filtered:
        record['Date of Sales'] = datetime.strptime(str(record['Date of Sales']), '%Y-%m-%d').date()
    
    # Create a dictionary to store aggregated sales data by date
    sales_by_date = defaultdict(int)
    
    # Aggregate sales data by date
    for record in product_data_filtered:
        sales_by_date[record['Date of Sales']] += record['Quantity Sold']
    
    # Calculate product trend based on the selected period
    if period == 'daily':
        # For daily trend, return sales data as is
        product_info = {'Product Name': product_name, 'Product Sales': product_data_filtered[0]['Total Sales'], 'Product Quantity Sold': sum(record['Quantity Sold'] for record in product_data_filtered)}
        print("Product trend calculation completed.")
        return sales_by_date, product_info
    
    elif period == 'weekly':
        # Aggregate sales data by week
        sales_by_week = defaultdict(int)
        start_date = min(sales_by_date.keys())
        end_date = max(sales_by_date.keys())
        current_date = start_date
        while current_date <= end_date:
            week_start = current_date - timedelta(days=current_date.weekday())
            week_end = week_start + timedelta(days=6)
            week_sales = sum(sales_by_date[date] for date in sales_by_date if week_start <= date <= week_end)
            sales_by_week[f'{week_start.strftime("%Y-%m-%d")} - {week_end.strftime("%Y-%m-%d")}'] += week_sales
            current_date = week_end + timedelta(days=1)
        product_info = {'Product Name': product_name, 'Product Sales': product_data_filtered[0]['Total Sales'], 'Product Quantity Sold': sum(record['Quantity Sold'] for record in product_data_filtered)}
        print("Product trend calculation completed.")
        return sales_by_week, product_info
    
    elif period == 'monthly':
        # Aggregate sales data by month
        sales_by_month = defaultdict(int)
        for date, sales in sales_by_date.items():
            month = date.strftime('%Y-%m')
            sales_by_month[month] += sales
        product_info = {'Product Name': product_name, 'Product Sales': product_data_filtered[0]['Total Sales'], 'Product Quantity Sold': sum(record['Quantity Sold'] for record in product_data_filtered)}
        print("Product trend calculation completed.")
        return sales_by_month, product_info
    
    elif period == 'quarterly':
        # Aggregate sales data by quarter
        sales_by_quarter = defaultdict(int)
        for date, sales in sales_by_date.items():
            quarter = f"{date.year}-Q{int((date.month - 1) / 3) + 1}"
            sales_by_quarter[quarter] += sales
        product_info = {'Product Name': product_name, 'Product Sales': product_data_filtered[0]['Total Sales'], 'Product Quantity Sold': sum(record['Quantity Sold'] for record in product_data_filtered)}
        print("Product trend calculation completed.")
        return sales_by_quarter, product_info
    
    elif period == 'yearly':
        # Aggregate sales data by year
        sales_by_year = defaultdict(int)
        for date, sales in sales_by_date.items():
            year = str(date.year)  # Convert year to string
            sales_by_year[year] += sales
        product_info = {'Product Name': product_name, 'Product Sales': product_data_filtered[0]['Total Sales'], 'Product Quantity Sold': sum(record['Quantity Sold'] for record in product_data_filtered)}
        print("Product trend calculation completed.")
        return sales_by_year, product_info
    
    else:
        # Invalid period
        print("Invalid period.")
        return None, None

def customer_analysis(product_data):
    """
    Calculate customer analysis, grouping customers into returning and non-returning categories,
    and compute the total quantity sold and total sales for each group.
    """
    returning_customers = {}
    non_returning_customers = {}

    for row in product_data:
        customer_id = row['Customer ID']
        qty_sold = row['Quantity Sold']
        total_sales = row['Total Sales']
        date_of_sales = row['Date of Sales']

        # Check if the customer has made more than one purchase
        if sum(1 for r in product_data if r['Customer ID'] == customer_id and r['Date of Sales'] != date_of_sales) > 0:
            if customer_id in returning_customers:
                # If customer has already made a purchase, update the totals
                returning_customers[customer_id]['Total Quantity Sold'] += qty_sold
                returning_customers[customer_id]['Total Sales'] += total_sales
            else:
                # If it's the first purchase, add the customer to the returning customers dictionary
                returning_customers[customer_id] = {
                    'Total Quantity Sold': qty_sold,
                    'Total Sales': total_sales
                }
        else:
            # Customer has only made one purchase
            if customer_id in non_returning_customers:
                non_returning_customers[customer_id]['Total Quantity Sold'] += qty_sold
                non_returning_customers[customer_id]['Total Sales'] += total_sales
            else:
                non_returning_customers[customer_id] = {
                    'Total Quantity Sold': qty_sold,
                    'Total Sales': total_sales
                }

    # Compute total quantity sold and total sales for returning customers
    total_qty_returning = sum(data['Total Quantity Sold'] for data in returning_customers.values())
    total_sales_returning = sum(data['Total Sales'] for data in returning_customers.values())
    # Count the number of returning customers
    num_returning_customers = len(returning_customers)

    # Compute total quantity sold and total sales for non-returning customers
    total_qty_non_returning = sum(data['Total Quantity Sold'] for data in non_returning_customers.values())
    total_sales_non_returning = sum(data['Total Sales'] for data in non_returning_customers.values())
    # Count the number of non-returning customers
    num_non_returning_customers = len(non_returning_customers)

    # Calculate the percentage of returning and non-returning customers
    total_customers = num_returning_customers + num_non_returning_customers
    returning_customers_percentage = (num_returning_customers / total_customers) * 100
    non_returning_customers_percentage = (num_non_returning_customers / total_customers) * 100

    return {
        'Returning Customers': {
            'Total Quantity Sold': total_qty_returning,
            'Total Sales': total_sales_returning
        },
        'Non-Returning Customers': {
            'Total Quantity Sold': total_qty_non_returning,
            'Total Sales': total_sales_non_returning
        },
        'Number of Returning Customers': num_returning_customers,
        'Number of Non-Returning Customers': num_non_returning_customers,
        'Returning Customers Percentage': returning_customers_percentage,
        'Non-Returning Customers Percentage': non_returning_customers_percentage
    }

def download_customer_analysis_csv(analysis_result):
    try:
        # Create a DataFrame from the analysis result
        df = pd.DataFrame(analysis_result)

        # Create a StringIO object to store CSV data
        csv_buffer = io.StringIO()

        # Write analysis result data to the StringIO buffer as CSV
        df.to_csv(csv_buffer, index=False)

        # Set response headers for CSV download
        headers = {
            "Content-Disposition": "attachment; filename=customer_analysis.csv",
            "Content-Type": "text/csv",
        }

        # Create response object with CSV data
        response = make_response(csv_buffer.getvalue())
        response.headers = headers

        return response
    except Exception as e:
        print("An error occurred while generating CSV:", e)
        return None

# Routes...

@app.route('/')
def index1():
    return render_template('index1.html')

@app.route('/login', methods=['POST'])
def login():
    try:
        username = request.form['username']
        password = request.form['password']
        store_id = request.form['storeID']  # Get store ID from form

        authentication_response, store_id = authenticate(base_url, developer_key, store_id, username, password)
        access_token = authentication_response.get("user", {}).get("defStore", {}).get("accessToken")
        session_id = authentication_response.get("user", {}).get("sessionId")

        return jsonify({"access_token": access_token, "session_id": session_id, "store_id": store_id})

    except AuthenticationError as auth_error:
        return jsonify({"error": f"Authentication Error: {auth_error}"}), 401

    except Exception as e:
        return jsonify({"error": f"Error: {e}"}), 500

@app.route('/analysis', methods=['GET', 'POST'])
def index2():
    if request.method == 'POST':
        analysis_type = request.form['analysis_type']
        if analysis_type == "1":
            return redirect(url_for('result1'))
        elif analysis_type == "2":
            return redirect(url_for('result2'))
        else:
            return redirect(url_for('error'))  # Redirect to error route for invalid analysis type
    else:
        return render_template('index2.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        access_token = request.form['access_token']
        session_id = request.form['session_id']
        store_id = request.form['store_id']
        analysis_type = request.form['analysis_type']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        print("Received analysis request.")
        print("Access token:", access_token)
        print("Session ID:", session_id)
        print("Store ID:", store_id)
        print("Analysis Type:", analysis_type)
        print("Start Date:", start_date)
        print("End Date:", end_date)

        # Fetch data from the API
        raw_data = get_data(base_url, access_token, session_id, store_id, start_date, end_date, analysis_type)
        print("Raw data:", raw_data)  # Print raw data for debugging

        # Convert raw data to list of dictionaries
        processed_data = convert_to_dict(raw_data)

        # Store the processed data in the session
        session['raw_data'] = processed_data

        if analysis_type == '1':
            print("Redirecting to result1 route")
            return redirect(url_for('result1', start_date=start_date, end_date=end_date))

        elif analysis_type == '2':
            # Convert raw data into a list of dictionaries
            data_dict = convert_product_data_to_dict(raw_data)
            
            # Aggregate raw sales data by product title and date
            product_data = aggregate_product_data(data_dict)

            # Integrate 'Product ID' into the data dictionary
            for item in data_dict:
                for product in product_data:
                    if item['Product Title'] == product['Product Title'] and item['Date of Sales'] == product['Date of Sales']:
                        item['Product ID'] = product['Product ID']
                        break

            # Print data_dict
            print("Data Dictionary:")
            print(data_dict)

            # Print product_data
            print("Product Data:")
            print(product_data)

            # Store the product data in the session
            session['product_data'] = product_data
            print("Product data stored in session:", session['product_data'])

            # Convert date strings to datetime.date objects
            for item in data_dict:
                item['Date of Sales'] = datetime.strptime(item['Date of Sales'], '%Y-%m-%d').date()

            # Convert datetime.date objects to strings
            for item in product_data:
                item['Date of Sales'] = item['Date of Sales'].strftime('%Y-%m-%d')

            print("Redirecting to result2 route")
            return redirect(url_for('result2', start_date=start_date, end_date=end_date))

    except Exception as e:
        print("An error occurred during analysis:", e)
        return jsonify({"error": f"Error: {e}"}), 500

@app.route('/result1', methods=['GET', 'POST'])
def result1():
    try:
        print("Session keys:", session.keys())  # Print session keys for debugging
        
        # Get processed data from the session
        processed_data = session.get('raw_data')
        
        if processed_data is None or len(processed_data) == 0:
            return jsonify({"error": "Processed data is None or empty."}), 500
        
        # Calculate totals
        totals = calculate_totals(processed_data)
        
        if request.method == 'POST':
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            selected_period = request.form.get('period')  # Get the selected period for forecasting
            
            # Print the selected data to the terminal for debugging
            print("Selected start date:", start_date)
            print("Selected end date:", end_date)
            print("Selected period:", selected_period)
            
            # Perform calculations based on the raw data
            total_sales = totals.get("total_sales")
            total_quantity_sold = totals.get("total_quantity_sold")
            
            # Check if totals are provided in the session
            if total_sales is None or total_quantity_sold is None:
                return jsonify({"error": "Total sales or total quantity sold is missing in the session."}), 500
            
            print("Total Sales:", total_sales)
            print("Total Quantity Sold:", total_quantity_sold)
            
            forecast_data = forecast_sales(processed_data, period=selected_period)  # Use the selected period for forecasting
            
            if forecast_data is None:
                # If forecast_data is None, return an error message
                return jsonify({"error": "Failed to forecast sales data."}), 500
            
            # Convert forecast data to JSON
            forecast_json = forecast_data.to_dict(orient='records')
            
            # Print the forecasted sales data to the terminal for debugging
            print("Forecasted sales data:", forecast_json)
            
            # Return the forecast data as JSON along with total sum and total quantity
            return jsonify({
                "start_date": start_date,
                "end_date": end_date,
                "period": selected_period,
                "total_sales": total_sales,
                "total_quantity_sold": total_quantity_sold,
                "forecast_data": forecast_json
            })
        
        else:
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            period = request.args.get('period')
            
            # Render the template with total sales and total quantity sold
            return render_template('result1.html', start_date=start_date, end_date=end_date, period=period, total_sales=totals.get("total_sales"), total_quantity_sold=totals.get("total_quantity_sold"))
    
    except Exception as e:
        if 'list' in str(e) and 'has no attribute' in str(e):
            print("An error occurred in the result1 route:", e)
            return jsonify({"error": f"Error: {e}"}), 500
        else:
            print("An error occurred in the result1 route:", e)
            return jsonify({"error": f"Error: {e}"}), 500

@app.route('/result1b', methods=['GET', 'POST'])
def result1b():
    if request.method == 'POST':
        try:
            # Get the processed data and sales DataFrame from the session
            processed_data = session.get('raw_data')
            sales_df = session.get('sales_df')  # Retrieve sales DataFrame from session

            print("Raw data retrieved from session:", processed_data)
            print("Sales DataFrame retrieved from session:", sales_df)

            if processed_data is None or len(processed_data) == 0:
                return jsonify({"error": "Processed data is None or empty."}), 500

            # Get the selected period from the form data
            selected_period = request.form.get('period')
            
            # Print the selected data to the terminal for debugging
            print("Selected period:", selected_period)

            # Define the forecast message based on the selected period
            forecast_message = "Peak sales period forecast - {}".format(selected_period.capitalize())
            print(forecast_message)

            # Check if the period is provided by the user
            if selected_period is None:
                return render_template('result2.html', error="Please select a period for forecasting.")

            # Perform peak sales period forecast calculation based on the sales data and selected period
            peak_sales_period = forecast_peak_sales_period(sales_df, period=selected_period)
            print("Peak sales period forecast:", peak_sales_period)

            if peak_sales_period is None:
                # If peak_sales_period is None, return an error message
                return jsonify({"error": "Failed to forecast peak sales period."}), 500

            # Convert DataFrame to list of dictionaries for proper serialization
            peak_sales_period_json = peak_sales_period.to_dict(orient='records')

            # Return the forecasted peak sales period
            return jsonify({
                "forecast_message": forecast_message,
                "peak_sales_period": peak_sales_period_json
            })

        except Exception as e:
            print("An error occurred in the result1b route:", e)
            return jsonify({"error": f"Error: {e}"}), 500
    else:
        # If it's a GET request, render the result1b.html template
        return render_template('result1b.html')

@app.route('/result2', methods=['GET', 'POST'])
def result2():
    if request.method == 'POST':
        try:
            # Get product data from the session
            product_data = session.get('product_data')
            
            if product_data is None or len(product_data) == 0:
                return jsonify({"error": "Product data is None or empty."}), 500
            
            # Get the selected period from the form data
            selected_period = request.form.get('period')
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            
            # Print the selected data to the terminal for debugging
            print("Selected period:", selected_period)
            print("Selected start date:", start_date)
            print("Selected end date:", end_date)
            
            # Define the forecast message based on the selected period
            forecast_message = "Product forecast - {}".format(selected_period.capitalize())
            print(forecast_message)
            
            # Check if the period is provided by the user
            if selected_period is None:
                return render_template('result2.html', error="Please select a period for forecasting.")
            
            # Perform product forecast calculation based on the product data and selected period
            forecast_data = forecast_product(product_data, period=selected_period)
            
            if forecast_data is None:
                # If forecast_data is None, return an error message
                return jsonify({"error": "Failed to forecast product data."}), 500

            # Aggregate the forecasted data by product title
            forecast_data_agg = forecast_data.groupby('Product Title').agg({'Forecasted Quantity': 'sum', 'Forecasted Sales': 'sum'}).reset_index()
            
            # Print the aggregated forecasted data
            print("Forecasted data:", forecast_data_agg.to_dict(orient='records'))
            
            # Convert forecast data to JSON
            forecast_json = forecast_data.to_dict(orient='records')
            
            # Sort products by forecasted sales in descending order
            sorted_forecast_data = sorted(forecast_data_agg.to_dict(orient='records'), key=lambda x: x['Forecasted Sales'], reverse=True)
            
            # Get top 10 products
            top_10_products = []
            for product in sorted_forecast_data[:10]:
                # Find product ID for the current product title
                product_id = None
                for item in product_data:
                    if item['Product Title'] == product['Product Title']:
                        product_id = item['Product ID']
                        break
                # Append product data to top_10_products list
                if product_id is not None:
                    top_10_products.append({
                        'Product ID': product_id,
                        'Product Title': product['Product Title'],
                        'Forecasted Sales': product['Forecasted Sales'],
                        'Forecasted Quantity': product['Forecasted Quantity']
                    })

            # Print top 10 products, prices, and quantities
            print("Top 10 Products:")
            for product in top_10_products:
                print(f"Product ID: {product['Product ID']}, Product Title: {product['Product Title']}, Forecasted Sales: {product['Forecasted Sales']}, Forecasted Quantity: {product['Forecasted Quantity']}")
                        
            # Return the forecast data as JSON along with start date, end date, period, forecast message, and top 10 products
            return jsonify({
                "start_date": start_date,
                "end_date": end_date,
                "period": selected_period,
                "forecast_message": forecast_message,
                "forecast_data": forecast_json,
                "top_10_products": top_10_products
            })
        
        except Exception as e:
            print("An error occurred in the result2 route:", e)
            return jsonify({"error": f"Error: {e}"}), 500
    else:
        # If it's a GET request, render the result2.html template
        return render_template('result2.html')
    
@app.route('/download_product_forecast', methods=['POST'])
def download_product_forecast():
    try:
        forecast_data = request.json.get('forecast_data')

        print("Received forecast data:", forecast_data)  # Debugging

        if forecast_data is None:
            print("Forecast data is missing in the request.")
            return jsonify({"error": "Forecast data is missing in the request."}), 400

        forecast_df = pd.DataFrame(forecast_data)

        print("Forecast DataFrame:", forecast_df)  # Debugging

        # Call the download_product_forecast_csv function to generate CSV file
        csv_response = download_product_forecast_csv(forecast_df)

        if csv_response is None:
            print("Failed to generate CSV file.")
            return jsonify({"error": "Failed to generate CSV file."}), 500

        return csv_response
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/result2b', methods=['GET', 'POST'])
def result2b():
    if request.method == 'POST':
        try:
            # Get product data from the session
            product_data = session.get('product_data')
            print("Product data retrieved from session:", product_data)
            
            if product_data is None or len(product_data) == 0:
                return jsonify({"error": "Product data is None or empty."}), 500
            
            # Get the selected period and product name from the form data
            selected_period = request.json.get('period')
            product_name = request.json.get('product_name')
            
            print("Selected period:", selected_period)
            print("Product name:", product_name)
            
            if not selected_period or not product_name:
                return jsonify({"error": "Missing period or product name in the request."}), 400
            
            # Perform product trend calculation based on the selected product name and period
            print("Calculating product trend...")
            product_trend_data, product_info = calculate_product_trend(product_data, product_name, selected_period)
            print("Product trend calculation completed.")
            
            if product_trend_data is None:
                return jsonify({"error": "Failed to calculate product trend."}), 500

            # Convert date objects to string representations
            product_trend_data_str = {str(key): value for key, value in product_trend_data.items()}

            # Print the result of product trend and product info
            print("Product trend data:", product_trend_data_str)
            print("Product info:", product_info)

            # Return the calculated product trend data, product info, and selected period
            return jsonify({
                "selected_period": selected_period,
                "product_info": product_info,
                "product_trend_data": product_trend_data_str
            })

        except Exception as e:
            print("An error occurred in the result2b route:", e)
            return jsonify({"error": f"Error: {e}"}), 500
        
    elif request.method == 'GET':
        # Handle GET request
        print("GET request received.")
        # Get product data from the session
        product_data = session.get('product_data')
        print("Product data retrieved from session:", product_data)
        
        if product_data is None or len(product_data) == 0:
            return jsonify({"error": "Product data is None or empty."}), 500
        
        # Extract product titles from product data
        product_titles = set(record['Product Title'] for record in product_data)
        print("Unique Product titles:", product_titles)
        
        # Render the template with product titles
        return render_template('result2b.html', product_titles=product_titles)
    
    else:
        return jsonify({"error": "Method not allowed."}), 405

@app.route('/result2c', methods=['GET', 'POST'])
def result2c():
    if request.method == 'POST':
        try:
            # Get product data from the session
            product_data = session.get('product_data')
            print("Product data retrieved from session:", product_data)
            
            # Get the start date and end date from the form data
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')

            if product_data is None or len(product_data) == 0:
                return jsonify({"error": "Product data is None or empty."}), 500
            
            # Perform customer analysis
            print("Performing customer analysis...")
            analysis_result = customer_analysis(product_data)
            print("Customer analysis completed.")
            print("Customer Analysis Result:", analysis_result)
            
            # Jsonify the analysis result
            return jsonify(analysis_result)

        except Exception as e:
            print("An error occurred in the result2c route:", e)
            return jsonify({"error": f"Error: {e}"}), 500
        
    elif request.method == 'GET':
        # Handle GET request
        print("GET request received.")
        # Get product data from the session
        product_data = session.get('product_data')
        print("Product data retrieved from session:", product_data)
        
        if product_data is None or len(product_data) == 0:
            return jsonify({"error": "Product data is None or empty."}), 500
        
        # Get the start date and end date from the query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # Check if both start_date and end_date are not None
        if start_date is not None and end_date is not None:
            # Render the template without redirection
            return render_template('result2c.html', start_date=start_date, end_date=end_date)
        else:
            # Redirect to the 'result2' route with start_date and end_date parameters
            print("Redirecting to result2 route")
            return redirect(url_for('result2c'))
    
    else:
        return jsonify({"error": "Method not allowed."}), 405
    
@app.route('/download_customer_analysis', methods=['POST'])
def download_customer_analysis():
    try:
        analysis_result = request.json.get('analysis_result')

        print("Received analysis result:", analysis_result)  # Debugging

        if analysis_result is None:
            print("Analysis result is missing in the request.")
            return jsonify({"error": "Analysis result is missing in the request."}), 400

        # Call the download_customer_analysis_csv function to generate CSV file
        csv_response = download_customer_analysis_csv(analysis_result)

        if csv_response is None:
            print("Failed to generate CSV file.")
            return jsonify({"error": "Failed to generate CSV file."}), 500

        return csv_response
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500
    
@app.route('/error')
def error():
    return render_template('error.html')

if __name__ == '__main__':
    developer_key = "6970001734eb46609694b4b1cbd6a426"
    base_url = "https://gateway.storeharmony.com"
    app.run(debug=True)