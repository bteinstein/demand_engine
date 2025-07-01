import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv

import folium

# Load environment variables from .env file
load_dotenv()
import pyodbc
def get_connection():
    # Retrieve connection parameters from environment variables
    server = os.getenv('DB_HOST_OMNIBIZ_REPLICA')
    database = os.getenv('DB_DATABASE_OMNIBIZ_REPLICA')
    username = os.getenv('DB_USER_OMNIBIZ_REPLICA')
    password = os.getenv('DB_PASSWORD_OMNIBIZ_REPLICA')
    driver = os.getenv('DB_DRIVER')

    # Establish the connection
    connection_string = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    conn = pyodbc.connect(connection_string)
    return conn