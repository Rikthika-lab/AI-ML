import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import mysql.connector
import re

# ------------------ MySQL Connection Setup ------------------
def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",     # replace with your host
        user="root",          # replace with your username
        password="",  # replace with your password
        database=""    # replace with your database name
    )
def create_table_if_not_exists(table_name, df, conn):
    cursor = conn.cursor()
    
    # Generate SQL columns dynamically from dataframe
    columns = []
    for col in df.columns:
        # Use TEXT for simplicity; you can improve by detecting numeric/date types
        columns.append(f"`{col}` TEXT")
    columns_sql = ", ".join(columns)
    
    # Create table if it doesn't exist
    sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({columns_sql});"
    cursor.execute(sql)
    conn.commit()

# ------------------ Add missing columns if table exists ------------------
def add_missing_columns(table_name, df, conn):
    """
    Adds missing columns from dataframe to an existing MySQL table
    """
    cursor = conn.cursor()
    
    # Get existing columns from MySQL table
    cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
    existing_columns = [col[0] for col in cursor.fetchall()]
    
    # Add any missing columns
    for col in df.columns:
        if col not in existing_columns:
            sql = f"ALTER TABLE `{table_name}` ADD COLUMN `{col}` TEXT"
            cursor.execute(sql)
            conn.commit()


def insert_dataframe_to_mysql(table_name, df, conn):
    cursor = conn.cursor()
    for _, row in df.iterrows():
        placeholders = ", ".join(["%s"] * len(row))
        columns = ", ".join([f"`{col}`" for col in df.columns])
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, tuple(row))
    conn.commit()

# ------------------ Automatic Structuring Function ------------------
def structure_raw_data(df):
    structured_df = df.copy()

    for col in structured_df.columns:
        # Try date conversion ONLY if column looks like a date
        if structured_df[col].astype(str).str.match(r"\d{1,2}-\d{1,2}-\d{4}").all():
            structured_df[col] = pd.to_datetime(structured_df[col], dayfirst=True)

        else:
            # Convert to numeric where possible
            structured_df[col] = pd.to_numeric(structured_df[col], errors="ignore")

    return structured_df


# ------------------ Streamlit UI ------------------
st.title("Automatic Data Structuring & MySQL Loader")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Removes the automatic pandas index column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Clean column names: removes spaces, colons, etc., and replace with underscores
    df.columns = [col.strip().replace(" ", "_").replace(":", "_").replace("-", "_") for col in df.columns]

    # to convert all column names to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]


    st.subheader("Raw Data")
    st.dataframe(df.head())


    # Connect to MySQL
    conn = get_mysql_connection()

    # Ask user for table name
    table_name = st.text_input("Enter MySQL Table Name:", "structured_data")

    if st.button("Structure & Save to MySQL"):
        structured_df = structure_raw_data(df)
        
        # Clean column names
        if "Unnamed: 0" in structured_df.columns:
            structured_df = structured_df.drop(columns=["Unnamed: 0"])
        structured_df.columns = [col.strip().replace(" ", "_").replace(":", "_").replace("-", "_").lower() for col in structured_df.columns]
        
        st.subheader("Structured Data")
        st.dataframe(structured_df.head())
        
        # Create table automatically based on dataframe
        create_table_if_not_exists(table_name, structured_df, conn)
        #  Add missing columns dynamically
        add_missing_columns(table_name, structured_df, conn)
        # Insert dataframe
        insert_dataframe_to_mysql(table_name, structured_df, conn)

        st.success(f"Data successfully saved to MySQL table '{table_name}'")
        conn.close()

