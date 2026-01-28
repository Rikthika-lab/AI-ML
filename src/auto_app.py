import streamlit as st
import pandas as pd
import mysql.connector
import os
import re

# ------------------ MySQL Connection ------------------
def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="your username",
        password="your password",
        database="data_db"
    )

# ------------------ Create Table ------------------
def create_table_if_not_exists(table_name, df, conn):
    cursor = conn.cursor()
    columns_sql = ", ".join([f"`{col}` TEXT" for col in df.columns])
    sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` (id INT AUTO_INCREMENT PRIMARY KEY, {columns_sql})"
    cursor.execute(sql)
    conn.commit()

# ------------------ Add Missing Columns ------------------
def add_missing_columns(table_name, df, conn):
    cursor = conn.cursor()
    cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
    existing_cols = [c[0] for c in cursor.fetchall()]

    for col in df.columns:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE `{table_name}` ADD COLUMN `{col}` TEXT")
            conn.commit()

# ------------------ Insert Data ------------------
def insert_dataframe_to_mysql(table_name, df, conn):
    cursor = conn.cursor()
    cols = ", ".join([f"`{c}`" for c in df.columns])
    placeholders = ", ".join(["%s"] * len(df.columns))

    sql = f"INSERT INTO `{table_name}` ({cols}) VALUES ({placeholders})"

    for _, row in df.iterrows():
        cursor.execute(sql, tuple(row.astype(str)))
    conn.commit()

# ------------------ Intelligent Structuring ------------------
def structure_raw_data(df):
    structured_df = df.copy()

    for col in structured_df.columns:
        # Detect date columns (dd-mm-yyyy or yyyy-mm-dd)
        if structured_df[col].astype(str).str.match(
            r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$|^\d{4}[-/]\d{1,2}[-/]\d{1,2}$"
        ).all():
            structured_df[col] = pd.to_datetime(structured_df[col], dayfirst=True)
        else:
            structured_df[col] = pd.to_numeric(structured_df[col], errors="ignore")

    return structured_df


# ------------------ Missing Value Handling ------------------(only after structuring)
def handle_missing_values(df, strategy="mean"):
    """
    Handles missing values in numeric columns.
    strategy: 'mean' or 'median'
    """
    cleaned_df = df.copy()

    for col in cleaned_df.columns:
        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
            if strategy == "mean":
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif strategy == "median":
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())

    return cleaned_df


# ------------------ Streamlit UI ------------------
st.title("Automatic Data Structuring & MySQL Loader")

uploaded_file = st.file_uploader(
    "Upload your CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Remove unwanted index column
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(":", "_")
        .str.replace("-", "_")
    )

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # ------------------ HYBRID TABLE NAMING ------------------
    file_base = os.path.splitext(uploaded_file.name)[0]
    suggested_table_name = re.sub(r"\W+", "_", file_base.lower())

    table_name = st.text_input(
        "MySQL Table Name (auto-generated, editable)",
        suggested_table_name
    )

    if st.button("Structure & Save to MySQL"):
        conn = get_mysql_connection()
        #1: Structure raw data
        structured_df = structure_raw_data(df)
        st.subheader("Structured Data (Before Missing Handling)")
        st.dataframe(structured_df.head())

        # 2: Count NaN BEFORE handling
        st.subheader("Missing Values BEFORE Handling")
        st.write(structured_df.isna().sum())
        #missing values
        structured_df = handle_missing_values(structured_df,strategy = "mean")

        # 3: Count NaN AFTER handling
        st.subheader("Missing Values AFTER Handling")
        st.write(structured_df.isna().sum())

        st.subheader("Structured Data (After Missing Handling)")
        st.dataframe(structured_df.head())


        st.subheader("Structured Data Preview")
        st.dataframe(structured_df.head())

        create_table_if_not_exists(table_name, structured_df, conn)
        add_missing_columns(table_name, structured_df, conn)
        insert_dataframe_to_mysql(table_name, structured_df, conn)

        conn.close()
        st.success(f"Data successfully stored in table `{table_name}`")
