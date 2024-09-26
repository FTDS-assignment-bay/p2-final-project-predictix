'''
============================================================================
This code report project, we will set up an Airflow DAG (Directed Acyclic Graph) to automate the process of extracting data 
from a PostgreSQL database and inserting it into Elasticsearch, a search engine designed for fast querying and data indexing.
The goal is to create a reliable, automated pipeline that fetches customer feedback data stored in PostgreSQL and transfers it to Elasticsearch 
for further analysis and querying.We will walk through the key steps and logic behind the code.
============================================================================
'''
# Import Libraries
import datetime as dt
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
import psycopg2 as db
from elasticsearch import Elasticsearch

# Function to query PostgreSQL
def queryPostgresql():
    '''
    Fetches data from the PostgreSQL database and saves it to a CSV file.

    This function connects to the PostgreSQL database using a connection string,
    executes a SQL query to retrieve all records from the `customer_feedback`, and saves
    the result as a CSV file. Finally, it closes the database connection.
    '''
    # Connect to the PostgreSQL database
    conn_string = "dbname='final_projects' host='postgres' user='airflow' password='airflow'"
    conn = db.connect(conn_string)

    # Select data using SQL query
    df = pd.read_sql("SELECT * FROM customer_feedback", conn)

    # Save data to CSV
    df.to_csv('/opt/airflow/dags/florist_customer_churn_raw_fix_cleaned.csv', index=False)

    # Close the database connection
    conn.close()
    print("-------Data Saved------")

# Function to insert data into Elasticsearch
def insertElasticsearch():
    '''
    Inserts cleaned data into Elasticsearch.

    This function connects to the Elasticsearch service, reads the cleaned data 
    from a CSV file, and inserts each row into an Elasticsearch index called `final_projects`.
    '''
    # Connect to Elasticsearch
    es = Elasticsearch('http://elasticsearch:9200')

    # Read cleaned data from CSV
    df = pd.read_csv('/opt/airflow/dags/florist_customer_churn_raw_fix_cleaned.csv')

    # Insert data into Elasticsearch
    for i, r in df.iterrows():
        doc = r.to_json()
        res = es.index(index="final_projects", body=doc)
        print(res)


with DAG('final_projects',
        start_date= dt.datetime(2024, 9, 12),
        schedule_interval= '30 6 * * *',  # Run every 06.30 AM
         ) as dag:

    # Task to query PostgreSQL
    getData = PythonOperator(task_id='QueryPostgreSQL',
                             python_callable=queryPostgresql)

    # Task to insert data into Elasticsearch
    insertData = PythonOperator(task_id='InsertDataElasticsearch',
                                python_callable=insertElasticsearch)

# Define task flow
getData >> insertData