/*
====================
This Syntax is using to create database and create table using PostgreSQL, then export the dataset table using PSQLTools
====================
*/


-- Create Database
CREATE DATABASE final_projects

-- Create New Table
CREATE TABLE customer_feedback (
    "customer_id" VARCHAR PRIMARY KEY,
	"churn" VARCHAR (10),
	"tenure" INT,
	"monthly_charges" FLOAT,
	"total_charges" FLOAT,
	"contract" VARCHAR (30),
	"payment_method" VARCHAR (30),
	"feedback" TEXT,
	"sentiment" VARCHAR (20),
	"topic" VARCHAR (50)
);

-- Note : Copy this code to your PSQLTools, and change the directory '/Users/...' because i run this using Docker Container
\copy customer_feedback FROM '/Users/danisarahadians/Hacktiv8/Final Projects/p2-final-project-predictix/florist_customer_churn_raw_fix_cleaned.csv' DELIMITER ',' CSV HEADER;


