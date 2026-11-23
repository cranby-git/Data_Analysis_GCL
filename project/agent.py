from google.cloud import bigquery 
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode
from pathlib import Path
import csv

PROJECT_ID = "project-842dbf51-43bc-4833-997"
DATASET_ID = "agent_dest_dataset"
AGENT_DIR = Path(__file__).resolve().parent


def detect_delimiter(file_path: Path, default: str = ";") -> str:
  """Detects the CSV delimiter from the first few kilobytes using csv.Sniffer."""
  with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    sample = f.read(8192)

  if not sample.strip():
    return default

  try:
    dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "|"])
    return dialect.delimiter
  except csv.Error:
    return default


def upload_csv_to_bigquery(csv_file_path: str, table_name: str) -> str:
    """Uploads a local CSV file into a BigQuery table with schema auto-detection.

    Args:
        csv_file_path: Path to the local CSV file.
        table_name: Target table name in BigQuery

    Returns:
        Confirmation message with the full BigQuery table ID and row count.
    """
    client = bigquery.Client(project=PROJECT_ID) # Needed for authentication to use BQ
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    delimiter = detect_delimiter(AGENT_DIR / csv_file_path)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        field_delimiter = delimiter, 
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE, #truncate to overwrite if it already exists
    )

    # Load table into BQ after collecting configurtion and collect LoadJob to wait on job
    with open(AGENT_DIR / csv_file_path, "rb") as source_file:
        job = client.load_table_from_file(
            source_file, table_id, job_config=job_config
        )

    job.result()  # Wait for the load job to complete

    table = client.get_table(table_id)
    return f"Successfully loaded {table.num_rows} rows into `{table_id}`."


# Configure BigQueryToolset with write access so it can run CREATE MODEL
bigquery_toolset = BigQueryToolset(
    bigquery_tool_config=BigQueryToolConfig(write_mode=WriteMode.ALLOWED) # Write is off by default
)

# Define the data agent
data_scientist_agent = Agent(
    name="data_scientist_agent",
    model="gemini-3.8-flash",
    description="Uploads CSV files to BigQuery and trains/evaluates BigQuery ML models. Reports metrics to user.",
    instruction=f"""
    You are a data science and BigQuery ML assistant working in GCP project `{PROJECT_ID}` and dataset `{DATASET_ID}`.

    When given a CSV file and a prediction task:
    1. Call `upload_csv_to_bigquery` to load the CSV into `{PROJECT_ID}.{DATASET_ID}`. If no table name is provided, default to the name of CSV file.
    2. Use `BigQueryToolset.get_table_info` to inspect the column names and data types of the created table.
    3. Choose an appropriate BigQuery ML model_type (`LINEAR_REG` for continuous targets, `LOGISTIC_REG` for classification).
    4. Execute a `CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET_ID}.<table_name>_model`` SQL query via `execute_sql`, setting `input_label_cols=['<target_column>']`.
    5. Run `SELECT * FROM ML.EVALUATE(MODEL `{PROJECT_ID}.{DATASET_ID}.<table_name>_model`)` via `BigQueryToolset.execute_sql` and report the training metrics clearly to the user.
    6. Report graphs of Loss, Duration, and Learn rate of the training metrics to the user.
    7. Summarize the contents of the dataset to the user quantatively.
    """,
    tools=[upload_csv_to_bigquery, bigquery_toolset],
    # sub_agents=[report_agent]
    output_key="data_summary"
)

# Report agent
report_agent = Agent(
    name="report_agent",
    model="gemini-3.8-flash",
    description="Takes data from data_scientist_agent and produces an execuive summary of the data.",
    instruction=f"""
    You are a data analyst assistant working in GCP project `{PROJECT_ID}` with dataset `{DATASET_ID}`.
    Use the findings of the data_scientist_agent to produce an clear report for the user. Include the following:
    1. An executive summary 
    2. The Methodology 
    3. The Findings
    4. The limitations
    5. Recommendations

    Findings:
    {{data_summary?}}

    """
)

root_agent = SequentialAgent(
    name="root_agent",
    description="Root sequential agent that orchestrates the data_scientist_agent and report_agent.",
    sub_agents=[
        data_scientist_agent,
        report_agent
    ],

)

