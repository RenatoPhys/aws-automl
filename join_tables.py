
'''aasdfasd'''
from dateutil.relativedelta import relativedelta

def calculate_date_offset(self, date_str: str, delay: str) -> str:
    if not delay or delay == 'm':
        return date_str
    
    current_date = datetime.strptime(date_str, '%Y%m')
    
    if 'm-' in delay:
        months_back = int(delay.split('-')[1])
        new_date = current_date - relativedelta(months=months_back)
        return new_date.strftime('%Y%m')
        
    # You could add support for other patterns like 'y-1' here
    return date_str


"""
Automated ML Pipeline for Feature Table Left Joins
Using AWS Athena and Polars in SageMaker Studio
Creates one output table per feature table
"""

import polars as pl
import pandas as pd
import boto3
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureTableJoiner:
    """Automated pipeline for joining feature tables with main table"""
    
    def __init__(self, 
                 database: str,
                 main_table: str,
                 main_doc_col: str,
                 main_date_col: str,
                 s3_output_location: str,
                 region: str = 'us-east-1'):
        """
        Initialize the Feature Table Joiner
        
        Args:
            database: Athena database name
            main_table: Name of the main/public table
            main_doc_col: Document ID column in main table
            main_date_col: Date column (YYYYMM) in main table
            s3_output_location: S3 location for Athena query results
            region: AWS region
        """
        self.database = database
        self.main_table = main_table
        self.main_doc_col = main_doc_col
        self.main_date_col = main_date_col
        self.s3_output_location = s3_output_location
        
        # Initialize Athena client
        #self.athena = boto3.client('athena', region_name=region)
        
        # Store created tables
        self.created_tables = {}
    
    def read_metadata(self, filepath: str) -> pl.DataFrame:
        """Read metadata file containing feature table information"""
        if filepath.endswith('.xlsx'):
            df = pl.read_excel(filepath)
        elif filepath.endswith('.csv'):
            df = pl.read_csv(filepath)
        else:
            df = pl.read_parquet(filepath)
        
        logger.info(f"Loaded {len(df)} feature tables from metadata")
        return df
    
    def calculate_date_offset(self, date_str: str, delay: str) -> str:
        """
        Calculate date with offset based on delay pattern (e.g., 'm-1', 'm-2')
        
        Args:
            date_str: Date string in YYYYMM format
            delay: Delay pattern like 'm-1', 'm-2'
        """
        if not delay or delay == 'm':
            return date_str
        
        # Parse YYYYMM
        year = int(date_str[:4])
        month = int(date_str[4:6])
        date = datetime(year, month, 1)
        
        # Extract offset number
        if 'm-' in delay:
            months_back = int(delay.split('-')[1])
            # Calculate new date
            for _ in range(months_back):
                date = date.replace(day=1) - timedelta(days=1)
                date = date.replace(day=1)
        
        return date.strftime('%Y%m')
    
    def execute_athena_query(self, query: str) -> str:
        """Execute Athena query and return query execution ID"""
        response = self.athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': self.database},
            ResultConfiguration={'OutputLocation': self.s3_output_location}
        )
        return response['QueryExecutionId']
    
    def wait_for_query(self, query_id: str, max_wait: int = 300) -> bool:
        """Wait for Athena query to complete"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = self.athena.get_query_execution(QueryExecutionId=query_id)
            status = response['QueryExecution']['Status']['State']
            
            if status in ['SUCCEEDED']:
                return True
            elif status in ['FAILED', 'CANCELLED']:
                error = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                logger.error(f"Query failed: {error}")
                return False
            
            time.sleep(2)
        
        logger.error("Query timeout")
        return False
    
    def get_feature_table_schema(self, table_name: str) -> List[Tuple[str, str]]:
        """Get schema of a feature table"""
        query = f"DESCRIBE {table_name}"
        query_id = self.execute_athena_query(query)
        
        if self.wait_for_query(query_id):
            result = self.athena.get_query_results(QueryExecutionId=query_id)
            rows = result['ResultSet']['Rows'][1:]  # Skip header
            
            schema = []
            for row in rows:
                col_name = row['Data'][0].get('VarCharValue', '')
                col_type = row['Data'][1].get('VarCharValue', '')
                if col_name and not col_name.startswith('#'):  # Skip partition info
                    schema.append((col_name, col_type))
            return schema
        return []
    
    def create_empty_joined_table(self, 
                                 feature_table: str,
                                 doc_col: str,
                                 date_col: str,
                                 output_table: str = None) -> str:
        """
        Create an empty table with schema for joined data
        
        Args:
            feature_table: Name of the feature table
            doc_col: Document column name in feature table
            date_col: Date column name in feature table
            output_table: Optional output table name
        """
        if not output_table:
            output_table = f"{feature_table}_joined_{int(time.time())}"
        
        # Get schemas
        main_schema = self.get_feature_table_schema(self.main_table)
        feature_schema = self.get_feature_table_schema(feature_table)
        
        # Build column definitions
        columns = []
        
        # Add main table columns
        for col_name, col_type in main_schema:
            if col_name != self.main_date_col:  # Date column will be partition
                columns.append(f"{col_name} {col_type}")
        
        # Add feature table columns (excluding join keys)
        for col_name, col_type in feature_schema:
            if col_name not in [doc_col, date_col]:
                # Prefix with table name to avoid conflicts
                columns.append(f"{feature_table}_{col_name} {col_type}")
        
        # Create table query
        create_query = f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS {output_table} (
            {', '.join(columns)}
        )
        PARTITIONED BY ({self.main_date_col} string)
        STORED AS PARQUET
        LOCATION 's3://{self.s3_output_location.replace('s3://', '')}/{output_table}/'
        """
        
        logger.info(f"Creating empty table: {output_table}")
        query_id = self.execute_athena_query(create_query)
        
        if self.wait_for_query(query_id):
            logger.info(f"Successfully created empty table: {output_table}")
            self.created_tables[feature_table] = output_table
            return output_table
        else:
            raise Exception(f"Failed to create table {output_table}")
    
    def insert_joined_data(self,
                          feature_table: str,
                          doc_col: str,
                          doc_type: str,
                          date_col: str,
                          date_type: str,
                          delay: str,
                          output_table: str,
                          target_date: str):
        """
        Insert joined data for a specific date partition
        
        Args:
            feature_table: Name of the feature table
            doc_col: Document column in feature table
            doc_type: Data type of document column
            date_col: Date column in feature table
            date_type: Data type of date column
            delay: Date delay pattern (m-1, m-2, etc.)
            output_table: Target output table
            target_date: Date to process (YYYYMM)
        """
        # Calculate adjusted date
        adjusted_date = self.calculate_date_offset(target_date, delay)
        
        # Build column selections
        main_cols = []
        feature_cols = []
        
        # Get schemas to build proper select
        main_schema = self.get_feature_table_schema(self.main_table)
        feature_schema = self.get_feature_table_schema(feature_table)
        
        # Main table columns
        for col_name, _ in main_schema:
            if col_name != self.main_date_col:
                main_cols.append(f"m.{col_name}")
        
        # Feature table columns
        for col_name, _ in feature_schema:
            if col_name not in [doc_col, date_col]:
                feature_cols.append(f"COALESCE(f.{col_name}, NULL) AS {feature_table}_{col_name}")
        
        # Build join condition with type casting
        doc_cast = f"CAST(f.{doc_col} AS VARCHAR)" if doc_type != 'string' else f"f.{doc_col}"
        main_doc_cast = f"CAST(m.{self.main_doc_col} AS VARCHAR)" if doc_type != 'string' else f"m.{self.main_doc_col}"
        date_cast = f"CAST(f.{date_col} AS VARCHAR)" if date_type != 'string' else f"f.{date_col}"
        
        # Build insert query
        insert_query = f"""
        INSERT INTO {output_table}
        SELECT 
            {', '.join(main_cols)},
            {', '.join(feature_cols)},
            '{target_date}' as {self.main_date_col}
        FROM {self.main_table} m
        LEFT JOIN {feature_table} f
            ON {doc_cast} = {main_doc_cast}
            AND {date_cast} = '{adjusted_date}'
        WHERE m.{self.main_date_col} = '{target_date}'
        """
        
        logger.info(f"Inserting data for {feature_table} - date: {target_date}")
        query_id = self.execute_athena_query(insert_query)
        
        if self.wait_for_query(query_id):
            logger.info(f"Successfully inserted data for {target_date}")
            # Add partition
            self.add_partition(output_table, target_date)
        else:
            raise Exception(f"Failed to insert data for {target_date}")
    
    def add_partition(self, table_name: str, partition_date: str):
        """Add partition to the table"""
        alter_query = f"""
        ALTER TABLE {table_name} 
        ADD IF NOT EXISTS PARTITION ({self.main_date_col} = '{partition_date}')
        """
        
        query_id = self.execute_athena_query(alter_query)
        if self.wait_for_query(query_id):
            logger.info(f"Added partition {partition_date} to {table_name}")
    
    def process_pipeline(self, 
                        metadata_file: str,
                        date_list: List[str],
                        table_prefix: str = "ft") -> Dict[str, str]:
        """
        Main pipeline: Create tables and process data
        
        Args:
            metadata_file: Path to metadata file
            date_list: List of dates in YYYYMM format to process
            table_prefix: Prefix for output table names
        
        Returns:
            Dictionary mapping feature tables to created output tables
        """
        # Read metadata
        feature_tables = self.read_metadata(metadata_file)
        
        # Step 1: Create empty tables for each feature table
        logger.info("Step 1: Creating empty tables...")
        for row in feature_tables.iter_rows(named=True):
            table_name = row['table_name']
            doc_col = row['doc_column_name']
            date_col = row['date_column_name']
            
            output_table = f"{table_prefix}_{table_name}_joined"
            self.create_empty_joined_table(
                feature_table=table_name,
                doc_col=doc_col,
                date_col=date_col,
                output_table=output_table
            )
        
        # Step 2: Insert data for each date
        logger.info("Step 2: Inserting data for each date...")
        for date in date_list:
            logger.info(f"Processing date: {date}")
            
            for row in feature_tables.iter_rows(named=True):
                table_name = row['table_name']
                output_table = self.created_tables[table_name]
                
                self.insert_joined_data(
                    feature_table=table_name,
                    doc_col=row['doc_column_name'],
                    doc_type=row['doc_column_type'],
                    date_col=row['date_column_name'],
                    date_type=row['date_column_type'],
                    delay=row['date_delay'],
                    output_table=output_table,
                    target_date=date
                )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Created tables: {self.created_tables}")
        return self.created_tables
    
    def read_table_with_polars(self, table_name: str, 
                               partition_filter: str = None) -> pl.DataFrame:
        """
        Read a specific table using Polars
        
        Args:
            table_name: Name of the table to read
            partition_filter: Optional WHERE clause for partitions
        """
        query = f"SELECT * FROM {table_name}"
        if partition_filter:
            query += f" WHERE {partition_filter}"
        
        query_id = self.execute_athena_query(query)
        
        if self.wait_for_query(query_id):
            result = self.athena.get_query_results(QueryExecutionId=query_id)
            rows = result['ResultSet']['Rows']
            
            if len(rows) > 1:
                headers = [col['VarCharValue'] for col in rows[0]['Data']]
                data = [[col.get('VarCharValue', None) for col in row['Data']] 
                       for row in rows[1:]]
                
                return pl.DataFrame(data, schema=headers)
        
        return pl.DataFrame()
    
    def validate_tables(self) -> Dict[str, int]:
        """Validate created tables and return row counts"""
        counts = {}
        for feature_table, output_table in self.created_tables.items():
            query = f"SELECT COUNT(*) as cnt FROM {output_table}"
            query_id = self.execute_athena_query(query)
            
            if self.wait_for_query(query_id):
                result = self.athena.get_query_results(QueryExecutionId=query_id)
                count = result['ResultSet']['Rows'][1]['Data'][0]['VarCharValue']
                counts[output_table] = int(count)
                logger.info(f"Table {output_table}: {count} rows")
        
        return counts


# Example usage
def main():
    # Initialize the pipeline
    pipeline = FeatureTableJoiner(
        database='your_database',
        main_table='main_documents',
        main_doc_col='document_id',
        main_date_col='date_yyyymm',
        s3_output_location='s3://your-bucket/athena-results/',
        region='us-east-1'
    )
    
    # Process pipeline - creates one table per feature table
    dates_to_process = ['202401', '202402', '202403']
    
    created_tables = pipeline.process_pipeline(
        metadata_file='tables_metadata.xlsx',
        date_list=dates_to_process,
        table_prefix='ml_feature'
    )
    
    # Validate results
    row_counts = pipeline.validate_tables()
    print(f"Created {len(created_tables)} tables")
    print(f"Row counts: {row_counts}")
    
    # Read specific table with Polars
    for feature_table, output_table in created_tables.items():
        df = pipeline.read_table_with_polars(
            output_table,
            partition_filter="date_yyyymm = '202401'"
        )
        print(f"\nTable: {output_table}")
        print(f"Shape: {df.shape}")
        print(df.head(3))
        
        # Save to parquet if needed
        df.write_parquet(f"{output_table}.parquet")


if __name__ == "__main__":
    main()