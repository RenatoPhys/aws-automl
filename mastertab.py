"""
Automated ML Pipeline for Master Table Creation
Using AWS Wrangler and Polars in SageMaker Studio
Creates a single master table with all feature tables joined at once
"""

import polars as pl
import pandas as pd
import awswrangler as wr
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MasterTableJoiner:
    """Automated pipeline for creating master table with all feature joins"""
    
    def __init__(self, 
                 database: str,
                 main_table: str,
                 main_doc_col: str,
                 main_date_col: str,
                 s3_output_location: str):
        """
        Initialize the Master Table Joiner
        
        Args:
            database: Athena database name
            main_table: Name of the main/public table
            main_doc_col: Document ID column in main table
            main_date_col: Date column (YYYYMM) in main table
            s3_output_location: S3 location for Athena query results and table storage
        """
        self.database = database
        self.main_table = main_table
        self.main_doc_col = main_doc_col
        self.main_date_col = main_date_col
        self.s3_output_location = s3_output_location.rstrip('/')
        
        # Store master table info
        self.master_table_name = None
        self.feature_columns = {}
        
        # Set awswrangler Athena parameters
        self.athena_params = {
            'database': self.database,
            's3_output': self.s3_output_location,
            'keep_files': True,
            'ctas_approach': False
        }
    
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
    
    def load_top_features(self, features_input: Dict[str, List[str]] | str) -> Dict[str, List[str]]:
        """
        Load top features from dict or JSON file
        
        Args:
            features_input: Either a dict mapping table_name -> features list,
                          or a path to JSON file containing the mapping
        """
        if isinstance(features_input, str):
            with open(features_input, 'r') as f:
                top_features = json.load(f)
        else:
            top_features = features_input
        
        logger.info(f"Loaded top features for {len(top_features)} tables")
        for table, features in top_features.items():
            logger.info(f"  {table}: {len(features)} features")
        
        self.feature_columns = top_features
        return top_features
    
    def calculate_date_offset(self, date_str: str, delay: str) -> str:
        """
        Calculate date with offset based on delay pattern (e.g., 'm-1', 'm-2')
        
        Args:
            date_str: Date string in YYYYMM format
            delay: Delay pattern like 'm-1', 'm-2'
        """
        if not delay or delay == 'm':
            return date_str
        
        current_date = datetime.strptime(date_str, '%Y%m')
        
        if 'm-' in delay:
            months_back = int(delay.split('-')[1])
            new_date = current_date - relativedelta(months=months_back)
            return new_date.strftime('%Y%m')
        
        return date_str
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute Athena query using awswrangler and return results
        
        Args:
            query: SQL query to execute
            
        Returns:
            Query results as pandas DataFrame
        """
        try:
            df = wr.athena.read_sql_query(
                sql=query,
                database=self.database,
                s3_output=self.s3_output_location,
                keep_files=False,
                ctas_approach=False
            )
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def execute_query_no_return(self, query: str) -> None:
        """
        Execute Athena query without expecting results (DDL operations)
        
        Args:
            query: SQL query to execute
        """
        try:
            wr.athena.start_query_execution(
                sql=query,
                database=self.database,
                s3_output=self.s3_output_location,
                wait=True
            )
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_table_columns(self, table_name: str) -> List[Tuple[str, str]]:
        """
        Get column names and types for a table using awswrangler
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of (column_name, column_type) tuples
        """
        try:
            # Get table metadata using awswrangler
            table_metadata = wr.catalog.table(
                database=self.database,
                table=table_name
            )
            
            columns = []
            for col in table_metadata['Column Name']:
                idx = table_metadata['Column Name'].tolist().index(col)
                col_type = table_metadata['Type'][idx]
                # Skip partition columns
                if col != self.main_date_col:
                    columns.append((col, col_type))
            
            return columns
        except Exception as e:
            logger.warning(f"Could not get metadata for {table_name}: {e}")
            # Fallback to DESCRIBE query
            query = f"DESCRIBE {table_name}"
            df = self.execute_query(query)
            
            columns = []
            for _, row in df.iterrows():
                col_name = row.get('col_name', row.get('Column', ''))
                col_type = row.get('data_type', row.get('Type', ''))
                if col_name and not col_name.startswith('#'):
                    columns.append((col_name, col_type))
            return columns
    
    def create_master_table(self, 
                           metadata: pl.DataFrame,
                           top_features: Dict[str, List[str]],
                           output_table: str = None) -> str:
        """
        Create empty master table with schema for all joined data
        
        Args:
            metadata: DataFrame with feature table metadata
            top_features: Dict mapping table_name -> list of columns to include
            output_table: Optional output table name
        """
        if not output_table:
            output_table = f"master_table_{int(time.time())}"
        
        # Get main table columns
        main_columns = self.get_table_columns(self.main_table)
        
        # Build column definitions
        columns = []
        
        # Add main table columns
        for col_name, col_type in main_columns:
            if col_name != self.main_date_col:
                columns.append(f"{col_name} {col_type}")
        
        # Add selected features from each feature table
        for row in metadata.iter_rows(named=True):
            table_name = row['table_name']
            if table_name in top_features:
                # Get columns for this feature table
                feature_columns = self.get_table_columns(table_name)
                column_dict = {col[0]: col[1] for col in feature_columns}
                
                # Add only the selected features
                for feature_col in top_features[table_name]:
                    if feature_col in column_dict:
                        col_type = column_dict[feature_col]
                        # Prefix with table name to avoid conflicts
                        columns.append(f"{table_name}_{feature_col} {col_type}")
                    else:
                        logger.warning(f"Column {feature_col} not found in {table_name}")
        
        # Create table using awswrangler
        location = f"{self.s3_output_location}/{output_table}/"
        
        create_query = f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS {output_table} (
            {', '.join(columns)}
        )
        PARTITIONED BY ({self.main_date_col} string)
        STORED AS PARQUET
        LOCATION '{location}'
        """
        
        logger.info(f"Creating master table: {output_table}")
        self.execute_query_no_return(create_query)
        
        logger.info(f"Successfully created master table: {output_table}")
        self.master_table_name = output_table
        return output_table
    
    def insert_master_data(self,
                          metadata: pl.DataFrame,
                          top_features: Dict[str, List[str]],
                          target_date: str):
        """
        Insert data into master table with all joins in single query
        
        Args:
            metadata: DataFrame with feature table metadata
            top_features: Dict mapping table_name -> list of columns
            target_date: Date to process (YYYYMM)
        """
        # Get main table columns
        main_columns = self.get_table_columns(self.main_table)
        select_cols = []
        
        # Main table columns
        for col_name, _ in main_columns:
            if col_name != self.main_date_col:
                select_cols.append(f"m.{col_name}")
        
        # Build JOIN clauses
        join_clauses = []
        table_counter = 0
        
        for row in metadata.iter_rows(named=True):
            table_name = row['table_name']
            if table_name not in top_features:
                continue
            
            table_counter += 1
            alias = f"t{table_counter}"  # Simple alias t1, t2, t3...
            
            doc_col = row['doc_column_name']
            date_col = row['date_column_name']
            doc_type = row['doc_column_type']
            date_type = row['date_column_type']
            delay = row['date_delay']
            
            # Calculate adjusted date
            adjusted_date = self.calculate_date_offset(target_date, delay)
            
            # Add feature columns to SELECT
            for feature_col in top_features[table_name]:
                select_cols.append(
                    f"COALESCE({alias}.{feature_col}, NULL) AS {table_name}_{feature_col}"
                )
            
            # Build join condition with type casting
            doc_cast = f"CAST({alias}.{doc_col} AS VARCHAR)" if doc_type != 'string' else f"{alias}.{doc_col}"
            main_doc_cast = f"CAST(m.{self.main_doc_col} AS VARCHAR)" if doc_type != 'string' else f"m.{self.main_doc_col}"
            date_cast = f"CAST({alias}.{date_col} AS VARCHAR)" if date_type != 'string' else f"{alias}.{date_col}"
            
            join_clause = f"""
        LEFT JOIN {table_name} {alias}
            ON {doc_cast} = {main_doc_cast}
            AND {date_cast} = '{adjusted_date}'"""
            
            join_clauses.append(join_clause)
        
        # Build complete INSERT query
        insert_query = f"""
        INSERT INTO {self.master_table_name}
        SELECT 
            {', '.join(select_cols)},
            '{target_date}' as {self.main_date_col}
        FROM {self.main_table} m
        {' '.join(join_clauses)}
        WHERE m.{self.main_date_col} = '{target_date}'
        """
        
        logger.info(f"Inserting data for date: {target_date}")
        self.execute_query_no_return(insert_query)
        logger.info(f"Successfully inserted data for {target_date} - partition auto-registered")
    
    def process_pipeline(self, 
                        metadata_file: str,
                        top_features: Dict[str, List[str]] | str,
                        date_list: List[str],
                        output_table: str = None) -> str:
        """
        Main pipeline: Create master table and process all dates
        
        Args:
            metadata_file: Path to metadata file
            top_features: Dict or JSON file path with table_name -> features mapping
            date_list: List of dates in YYYYMM format to process
            output_table: Optional name for master table
        
        Returns:
            Name of created master table
        """
        # Load metadata and features
        metadata = self.read_metadata(metadata_file)
        top_features = self.load_top_features(top_features)
        
        # Step 1: Create empty master table
        logger.info("="*50)
        logger.info("Step 1: Creating master table schema...")
        logger.info("="*50)
        
        master_table = self.create_master_table(
            metadata=metadata,
            top_features=top_features,
            output_table=output_table
        )
        
        # Step 2: Insert data for each date
        logger.info("="*50)
        logger.info("Step 2: Inserting data for all dates...")
        logger.info("="*50)
        
        for i, date in enumerate(date_list, 1):
            logger.info(f"Processing date {i}/{len(date_list)}: {date}")
            self.insert_master_data(
                metadata=metadata,
                top_features=top_features,
                target_date=date
            )
        
        # Step 3: Validate results
        logger.info("="*50)
        logger.info("Step 3: Validating master table...")
        logger.info("="*50)
        
        validation_results = self.validate_master_table()
        
        logger.info("="*50)
        logger.info(f"✓ Pipeline completed successfully!")
        logger.info(f"✓ Master table: {master_table}")
        logger.info(f"✓ Total rows: {validation_results['total_rows']}")
        logger.info(f"✓ Partitions: {validation_results['partitions']}")
        logger.info("="*50)
        
        return master_table
    
    def validate_master_table(self) -> Dict:
        """Validate the created master table"""
        results = {}
        
        # Get total row count
        query = f"SELECT COUNT(*) as cnt FROM {self.master_table_name}"
        df = self.execute_query(query)
        results['total_rows'] = int(df['cnt'].iloc[0])
        
        # Get partition count
        query = f"""
        SELECT COUNT(DISTINCT {self.main_date_col}) as cnt 
        FROM {self.master_table_name}
        """
        df = self.execute_query(query)
        results['partitions'] = int(df['cnt'].iloc[0])
        
        return results
    
    def read_master_table(self, 
                         partition_filter: str = None,
                         columns: List[str] = None,
                         as_polars: bool = True) -> pl.DataFrame | pd.DataFrame:
        """
        Read the master table using awswrangler
        
        Args:
            partition_filter: Optional WHERE clause for partitions
            columns: Optional list of columns to select
            as_polars: If True, return as Polars DataFrame, else Pandas
            
        Returns:
            DataFrame with master table data
        """
        # Build query
        cols = ', '.join(columns) if columns else '*'
        query = f"SELECT {cols} FROM {self.master_table_name}"
        if partition_filter:
            query += f" WHERE {partition_filter}"
        
        # Execute with awswrangler
        df_pandas = wr.athena.read_sql_query(
            sql=query,
            database=self.database,
            s3_output=self.s3_output_location,
            keep_files=False
        )
        
        if as_polars:
            return pl.from_pandas(df_pandas)
        return df_pandas
    
    def read_master_table_partitioned(self, 
                                     partitions: List[str],
                                     columns: List[str] = None) -> pl.DataFrame:
        """
        Read specific partitions from master table efficiently
        
        Args:
            partitions: List of partition values (dates in YYYYMM format)
            columns: Optional list of columns to select
            
        Returns:
            Polars DataFrame with the data
        """
        # Build partition filter
        partition_values = ', '.join([f"'{p}'" for p in partitions])
        partition_filter = f"{self.main_date_col} IN ({partition_values})"
        
        return self.read_master_table(
            partition_filter=partition_filter,
            columns=columns,
            as_polars=True
        )
    
    def get_table_statistics(self) -> pd.DataFrame:
        """Get statistics about the master table"""
        query = f"""
        SELECT 
            {self.main_date_col} as partition_date,
            COUNT(*) as row_count,
            COUNT(DISTINCT {self.main_doc_col}) as unique_docs
        FROM {self.master_table_name}
        GROUP BY {self.main_date_col}
        ORDER BY {self.main_date_col}
        """
        
        return self.execute_query(query)


# Example usage
def main():
    # Initialize the master table pipeline
    pipeline = MasterTableJoiner(
        database='your_database',
        main_table='main_documents',
        main_doc_col='document_id',
        main_date_col='date_yyyymm',
        s3_output_location='s3://your-bucket/athena-results/'
    )
    
    # Define top k features for each table
    top_features = {
        'feature_table_1': ['score', 'flag', 'amount', 'category'],
        'feature_table_2': ['risk_level', 'priority', 'status'],
        'feature_table_3': ['value', 'rating', 'type', 'class']
    }
    
    # Or load from JSON file
    # top_features = 'top_features.json'
    
    # Process pipeline - creates single master table
    dates_to_process = ['202401', '202402', '202403']
    
    master_table = pipeline.process_pipeline(
        metadata_file='tables_metadata.xlsx',
        top_features=top_features,
        date_list=dates_to_process,
        output_table='ml_master_table'
    )
    
    # Read sample data using awswrangler
    df = pipeline.read_master_table(
        partition_filter="date_yyyymm = '202401'",
        as_polars=True
    )
    print(f"\nMaster Table Sample:")
    print(f"Shape: {df.shape}")
    print(df.head(5))
    
    # Get statistics
    stats = pipeline.get_table_statistics()
    print(f"\nTable Statistics:")
    print(stats)
    
    # Read multiple partitions efficiently
    df_multi = pipeline.read_master_table_partitioned(
        partitions=['202401', '202402'],
        columns=['document_id', 'feature_table_1_score', 'feature_table_2_risk_level']
    )
    print(f"\nMulti-partition data shape: {df_multi.shape}")
    
    # Save to parquet
    df.write_parquet(f"{master_table}_sample.parquet")


if __name__ == "__main__":
    main()