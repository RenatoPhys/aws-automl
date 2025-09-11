import boto3
import polars as pl
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import lightgbm as lgb
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import io
import warnings
warnings.filterwarnings('ignore')

class ModelPipeline:
    """
    A simplified pipeline for training LightGBM classification models on multiple tables from S3 using Polars,
    with minimal preprocessing (no encoding, no missing value filling).
    """
    
    def __init__(self, 
                 s3_paths: List[str], 
                 target_column: str,
                 date_column: str,
                 train_cutoff_date: int,
                 top_k_features: int = 10,
                 lgbm_params: Optional[Dict] = None):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        s3_paths : list
            List of S3 paths to the tables
        target_column : str
            Name of the target variable column
        date_column : str
            Name of the date column for train/test split (integer YYYYMM format)
        train_cutoff_date : int
            Date to split train/test (format: YYYYMM, e.g., 202401 for Jan 2024)
        top_k_features : int
            Number of top important features to extract
        lgbm_params : dict, optional
            Custom LightGBM parameters
        """
        self.s3_paths = s3_paths
        self.target_column = target_column
        self.date_column = date_column
        self.train_cutoff_date = train_cutoff_date
        self.top_k_features = top_k_features
        self.s3_client = boto3.client('s3')
        self.results = {}
        
        # Default LightGBM parameters for classification
        self.lgbm_params = lgbm_params or {
            'objective': 'binary',  # or 'multiclass' for multi-class
            'metric': 'binary_logloss',  # or 'multi_logloss'
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100,
            'early_stopping_rounds': 10,
            'use_missing': True,  # Allow LightGBM to handle missing values
            'zero_as_missing': False
        }
        
    def read_s3_table(self, s3_path: str) -> pl.DataFrame:
        """
        Read a table from S3 using Polars.
        
        Parameters:
        -----------
        s3_path : str
            S3 path (e.g., 's3://bucket/path/to/file.csv')
        
        Returns:
        --------
        pl.DataFrame
        """
        # Parse S3 path
        path_parts = s3_path.replace("s3://", "").split("/")
        bucket = path_parts[0]
        key = "/".join(path_parts[1:])
        
        # Get object from S3
        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        
        # Read based on file type
        if s3_path.endswith('.csv'):
            df = pl.read_csv(io.BytesIO(obj['Body'].read()))
        elif s3_path.endswith('.parquet'):
            df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        else:
            raise ValueError(f"Unsupported file format for {s3_path}")
            
        return df
    
    def split_by_date(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split data into train and test based on integer date partition (YYYYMM).
        
        Parameters:
        -----------
        df : pl.DataFrame
            Input dataframe
        
        Returns:
        --------
        tuple of (train_df, test_df)
        """
        # Ensure date column is integer
        if df[self.date_column].dtype != pl.Int32 and df[self.date_column].dtype != pl.Int64:
            df = df.with_columns(
                pl.col(self.date_column).cast(pl.Int32)
            )
        
        # Split by date (integer comparison)
        train_df = df.filter(pl.col(self.date_column) <= self.train_cutoff_date)
        test_df = df.filter(pl.col(self.date_column) > self.train_cutoff_date)
        
        print(f"  Date range in train: {train_df[self.date_column].min()} to {train_df[self.date_column].max()}")
        print(f"  Date range in test: {test_df[self.date_column].min()} to {test_df[self.date_column].max()}")
        
        return train_df, test_df
    
    def prepare_features(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, np.ndarray, List[str], List[str]]:
        """
        Minimal feature preparation - just separate features and target.
        Filters out datetime columns and lets LightGBM handle categorical features and missing values natively.
        
        Parameters:
        -----------
        df : pl.DataFrame
            Input dataframe
        
        Returns:
        --------
        tuple of (X_dataframe, y_array, feature_names, categorical_features)
        """
        # Get all columns excluding target and date partition
        all_feature_cols = [col for col in df.columns 
                           if col not in [self.target_column, self.date_column]]
        
        # Filter out datetime columns
        datetime_types = [pl.Datetime, pl.Date, pl.Time, pl.Duration]
        feature_cols = []
        excluded_datetime_cols = []
        
        for col in all_feature_cols:
            if df[col].dtype in datetime_types:
                excluded_datetime_cols.append(col)
            else:
                feature_cols.append(col)
        
        if excluded_datetime_cols:
            print(f"  Excluded {len(excluded_datetime_cols)} datetime columns: {', '.join(excluded_datetime_cols[:5])}" + 
                  (f"... (+{len(excluded_datetime_cols)-5} more)" if len(excluded_datetime_cols) > 5 else ""))
        
        # Select features and target
        X = df.select(feature_cols)
        y = df[self.target_column].to_numpy()
        
        # Identify categorical columns for LightGBM
        categorical_features = [col for col in feature_cols 
                              if X[col].dtype == pl.Utf8]
        
        # Identify Decimal columns (will be converted to Float64 later)
        decimal_features = [col for col in feature_cols 
                           if X[col].dtype == pl.Decimal]
        
        # Ensure target is integer for classification
        if y.dtype not in [np.int32, np.int64]:
            y = y.astype(np.int32)
        
        print(f"  Features: {len(feature_cols)} total, {len(categorical_features)} categorical, {len(decimal_features)} decimal")
        print(f"  Missing values total: {X.null_count().sum_horizontal()[0]}")
        
        return X, y, feature_cols, categorical_features
    
    def train_lgbm_model(self, X_train: pl.DataFrame, y_train: np.ndarray,
                    X_test: pl.DataFrame, y_test: np.ndarray,
                    feature_names: List[str], 
                    categorical_features: List[str]) -> Tuple[lgb.LGBMClassifier, Dict, Dict, np.ndarray, np.ndarray]:
        """
        Train LightGBM model with native categorical support and missing value handling.
        Now returns train and test prediction probabilities for KS calculation.
        
        Returns:
        --------
        tuple of (model, metrics_dict, additional_info, y_train_pred_proba, y_test_pred_proba)
        """
        
        # Handle Decimal columns - convert to Float64 before pandas conversion
        decimal_cols = [col for col in X_train.columns if X_train[col].dtype == pl.Decimal]
        if decimal_cols:
            print(f"  Converting {len(decimal_cols)} Decimal columns to Float64")
            X_train = X_train.with_columns([
                pl.col(col).cast(pl.Float64) for col in decimal_cols
            ])
            X_test = X_test.with_columns([
                pl.col(col).cast(pl.Float64) for col in decimal_cols
            ])

        # Convert Polars DataFrames to pandas for LightGBM
        X_train_pd = X_train.to_pandas()
        X_test_pd = X_test.to_pandas()
        
        # Convert categorical columns to 'category' dtype for LightGBM
        for cat_col in categorical_features:
            if cat_col in X_train_pd.columns:
                X_train_pd[cat_col] = X_train_pd[cat_col].astype('category')
                X_test_pd[cat_col] = X_test_pd[cat_col].astype('category')
        
        # Determine number of classes
        n_classes = len(np.unique(y_train))
        
        # Update parameters based on number of classes
        params = self.lgbm_params.copy()
        if n_classes > 2:
            params['objective'] = 'multiclass'
            params['metric'] = 'multi_logloss'
            params['num_class'] = n_classes
        
        # Create LightGBM classifier
        model = lgb.LGBMClassifier(**params)
        
        # Train model with validation set for early stopping
        model.fit(
            X_train_pd, y_train,
            eval_set=[(X_test_pd, y_test)],
            eval_metric='logloss' if n_classes == 2 else 'multi_logloss',
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Make predictions
        y_pred = model.predict(X_test_pd)
        y_test_pred_proba = model.predict_proba(X_test_pd)
        
        # Get train predictions for overfitting analysis
        y_train_pred_proba = model.predict_proba(X_train_pd)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC for binary classification
        if n_classes == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_test_pred_proba[:, 1])
        else:
            # Multi-class ROC AUC
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = None
        
        # Additional information
        additional_info = {
            'n_classes': n_classes,
            'best_iteration': model.best_iteration_,
            'feature_names': feature_names,
            'categorical_features': categorical_features,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'class_distribution_train': dict(zip(*np.unique(y_train, return_counts=True))),
            'class_distribution_test': dict(zip(*np.unique(y_test, return_counts=True)))
        }
        
        return model, metrics, additional_info, y_train_pred_proba, y_test_pred_proba
    
    def get_feature_importance(self, model: lgb.LGBMClassifier, 
                              feature_names: List[str],
                              importance_type: str = 'gain') -> Dict[str, float]:
        """
        Extract feature importance from LightGBM model.
        
        Parameters:
        -----------
        model : LightGBM model
        feature_names : List of feature names
        importance_type : 'gain' or 'split'
        
        Returns:
        --------
        Dictionary of feature importance scores
        """
        # Get importance scores
        importance_scores = model.feature_importances_
        
        # Create dictionary
        feature_importance = dict(zip(feature_names, importance_scores))
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True))
        
        # Normalize scores to sum to 1
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        return feature_importance
    
    def calculate_ks_statistic(y_true, y_pred_proba):
        """
        Calculate Kolmogorov-Smirnov statistic for binary classification.
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_pred_proba : array-like
            Predicted probabilities for positive class
        
        Returns:
        --------
        float : KS statistic
        """
        # Get predictions for each class
        preds_class0 = y_pred_proba[y_true == 0]
        preds_class1 = y_pred_proba[y_true == 1]
        
        # Calculate KS statistic
        ks_statistic = stats.ks_2samp(preds_class0, preds_class1).statistic
        
        return ks_statistic

    def plot_diagnostics(self, table_name: str, metrics: Dict, 
                        feature_importance: Dict, additional_info: Dict,
                        y_train: np.ndarray, y_test: np.ndarray,
                        y_train_pred_proba: np.ndarray, y_test_pred_proba: np.ndarray) -> None:
        """
        Create simplified diagnostic plots for a table.
        
        Parameters:
        -----------
        table_name : str
            Name of the table
        metrics : dict
            Performance metrics
        feature_importance : dict
            Feature importance scores
        additional_info : dict
            Additional model information
        y_train : np.ndarray
            Training target values
        y_test : np.ndarray
            Test target values
        y_train_pred_proba : np.ndarray
            Training predictions probabilities
        y_test_pred_proba : np.ndarray
            Test predictions probabilities
        """
        fig = plt.figure(figsize=(20, 10))
        
        # Create grid - custom layout
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25, 
                            width_ratios=[2, 1, 1.5])
        
        # Plot 1: Top K Feature Importance (left side, spanning 2 rows)
        ax1 = fig.add_subplot(gs[:, 0])
        top_features = dict(list(feature_importance.items())[:self.top_k_features])
        
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        bars = ax1.barh(y_pos, importances, color='steelblue', edgecolor='navy', alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features, fontsize=10)
        ax1.set_xlabel('Normalized Importance Score', fontsize=11)
        ax1.set_title(f'Top {self.top_k_features} Important Features - {table_name}', 
                    fontsize=13, fontweight='bold', pad=15)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, value in zip(bars, importances):
            width = bar.get_width()
            ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
                    f'{value:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Plot 2: Model Information (right of feature importance, top)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        # Get class distribution info
        train_dist = additional_info['class_distribution_train']
        test_dist = additional_info['class_distribution_test']
        train_total = sum(train_dist.values())
        test_total = sum(test_dist.values())
        
        # Get top 3 features for display
        top_3_features = features[:3] if len(features) >= 3 else features
        
        info_text = f"""MODEL INFORMATION
    {'='*25}

    Dataset: {table_name}

    SAMPLES:
    Train: {train_total:,}
    Test:  {test_total:,}
    Total: {train_total + test_total:,}

    FEATURES:
    Total: {len(additional_info['feature_names'])}
    Categorical: {len(additional_info['categorical_features'])}
    Numeric: {len(additional_info['feature_names']) - len(additional_info['categorical_features'])}

    MODEL:
    Classes: {additional_info['n_classes']}
    Best Iter: {additional_info['best_iteration']}
    Algorithm: LightGBM

    TOP 3 FEATURES:
    """
        
        for i, feat in enumerate(top_3_features, 1):
            # Truncate long feature names
            feat_display = feat[:20] + '...' if len(feat) > 23 else feat
            info_text += f"  {i}. {feat_display}\n"
        
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.2))
        
        # Plot 3: Class Distribution (bottom middle)
        ax3 = fig.add_subplot(gs[1, 1])
        
        classes = sorted(set(list(train_dist.keys()) + list(test_dist.keys())))
        train_counts = [train_dist.get(c, 0) for c in classes]
        test_counts = [test_dist.get(c, 0) for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, train_counts, width, label='Train', 
                    color='skyblue', edgecolor='navy', alpha=0.8)
        bars2 = ax3.bar(x + width/2, test_counts, width, label='Test', 
                    color='lightcoral', edgecolor='darkred', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=9)
        
        ax3.set_xlabel('Class', fontsize=11)
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title('Class Distribution', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(classes)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Plot 4: ROC AUC & KS Metrics (right side, spanning both rows)
        ax4 = fig.add_subplot(gs[:, 2])
        ax4.axis('off')
        
        # Calculate metrics for binary classification
        metrics_text = f"""PERFORMANCE METRICS
    {'='*30}

    """
        
        n_classes = additional_info['n_classes']
        
        if n_classes == 2:
            # Binary classification - calculate ROC AUC and KS
            try:
                # ROC AUC for train and test
                train_roc_auc = roc_auc_score(y_train, y_train_pred_proba[:, 1])
                test_roc_auc = roc_auc_score(y_test, y_test_pred_proba[:, 1])
                
                # KS Statistic for train and test
                train_ks = calculate_ks_statistic(y_train, y_train_pred_proba[:, 1])
                test_ks = calculate_ks_statistic(y_test, y_test_pred_proba[:, 1])
                
                # Determine overfitting status
                roc_diff = abs(train_roc_auc - test_roc_auc)
                ks_diff = abs(train_ks - test_ks)
                
                if roc_diff < 0.03:
                    overfit_status = "✅ Low"
                    overfit_color = "green"
                elif roc_diff < 0.07:
                    overfit_status = "⚠️  Moderate"
                    overfit_color = "orange"
                else:
                    overfit_status = "❌ High"
                    overfit_color = "red"
                
                metrics_text += f"""TRAIN METRICS:
    ROC AUC:      {train_roc_auc:.4f}
    KS Statistic: {train_ks:.4f}
    
    TEST METRICS:
    ROC AUC:      {test_roc_auc:.4f}
    KS Statistic: {test_ks:.4f}
    
    OVERFITTING CHECK:
    ROC AUC Diff: {roc_diff:.4f}
    KS Diff:      {ks_diff:.4f}
    Status: {overfit_status}
    
    {'='*30}
    """
                
            except Exception as e:
                metrics_text += f"Error calculating metrics:\n{str(e)[:50]}\n\n"
        else:
            # Multi-class classification
            try:
                train_roc_auc = roc_auc_score(y_train, y_train_pred_proba, 
                                            multi_class='ovr', average='weighted')
                test_roc_auc = roc_auc_score(y_test, y_test_pred_proba, 
                                            multi_class='ovr', average='weighted')
                
                roc_diff = abs(train_roc_auc - test_roc_auc)
                
                if roc_diff < 0.03:
                    overfit_status = "✅ Low"
                elif roc_diff < 0.07:
                    overfit_status = "⚠️  Moderate"
                else:
                    overfit_status = "❌ High"
                
                metrics_text += f"""TRAIN METRICS:
    ROC AUC (weighted): {train_roc_auc:.4f}
    
    TEST METRICS:
    ROC AUC (weighted): {test_roc_auc:.4f}
    
    OVERFITTING CHECK:
    ROC AUC Diff: {roc_diff:.4f}
    Status: {overfit_status}
    
    Note: KS statistic N/A for
    multi-class classification

    {'='*30}
    """
                
            except Exception as e:
                metrics_text += f"Error calculating metrics:\n{str(e)[:50]}\n\n"
        
        metrics_text += f"""
    BASIC METRICS (Test):
    Accuracy:  {metrics['accuracy']:.4f}
    F1 Score:  {metrics['f1']:.4f}
    Precision: {metrics['precision']:.4f}
    Recall:    {metrics['recall']:.4f}

    {'='*30}

    {'='*30}

    CONFUSION MATRIX:
    """
        
        # Add confusion matrix as text
        cm = np.array(additional_info['confusion_matrix'])
        if cm.shape[0] == 2:
            metrics_text += f"""
        Pred 0  Pred 1
    Act 0  {cm[0,0]:5d}  {cm[0,1]:5d}
    Act 1  {cm[1,0]:5d}  {cm[1,1]:5d}
    """
        else:
            metrics_text += f"\n{cm.shape[0]} classes (see full matrix in logs)"
        
        ax4.text(0.05, 0.98, metrics_text, transform=ax4.transAxes, 
                fontsize=9.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.2))
        
        plt.suptitle(f'Model Diagnostic Report - {table_name}', 
                    fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
    
    def run_pipeline(self) -> Dict[str, Dict]:
        """
        Run the complete pipeline for all tables.
        
        Returns:
        --------
        Dictionary with results for each table
        """
        all_results = {}
        
        print(f"\n{'='*80}")
        print(f"LIGHTGBM CLASSIFICATION PIPELINE (MINIMAL PREPROCESSING)")
        print(f"Train/Test Split Date: {self.train_cutoff_date}")
        print(f"Number of tables to process: {len(self.s3_paths)}")
        print(f"Preprocessing: None (LightGBM handles categorical & missing values)")
        print(f"{'='*80}")
        
        for i, s3_path in enumerate(self.s3_paths):
            print(f"\n{'-'*60}")
            print(f"Processing table {i+1}/{len(self.s3_paths)}: {s3_path}")
            print(f"{'-'*60}")
            
            try:
                # Extract table name from path
                table_name = s3_path.split('/')[-1].split('.')[0]
                
                # Read data
                print(f"📊 Reading data from S3...")
                df = self.read_s3_table(s3_path)
                print(f"  Data shape: {df.shape}")
                print(f"  Columns: {', '.join(df.columns[:10])}" + 
                    (f"... (+{len(df.columns)-10} more)" if len(df.columns) > 10 else ""))
                
                # Split by date
                print(f"\n📅 Splitting data by date (cutoff: {self.train_cutoff_date})...")
                train_df, test_df = self.split_by_date(df)
                print(f"  Train shape: {train_df.shape}, Test shape: {test_df.shape}")
                
                # Prepare features (minimal processing)
                print(f"\n🔧 Preparing features (minimal preprocessing)...")
                X_train, y_train, feature_names, categorical_features = self.prepare_features(train_df)
                X_test, y_test, _, _ = self.prepare_features(test_df)
                
                # Train model and get metrics (now also returns prediction probabilities)
                print(f"\n🚀 Training LightGBM classifier...")
                model, metrics, additional_info, y_train_pred_proba, y_test_pred_proba = self.train_lgbm_model(
                    X_train, y_train, X_test, y_test, feature_names, categorical_features
                )
                print(f"  Training completed at iteration {additional_info['best_iteration']}")
                
                # Get feature importance
                print(f"\n📈 Extracting feature importance...")
                feature_importance = self.get_feature_importance(model, feature_names)
                
                # Get top K features
                top_k_features = dict(list(feature_importance.items())[:self.top_k_features])
                
                # Calculate additional metrics for storage
                n_classes = additional_info['n_classes']
                if n_classes == 2:
                    try:
                        # Calculate KS statistics
                        from scipy import stats
                        
                        # KS for train
                        preds_class0_train = y_train_pred_proba[y_train == 0][:, 1]
                        preds_class1_train = y_train_pred_proba[y_train == 1][:, 1]
                        train_ks = stats.ks_2samp(preds_class0_train, preds_class1_train).statistic
                        
                        # KS for test
                        preds_class0_test = y_test_pred_proba[y_test == 0][:, 1]
                        preds_class1_test = y_test_pred_proba[y_test == 1][:, 1]
                        test_ks = stats.ks_2samp(preds_class0_test, preds_class1_test).statistic
                        
                        # ROC AUC for train
                        from sklearn.metrics import roc_auc_score
                        train_roc_auc = roc_auc_score(y_train, y_train_pred_proba[:, 1])
                        
                        additional_metrics = {
                            'train_roc_auc': train_roc_auc,
                            'test_roc_auc': metrics['roc_auc'],
                            'train_ks': train_ks,
                            'test_ks': test_ks,
                            'overfitting_roc_auc': abs(train_roc_auc - metrics['roc_auc']),
                            'overfitting_ks': abs(train_ks - test_ks)
                        }
                    except Exception as e:
                        print(f"  Warning: Could not calculate KS statistics: {e}")
                        additional_metrics = {}
                else:
                    # Multi-class case
                    try:
                        from sklearn.metrics import roc_auc_score
                        train_roc_auc = roc_auc_score(y_train, y_train_pred_proba, 
                                                    multi_class='ovr', average='weighted')
                        additional_metrics = {
                            'train_roc_auc': train_roc_auc,
                            'test_roc_auc': metrics.get('roc_auc'),
                            'overfitting_roc_auc': abs(train_roc_auc - metrics.get('roc_auc', 0)) if metrics.get('roc_auc') else None
                        }
                    except Exception as e:
                        print(f"  Warning: Could not calculate multi-class ROC AUC: {e}")
                        additional_metrics = {}
                
                # Store results
                all_results[table_name] = {
                    'metrics': metrics,
                    'additional_metrics': additional_metrics,
                    'feature_importance': feature_importance,
                    'top_k_features': list(top_k_features.keys()),
                    'top_k_importance_scores': top_k_features,
                    'train_samples': train_df.height,
                    'test_samples': test_df.height,
                    'total_features': len(feature_names),
                    'additional_info': additional_info
                }
                
                # Print summary
                print(f"\n✅ Results for {table_name}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 Score: {metrics['f1']:.4f}")
                print(f"  Test ROC AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "  ROC AUC: N/A")
                
                if additional_metrics:
                    if 'train_roc_auc' in additional_metrics:
                        print(f"  Train ROC AUC: {additional_metrics['train_roc_auc']:.4f}")
                    if 'test_ks' in additional_metrics:
                        print(f"  Test KS: {additional_metrics['test_ks']:.4f}")
                    if 'overfitting_roc_auc' in additional_metrics and additional_metrics['overfitting_roc_auc'] is not None:
                        overfitting_level = "Low" if additional_metrics['overfitting_roc_auc'] < 0.05 else "Moderate" if additional_metrics['overfitting_roc_auc'] < 0.1 else "High"
                        print(f"  Overfitting: {overfitting_level} (ROC AUC diff: {additional_metrics['overfitting_roc_auc']:.4f})")
                
                print(f"  Top 3 Features: {', '.join(list(top_k_features.keys())[:3])}")
                
                # Create diagnostic plots with new parameters
                self.plot_diagnostics(
                    table_name, 
                    metrics, 
                    feature_importance, 
                    additional_info,
                    y_train,
                    y_test,
                    y_train_pred_proba,
                    y_test_pred_proba
                )
                
            except Exception as e:
                print(f"❌ Error processing {s3_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                all_results[table_name] = {'error': str(e)}
        
        self.results = all_results
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETED")
        print(f"Successfully processed: {sum(1 for r in all_results.values() if 'error' not in r)}/{len(self.s3_paths)} tables")
        print(f"{'='*80}")
        
        return all_results
    
    def create_feature_dictionary(self) -> Dict[str, List[str]]:
        """
        Create a dictionary mapping each table to its top K most important features.
        
        Returns:
        --------
        Dictionary with table names as keys and list of top features as values
        """
        feature_dict = {}
        
        for table_name, results in self.results.items():
            if 'error' not in results:
                feature_dict[table_name] = results['top_k_features']
            else:
                feature_dict[table_name] = []
        
        return feature_dict
    
    def save_results(self, output_path: str = 'pipeline_results.json') -> None:
        """
        Save results to a JSON file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the results
        """
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_to_save = convert_types(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n💾 Results saved to {output_path}")
    
    def generate_summary_report(self) -> pl.DataFrame:
        """
        Generate a summary report of all tables using Polars.
        
        Returns:
        --------
        Polars DataFrame with summary statistics
        """
        summary_data = []
        
        for table_name, results in self.results.items():
            if 'error' not in results:
                row = {
                    'table': table_name,
                    'train_samples': results['train_samples'],
                    'test_samples': results['test_samples'],
                    'total_features': results['total_features'],
                    'n_categorical': len(results['additional_info']['categorical_features']),
                    'n_classes': results['additional_info']['n_classes'],
                    'best_iteration': results['additional_info']['best_iteration'],
                    'top_feature_1': results['top_k_features'][0] if len(results['top_k_features']) > 0 else None,
                    'top_feature_2': results['top_k_features'][1] if len(results['top_k_features']) > 1 else None,
                    'top_feature_3': results['top_k_features'][2] if len(results['top_k_features']) > 2 else None,
                }
                
                # Add metrics
                for metric_name, metric_value in results['metrics'].items():
                    row[metric_name] = metric_value
                
                summary_data.append(row)
        
        summary_df = pl.DataFrame(summary_data)
        
        # Sort by accuracy
        if 'accuracy' in summary_df.columns:
            summary_df = summary_df.sort('accuracy', descending=True)
        
        return summary_df
    
    def compare_models(self) -> pl.DataFrame:
        """
        Create a comparison table of all models.
        
        Returns:
        --------
        Polars DataFrame comparing model performances
        """
        comparison_data = []
        
        for table_name, results in self.results.items():
            if 'error' not in results:
                metrics = results['metrics']
                comparison_data.append({
                    'table': table_name,
                    'accuracy': metrics.get('accuracy', 0),
                    'f1_score': metrics.get('f1', 0),
                    'roc_auc': metrics.get('roc_auc', 0) if metrics.get('roc_auc') else 0,
                    'rank_accuracy': 0,  # Will be filled
                    'rank_f1': 0,  # Will be filled
                    'rank_overall': 0  # Will be filled
                })
        
        if not comparison_data:
            return pl.DataFrame()
        
        df = pl.DataFrame(comparison_data)
        
        # Add rankings
        df = df.with_columns([
            pl.col('accuracy').rank(method='dense', descending=True).alias('rank_accuracy'),
            pl.col('f1_score').rank(method='dense', descending=True).alias('rank_f1'),
        ])
        
        # Calculate overall rank (average of individual ranks)
        df = df.with_columns(
            ((pl.col('rank_accuracy') + pl.col('rank_f1')) / 2).alias('rank_overall')
        )
        
        # Sort by overall rank
        df = df.sort('rank_overall')
        
        return df


# Example usage
def main():
    # Define your S3 paths
    s3_paths = [
        's3://your-bucket/data/table1.csv',
        's3://your-bucket/data/table2.parquet',
        's3://your-bucket/data/table3.csv',
        # Add more paths as needed
    ]
    
    # Custom LightGBM parameters (optional)
    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 50,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_child_samples': 20,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 200,
        'early_stopping_rounds': 20,
        'use_missing': True,  # Let LightGBM handle missing values
        'zero_as_missing': False
    }
    
    # Initialize pipeline
    pipeline = ModelPipeline(
        s3_paths=s3_paths,
        target_column='target',  # Replace with your target column name
        date_column='month',      # Replace with your date column name (YYYYMM format)
        train_cutoff_date=202312,  # December 2023 as cutoff (YYYYMM format)
        top_k_features=15,
        lgbm_params=lgbm_params
    )
    
    # Run the pipeline
    results = pipeline.run_pipeline()
    
    # Create feature importance dictionary
    feature_dict = pipeline.create_feature_dictionary()
    print("\n" + "="*60)
    print("📊 FEATURE IMPORTANCE DICTIONARY")
    print("="*60)
    for table, features in feature_dict.items():
        print(f"\n{table}:")
        for i, feature in enumerate(features, 1):
            print(f"  {i:2d}. {feature}")
    
    # Generate summary report
    summary_df = pipeline.generate_summary_report()
    print("\n" + "="*60)
    print("📈 SUMMARY REPORT")
    print("="*60)
    print(summary_df)
    
    # Generate model comparison
    comparison_df = pipeline.compare_models()
    print("\n" + "="*60)
    print("🏆 MODEL COMPARISON (RANKED)")
    print("="*60)
    print(comparison_df)
    
    # Save results
    pipeline.save_results('lgbm_pipeline_results.json')
    
    # Save feature dictionary separately
    with open('feature_importance_dictionary.json', 'w') as f:
        json.dump(feature_dict, f, indent=2)
    print("\n💾 Feature importance dictionary saved to feature_importance_dictionary.json")
    
    # Save summary as CSV using Polars
    summary_df.write_csv('summary_report.csv')
    print("💾 Summary report saved to summary_report.csv")
    
    # Save comparison as CSV
    comparison_df.write_csv('model_comparison.csv')
    print("💾 Model comparison saved to model_comparison.csv")
    
    return results, feature_dict, summary_df


if __name__ == "__main__":
    # Run the pipeline
    results, feature_dict, summary_df = main()