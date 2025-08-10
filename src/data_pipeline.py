import os
import zipfile
import logging
from pathlib import Path
from typing import Union, Dict, Tuple, Optional
from datetime import datetime
import json

import pandas as pd
from sklearn.model_selection import train_test_split
import dotenv

# Configure logging (to print logs and errors in a better way)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables (needed for Kaggle download)
dotenv.load_dotenv()


class DataPipelineConfig:
    """Configuration class for data pipeline parameters."""
    
    def __init__(self, config_path: str):
        """Initialize configuration from JSON file."""
        if Path(config_path).exists():
            self._load_from_file(str(Path(config_path)))
        else:
            raise ValueError(f"Configuration file path doesn't exist: {config_path}")

    def _load_from_file(self, config_path: str):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'paths.raw_data_dir')."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value


class KaggleDataDownloader:
    """Handle Kaggle dataset downloads with proper error handling."""
    
    def __init__(self):
        """Initialize Kaggle API if available."""
        self.kaggle = None
        # Don't import kaggle here - do it lazily when needed
        logger.info("KaggleDataDownloader initialized (API will be loaded when needed)")
    
    def download_file(self, file_name: str, path: Path, dataset_name: str) -> bool:
        """
        Download file from Kaggle competition.
        
        Args:
            file_name: Name of file to download
            path: Download destination path
            dataset_name: Kaggle dataset/competition name
            
        Returns:
            True if successful, False otherwise
        """
        if not self.kaggle:
            try:
                import kaggle
                self.kaggle = kaggle
                logger.info("Kaggle API initialized successfully")
            except ImportError:
                logger.warning("Kaggle package not available. Cannot download data.")
                return False
            except Exception as e:
                logger.warning(f"Failed to initialize Kaggle API: {e}")
                return False
        
        zip_file_name = f"{file_name}.zip"
        zip_file_path = path / zip_file_name
        extracted_file_path = path / file_name

        # Check if file already exists
        if extracted_file_path.exists():
            logger.info(f"{file_name} already exists at {extracted_file_path}")
            return True

        try:
            # Ensure download directory exists
            path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {file_name} from {dataset_name}...")
            
            # Download from Kaggle
            self.kaggle.api.competition_download_file(
                dataset_name,
                file_name=file_name,
                path=path
            )

            # Handle Kaggle's quirky file naming
            downloaded_path = path / file_name
            downloaded_path.rename(zip_file_path)

            # Extract the zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(path)
                    
            # Clean up zip file
            os.remove(zip_file_path)
            logger.info(f"Downloaded and extracted {file_name} successfully")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {file_name}: {str(e)}")
            return False


class DataProcessor:
    """Handle data cleaning and preprocessing operations."""
    
    def __init__(self, config: DataPipelineConfig):
        """Initialize with configuration."""
        self.config = config
    
    def load_and_clean_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load and clean raw data.
        
        Args:
            file_path: Path to raw data file
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Load raw data
            raw_df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(raw_df):,} rows from raw data")
            
            # Select specified columns
            columns_to_keep = self.config.get('dataset.columns_to_keep')
            available_columns = [col for col in columns_to_keep if col in raw_df.columns]
            
            cleaned_df = raw_df[available_columns].copy()
            
            # Drop rows with missing values
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.dropna()
            dropped_rows = initial_rows - len(cleaned_df)
            
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows:,} rows with missing values")
            
            # Clean datetime column if present
            if 'created_date' in cleaned_df.columns:
                cleaned_df['created_date'] = pd.to_datetime(
                    cleaned_df['created_date'], 
                    errors='coerce'
                ).dt.tz_localize(None)
                
                # Drop rows where datetime conversion failed
                datetime_nulls = cleaned_df['created_date'].isnull().sum()
                if datetime_nulls > 0:
                    logger.warning(f"Found {datetime_nulls} rows with invalid dates")
                    cleaned_df = cleaned_df.dropna(subset=['created_date'])
            
            logger.info(f"Data cleaning completed. Final dataset: {len(cleaned_df):,} rows")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Failed to load and clean data: {str(e)}")
            raise
    
    def create_train_val_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits with stratification.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating train/validation/test splits...")
        
        try:
            # Filter data for main split (exclude holdout period)
            holdout_year = self.config.get('processing.holdout_year')
            holdout_start = self.config.get('processing.holdout_month_start')
            
            if 'created_date' in df.columns and holdout_year and holdout_start:
                cutoff_date = pd.Timestamp(f'{holdout_year}-{holdout_start:02d}-01')
                train_val_test_df = df[df['created_date'] < cutoff_date].copy()
                logger.info(f"Filtered to {len(train_val_test_df):,} rows before holdout period")
            else:
                train_val_test_df = df.copy()
                logger.warning("No date filtering applied - using all data for splits")
            
            # Create binary labels for stratification
            stratify_threshold = self.config.get('processing.stratify_threshold', 0.5)
            train_val_test_df['stratify_label'] = (
                train_val_test_df['target'] >= stratify_threshold
            ).astype(int)
            
            # First split: train vs (val + test)
            test_size = self.config.get('processing.test_size', 0.2)
            random_state = self.config.get('processing.random_state', 42)
            
            train_set, val_test_set = train_test_split(
                train_val_test_df,
                test_size=test_size,
                stratify=train_val_test_df['stratify_label'],
                random_state=random_state
            )
            
            # Second split: val vs test
            val_test_ratio = self.config.get('processing.val_test_ratio', 0.5)
            val_set, test_set = train_test_split(
                val_test_set,
                test_size=val_test_ratio,
                stratify=val_test_set['stratify_label'],
                random_state=random_state
            )
            
            # Remove stratification column
            for dataset in [train_set, val_set, test_set]:
                dataset.drop(columns=['stratify_label'], inplace=True)
            
            logger.info(f"Dataset splits created:")
            logger.info(f"  Training: {len(train_set):,} rows")
            logger.info(f"  Validation: {len(val_set):,} rows")
            logger.info(f"  Test: {len(test_set):,} rows")
            
            # Log class distributions
            for name, dataset in [("Train", train_set), ("Validation", val_set), ("Test", test_set)]:
                pos_rate = (dataset['target'] >= stratify_threshold).mean()
                logger.info(f"  {name} positive rate: {pos_rate:.3f}")
            
            return train_set, val_set, test_set
            
        except Exception as e:
            logger.error(f"Failed to create data splits: {str(e)}")
            raise
    
    def create_holdout_splits(self, df: pd.DataFrame, output_dir: Path) -> Dict[str, int]:
        """
        Create monthly holdout datasets.
        
        Args:
            df: Cleaned DataFrame
            output_dir: Directory to save holdout files
            
        Returns:
            Dictionary with month -> sample count
        """
        logger.info("Creating holdout splits by month...")
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            holdout_year = self.config.get('processing.holdout_year')
            holdout_start = self.config.get('processing.holdout_month_start')
            holdout_end = self.config.get('processing.holdout_month_end')
            
            holdout_stats = {}
            
            for month in range(holdout_start, holdout_end + 1):
                # Filter data for this month
                month_mask = (
                    (df['created_date'].dt.year == holdout_year) & 
                    (df['created_date'].dt.month == month)
                )
                month_data = df[month_mask].copy()
                
                if len(month_data) > 0:
                    # Save to file
                    filename = f'{holdout_year}_{month:02d}.csv'
                    filepath = output_dir / filename
                    month_data.to_csv(filepath, index=False)
                    
                    holdout_stats[month] = len(month_data)
                    logger.info(f"  {filename}: {len(month_data):,} rows")
                else:
                    logger.warning(f"No data found for {holdout_year}-{month:02d}")
            
            return holdout_stats
            
        except Exception as e:
            logger.error(f"Failed to create holdout splits: {str(e)}")
            raise


class DataPipeline:
    """Main data pipeline orchestrator."""
    
    def __init__(self, config: Union[str, DataPipelineConfig, None] = None):
        """
        Initialize data pipeline.
        
        Args:
            config: Configuration file path, config object, or None for defaults
        """
        if isinstance(config, str):
            self.config = DataPipelineConfig(config)
        elif isinstance(config, DataPipelineConfig):
            self.config = config
        else:
            self.config = DataPipelineConfig()
        
        self.downloader = KaggleDataDownloader()
        self.processor = DataProcessor(self.config)
        self.pipeline_metadata = {
            'start_time': datetime.now().isoformat(),
            'config': self.config.config
        }
    
    def run_full_pipeline(self, download_data: bool = True, save_metadata: bool = True) -> Dict:
        """
        Run the complete data pipeline.
        
        Args:
            download_data: Whether to attempt data download
            save_metadata: Whether to save pipeline metadata
            
        Returns:
            Pipeline execution results
        """
        logger.info("Starting data pipeline execution...")
        
        try:
            # Setup paths
            raw_data_dir = Path(self.config.get('paths.raw_data_dir'))
            processed_data_dir = Path(self.config.get('paths.processed_data_dir'))
            holdout_dir = Path(self.config.get('paths.holdout_dir'))
            
            # Download data (if requested and available)
            if download_data:
                dataset_name = self.config.get('dataset.name')
                file_name = self.config.get('dataset.file_name')
                
                download_success = self.downloader.download_file(
                    file_name, raw_data_dir, dataset_name
                )
                
                if not download_success:
                    logger.warning("Data download failed, proceeding with existing files")
            
            # Load and clean data
            raw_file_path = raw_data_dir / self.config.get('dataset.file_name')
            
            if not raw_file_path.exists():
                raise FileNotFoundError(f"Raw data file not found: {raw_file_path}")
            
            cleaned_df = self.processor.load_and_clean_data(raw_file_path)
            
            # Create train/val/test splits
            train_df, val_df, test_df = self.processor.create_train_val_test_split(cleaned_df)
            
            # Save main datasets
            processed_data_dir.mkdir(parents=True, exist_ok=True)
            
            train_df.to_csv(processed_data_dir / 'train.csv', index=False)
            val_df.to_csv(processed_data_dir / 'val.csv', index=False)
            test_df.to_csv(processed_data_dir / 'test.csv', index=False)
            
            logger.info("Main datasets saved successfully")
            
            # Create holdout splits
            holdout_stats = self.processor.create_holdout_splits(cleaned_df, holdout_dir)
            
            # Compile results
            results = {
                'success': True,
                'execution_time': datetime.now().isoformat(),
                'datasets': {
                    'train': len(train_df),
                    'validation': len(val_df),
                    'test': len(test_df),
                    'holdout_months': holdout_stats
                },
                'total_samples': len(cleaned_df),
                'data_paths': {
                    'train': str(processed_data_dir / 'train.csv'),
                    'validation': str(processed_data_dir / 'val.csv'),
                    'test': str(processed_data_dir / 'test.csv'),
                    'holdout_dir': str(holdout_dir)
                }
            }
            
            # Update pipeline metadata
            self.pipeline_metadata.update({
                'end_time': datetime.now().isoformat(),
                'results': results
            })
            
            # Save metadata
            if save_metadata:
                metadata_path = processed_data_dir / 'pipeline_metadata.json'
                with open(metadata_path, 'w') as f:
                    json.dump(self.pipeline_metadata, f, indent=2)
                logger.info(f"Pipeline metadata saved to {metadata_path}")
            
            logger.info("Data pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Data pipeline failed: {str(e)}")
            self.pipeline_metadata.update({
                'end_time': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            })
            raise


def main():
    """Main function to run the data pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run abuse detection data pipeline')
    parser.add_argument('--config', type=str, default='config/data_pipeline_config.json')
    parser.add_argument('--no-download', action='store_true', default=False)
    parser.add_argument('--output-dir', type=str, default='data/processed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DataPipelineConfig(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config.config['paths']['processed_data_dir'] = args.output_dir
        config.config['paths']['holdout_dir'] = f"{args.output_dir}/holdout"
    
    # Run pipeline
    pipeline = DataPipeline(config)
    results = pipeline.run_full_pipeline(download_data=not args.no_download)
    
    # Print summary
    print("\n" + "=" * 50)
    print("DATA PIPELINE SUMMARY")
    print("=" * 50)
    print("Pipeline completed successfully!")
    print(f"Total samples processed: {results['total_samples']:,}")

    print("Datasets created:")
    for dataset, count in results['datasets'].items():
        if dataset != 'holdout_months':
            print(f"  {dataset.title()}: {count:,} samples")

    holdout_months = results['datasets'].get('holdout_months')
    if holdout_months:
        print(f"  Holdout months: {len(holdout_months)} files")

    print("\nOutput files saved to:")
    for path_name, path_value in results['data_paths'].items():
        if path_name != 'holdout_dir':
            print(f"  {path_name}: {path_value}")
    print(f"  Holdout files: {results['data_paths']['holdout_dir']}")


if __name__ == '__main__':
    main()
