"""
Dataset Loader for UNSW_NB15 Dataset
Provides utilities to load and stream network traffic data for federated learning simulation.
"""

import pandas as pd
from typing import Generator, Tuple, List, Optional
from pathlib import Path


class UNSW_NB15_Loader:
    """
    Helper class to load and process the UNSW_NB15 training dataset.
    Simulates live network traffic by yielding packets one at a time.
    """
    
    # Feature columns to use as attack signature vector
    # These are numerical features that represent network flow characteristics
    FEATURE_COLUMNS = [
        'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
        'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
        'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
        'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
        'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
        'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
        'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
    ]
    
    def __init__(self, csv_path: str):
        """
        Initialize the dataset loader.
        
        Args:
            csv_path: Path to the UNSW_NB15_training-set.csv file
        """
        self.csv_path = Path(csv_path)
        self.dataframe: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        
    def load_dataset(self) -> bool:
        """
        Load the UNSW_NB15 dataset from CSV.
        
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            if not self.csv_path.exists():
                print(f"❌ Dataset not found at: {self.csv_path}")
                print("   Please ensure 'UNSW_NB15_training-set.csv' is in the data directory.")
                return False
            
            print(f"📂 Loading dataset from: {self.csv_path}")
            self.dataframe = pd.read_csv(self.csv_path, low_memory=False)
            
            # Determine available feature columns
            available_features = [col for col in self.FEATURE_COLUMNS 
                                  if col in self.dataframe.columns]
            
            if not available_features:
                # Fallback: use all numeric columns except label columns
                numeric_cols = self.dataframe.select_dtypes(include=['float64', 'int64']).columns
                exclude_cols = {'id', 'label', 'Label'}
                available_features = [col for col in numeric_cols if col not in exclude_cols]
            
            self.feature_columns = available_features
            
            print(f"✅ Dataset loaded successfully!")
            print(f"   Total records: {len(self.dataframe)}")
            print(f"   Features used: {len(self.feature_columns)}")
            print(f"   Columns: {self.dataframe.columns.tolist()}")
            
            # Show attack category distribution
            if 'attack_cat' in self.dataframe.columns:
                print(f"   Attack categories: {self.dataframe['attack_cat'].unique().tolist()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def yield_packet(
        self, 
        start_idx: int = 0, 
        end_idx: Optional[int] = None
    ) -> Generator[Tuple[List[float], str], None, None]:
        """
        Generator function that simulates live network traffic.
        Yields one packet (row) at a time.
        
        Args:
            start_idx: Starting row index (inclusive)
            end_idx: Ending row index (exclusive), None for all remaining rows
            
        Yields:
            Tuple of (feature_vector, attack_category)
            - feature_vector: List of float values representing packet features
            - attack_category: String label ('Normal' or attack type like 'Fuzzers')
        """
        if self.dataframe is None:
            if not self.load_dataset():
                return
        
        if end_idx is None:
            end_idx = len(self.dataframe)
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(self.dataframe))
        
        # Determine label column
        label_col = 'attack_cat' if 'attack_cat' in self.dataframe.columns else 'Label'
        
        print(f"🔄 Streaming packets from index {start_idx} to {end_idx}")
        
        for idx in range(start_idx, end_idx):
            row = self.dataframe.iloc[idx]
            
            # Extract feature vector (handle missing values)
            features = []
            for col in self.feature_columns:
                val = row.get(col, 0)
                # Convert to float, handle NaN
                try:
                    features.append(float(val) if pd.notna(val) else 0.0)
                except (ValueError, TypeError):
                    features.append(0.0)
            
            # Get attack category label
            label = row.get(label_col, 'Normal')
            if pd.isna(label) or label == '' or label == ' ':
                label = 'Normal'
            
            yield features, str(label).strip()
    
    def get_packet_batch(
        self, 
        start_idx: int, 
        batch_size: int
    ) -> List[Tuple[List[float], str]]:
        """
        Get a batch of packets at once.
        
        Args:
            start_idx: Starting row index
            batch_size: Number of packets to retrieve
            
        Returns:
            List of (feature_vector, attack_category) tuples
        """
        return list(self.yield_packet(start_idx, start_idx + batch_size))
    
    def get_statistics(self) -> dict:
        """Get dataset statistics"""
        if self.dataframe is None:
            return {"error": "Dataset not loaded"}
        
        label_col = 'attack_cat' if 'attack_cat' in self.dataframe.columns else 'Label'
        
        return {
            "total_records": len(self.dataframe),
            "feature_count": len(self.feature_columns),
            "label_distribution": self.dataframe[label_col].value_counts().to_dict(),
            "feature_columns": self.feature_columns
        }


# ============================================================================
# Utility Functions
# ============================================================================

def create_loader(data_dir: str = "data", use_testing: bool = True) -> UNSW_NB15_Loader:
    """
    Factory function to create a dataset loader with default path.
    
    Args:
        data_dir: Directory containing the dataset
        use_testing: If True, use testing dataset; otherwise use training dataset
        
    Returns:
        Configured UNSW_NB15_Loader instance
    """
    filename = "UNSW_NB15_testing-set.csv" if use_testing else "UNSW_NB15_training-set.csv"
    csv_path = Path(data_dir) / filename
    return UNSW_NB15_Loader(str(csv_path))


if __name__ == "__main__":
    # Test the loader
    loader = create_loader()
    
    if loader.load_dataset():
        print("\n--- Testing packet generator ---")
        count = 0
        for features, label in loader.yield_packet(0, 5):
            print(f"Packet {count}: Label={label}, Features={len(features)} dims")
            count += 1
        
        print("\n--- Dataset Statistics ---")
        stats = loader.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
