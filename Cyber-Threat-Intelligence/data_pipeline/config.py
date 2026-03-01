"""
Configuration module for UNSW-NB15 dataset.

This module defines the schema, feature specifications, and configuration
parameters for the UNSW-NB15 network intrusion detection dataset.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class AttackCategory(Enum):
    """Enumeration of attack categories in UNSW-NB15 dataset."""
    
    NORMAL = "Normal"
    FUZZERS = "Fuzzers"
    ANALYSIS = "Analysis"
    BACKDOORS = "Backdoors"
    DOS = "DoS"
    EXPLOITS = "Exploits"
    GENERIC = "Generic"
    RECONNAISSANCE = "Reconnaissance"
    SHELLCODE = "Shellcode"
    WORMS = "Worms"


@dataclass
class DatasetConfig:
    """
    Configuration class for UNSW-NB15 dataset parameters.
    
    This class encapsulates all configuration parameters for loading,
    preprocessing, and batching the UNSW-NB15 dataset.
    
    Attributes:
        numerical_features: List of numerical feature column names.
        categorical_features: List of categorical feature column names.
        binary_features: List of binary feature column names.
        label_column: Name of the binary label column (0=normal, 1=attack).
        attack_category_column: Name of the multi-class attack category column.
        id_columns: Columns to exclude from features (identifiers).
        default_batch_size: Default batch size for data iteration.
        test_split_ratio: Ratio of data to use for testing.
        random_seed: Random seed for reproducibility.
        normalization_method: Method for numerical normalization ('minmax', 'standard', 'robust').
        handle_outliers: Whether to handle outliers during preprocessing.
        outlier_threshold: Z-score threshold for outlier detection.
    
    Example:
        >>> config = DatasetConfig()
        >>> print(config.numerical_features[:3])
        ['dur', 'spkts', 'dpkts']
    """
    
    # UNSW-NB15 Numerical Features (continuous/discrete)
    numerical_features: List[str] = field(default_factory=lambda: [
        "dur",          # Duration of the connection
        "spkts",        # Source to destination packet count
        "dpkts",        # Destination to source packet count
        "sbytes",       # Source to destination bytes
        "dbytes",       # Destination to source bytes
        "rate",         # Connection rate
        "sttl",         # Source to destination time to live
        "dttl",         # Destination to source time to live
        "sload",        # Source bits per second
        "dload",        # Destination bits per second
        "sloss",        # Source packets retransmitted or dropped
        "dloss",        # Destination packets retransmitted or dropped
        "sinpkt",       # Source inter-packet arrival time (ms)
        "dinpkt",       # Destination inter-packet arrival time (ms)
        "sjit",         # Source jitter (ms)
        "djit",         # Destination jitter (ms)
        "swin",         # Source TCP window advertisement
        "stcpb",        # Source TCP base sequence number
        "dtcpb",        # Destination TCP base sequence number
        "dwin",         # Destination TCP window advertisement
        "tcprtt",       # TCP connection setup round-trip time
        "synack",       # Time between SYN and SYN-ACK
        "ackdat",       # Time between SYN-ACK and ACK
        "smean",        # Mean of packet size transmitted by source
        "dmean",        # Mean of packet size transmitted by destination
        "trans_depth",  # Connection depth (HTTP)
        "response_body_len",  # Content size of HTTP response
        "ct_srv_src",   # Connections with same service and source address
        "ct_state_ttl", # Connections with same state and TTL
        "ct_dst_ltm",   # Connections with same destination address
        "ct_src_dport_ltm",   # Connections with same source and dest port
        "ct_dst_sport_ltm",   # Connections with same dest and source port
        "ct_dst_src_ltm",     # Connections with same source and dest
        "ct_ftp_cmd",   # Number of FTP commands
        "ct_flw_http_mthd",   # Number of HTTP methods
        "ct_src_ltm",   # Connections with same source address
        "ct_srv_dst",   # Connections with same service and destination
    ])
    
    # UNSW-NB15 Categorical Features
    categorical_features: List[str] = field(default_factory=lambda: [
        "proto",        # Protocol type (tcp, udp, etc.)
        "service",      # Network service (http, ftp, ssh, etc.)
        "state",        # Connection state (FIN, CON, etc.)
    ])
    
    # Binary Features
    binary_features: List[str] = field(default_factory=lambda: [
        "is_ftp_login", # 1 if FTP session accessed with user/password
        "is_sm_ips_ports",  # 1 if source and dest IPs/ports are equal
    ])
    
    # Target/Label Columns
    label_column: str = "label"
    attack_category_column: str = "attack_cat"
    
    # Identifier columns (excluded from features)
    id_columns: List[str] = field(default_factory=lambda: [
        "id",
        "srcip",
        "sport",
        "dstip",
        "dsport",
    ])
    
    # Batch and split configuration
    default_batch_size: int = 64
    test_split_ratio: float = 0.2
    validation_split_ratio: float = 0.1
    random_seed: int = 42
    
    # Preprocessing configuration
    normalization_method: str = "minmax"  # 'minmax', 'standard', 'robust'
    handle_outliers: bool = True
    outlier_threshold: float = 3.0  # Z-score threshold
    
    # Missing value handling
    numerical_fill_strategy: str = "median"  # 'mean', 'median', 'zero'
    categorical_fill_strategy: str = "mode"  # 'mode', 'unknown'
    
    @property
    def all_feature_columns(self) -> List[str]:
        """
        Returns all feature column names.
        
        Returns:
            List of all feature column names (numerical + categorical + binary).
        """
        return self.numerical_features + self.categorical_features + self.binary_features
    
    @property
    def columns_to_drop(self) -> List[str]:
        """
        Returns columns to exclude from the feature set.
        
        Returns:
            List of column names to drop during preprocessing.
        """
        return self.id_columns
    
    def get_attack_categories(self) -> List[str]:
        """
        Returns list of attack category names.
        
        Returns:
            List of attack category string values.
        """
        return [cat.value for cat in AttackCategory]
    
    def validate(self) -> bool:
        """
        Validates configuration parameters.
        
        Returns:
            True if configuration is valid.
            
        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if not 0.0 < self.test_split_ratio < 1.0:
            raise ValueError(f"test_split_ratio must be between 0 and 1, got {self.test_split_ratio}")
        
        if not 0.0 < self.validation_split_ratio < 1.0:
            raise ValueError(f"validation_split_ratio must be between 0 and 1, got {self.validation_split_ratio}")
        
        if self.test_split_ratio + self.validation_split_ratio >= 1.0:
            raise ValueError("Sum of test and validation ratios must be less than 1.0")
        
        if self.default_batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.default_batch_size}")
        
        valid_norm_methods = {"minmax", "standard", "robust"}
        if self.normalization_method not in valid_norm_methods:
            raise ValueError(f"normalization_method must be one of {valid_norm_methods}")
        
        valid_num_fill = {"mean", "median", "zero"}
        if self.numerical_fill_strategy not in valid_num_fill:
            raise ValueError(f"numerical_fill_strategy must be one of {valid_num_fill}")
        
        valid_cat_fill = {"mode", "unknown"}
        if self.categorical_fill_strategy not in valid_cat_fill:
            raise ValueError(f"categorical_fill_strategy must be one of {valid_cat_fill}")
        
        return True
