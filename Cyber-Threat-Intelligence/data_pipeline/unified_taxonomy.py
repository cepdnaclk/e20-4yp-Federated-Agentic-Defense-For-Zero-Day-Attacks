"""
Unified Attack Taxonomy for Multi-Dataset Training.

This module provides a unified attack category mapping that works across
multiple intrusion detection datasets (UNSW-NB15, CIC-IDS2017, etc.).

The unified taxonomy enables:
- Cross-dataset training and evaluation
- Consistent attack semantics across different sources
- MITRE ATT&CK framework alignment

Usage:
    >>> from data_pipeline.unified_taxonomy import UnifiedTaxonomy
    >>> taxonomy = UnifiedTaxonomy()
    >>> unified_label = taxonomy.map_unsw_nb15("Exploits")  # -> "Exploits"
    >>> unified_label = taxonomy.map_cic_ids2017("DDoS")     # -> "DoS/DDoS"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class UnifiedCategory(Enum):
    """Unified attack categories across datasets."""
    NORMAL = "Normal"
    DOS_DDOS = "DoS/DDoS"
    RECONNAISSANCE = "Reconnaissance"
    EXPLOITS = "Exploits"
    BRUTE_FORCE = "Brute_Force"
    MALWARE = "Malware"
    ANALYSIS = "Analysis"


@dataclass
class CategoryInfo:
    """Information about a unified attack category."""
    category: UnifiedCategory
    description: str
    mitre_tactics: List[str]
    mitre_techniques: List[str]
    severity: int  # 1=Low, 2=Medium, 3=High, 4=Critical


# Unified category metadata with MITRE ATT&CK mapping
CATEGORY_INFO: Dict[UnifiedCategory, CategoryInfo] = {
    UnifiedCategory.NORMAL: CategoryInfo(
        category=UnifiedCategory.NORMAL,
        description="Benign/legitimate network traffic",
        mitre_tactics=[],
        mitre_techniques=[],
        severity=0,
    ),
    UnifiedCategory.DOS_DDOS: CategoryInfo(
        category=UnifiedCategory.DOS_DDOS,
        description="Denial of Service and Distributed DoS attacks",
        mitre_tactics=["Impact"],
        mitre_techniques=["T1498", "T1499"],  # Network DoS, Endpoint DoS
        severity=4,
    ),
    UnifiedCategory.RECONNAISSANCE: CategoryInfo(
        category=UnifiedCategory.RECONNAISSANCE,
        description="Network scanning, port scanning, service enumeration",
        mitre_tactics=["Reconnaissance", "Discovery"],
        mitre_techniques=["T1046", "T1595", "T1592"],  # Network Service Scan
        severity=2,
    ),
    UnifiedCategory.EXPLOITS: CategoryInfo(
        category=UnifiedCategory.EXPLOITS,
        description="Exploitation of vulnerabilities (web, system, application)",
        mitre_tactics=["Initial Access", "Execution"],
        mitre_techniques=["T1190", "T1059", "T1203"],  # Exploit Public-Facing App
        severity=4,
    ),
    UnifiedCategory.BRUTE_FORCE: CategoryInfo(
        category=UnifiedCategory.BRUTE_FORCE,
        description="Password guessing, credential stuffing, fuzzing",
        mitre_tactics=["Credential Access"],
        mitre_techniques=["T1110", "T1078"],  # Brute Force, Valid Accounts
        severity=3,
    ),
    UnifiedCategory.MALWARE: CategoryInfo(
        category=UnifiedCategory.MALWARE,
        description="Backdoors, worms, bots, trojans, infiltration",
        mitre_tactics=["Persistence", "Command and Control"],
        mitre_techniques=["T1059", "T1071", "T1105"],  # C2, Ingress Tool Transfer
        severity=4,
    ),
    UnifiedCategory.ANALYSIS: CategoryInfo(
        category=UnifiedCategory.ANALYSIS,
        description="Traffic analysis, protocol analysis attacks",
        mitre_tactics=["Collection", "Discovery"],
        mitre_techniques=["T1040", "T1557"],  # Network Sniffing, MITM
        severity=2,
    ),
}


@dataclass
class UnifiedTaxonomy:
    """
    Unified attack taxonomy mapping for multi-dataset training.
    
    Maps attack labels from various datasets to a common taxonomy
    that enables cross-dataset training and consistent evaluation.
    
    Supported Datasets:
        - UNSW-NB15
        - CIC-IDS2017
    
    Attributes:
        num_classes: Number of unified categories (including Normal).
        category_names: List of category names in consistent order.
    
    Example:
        >>> taxonomy = UnifiedTaxonomy()
        >>> taxonomy.map_unsw_nb15("DoS")
        'DoS/DDoS'
        >>> taxonomy.map_cic_ids2017("DDoS")
        'DoS/DDoS'
        >>> taxonomy.get_category_id("DoS/DDoS")
        1
    """
    
    # UNSW-NB15 -> Unified mapping
    _unsw_nb15_mapping: Dict[str, str] = field(default_factory=lambda: {
        # Normal
        "Normal": "Normal",
        "normal": "Normal",
        " Normal": "Normal",
        
        # DoS/DDoS
        "DoS": "DoS/DDoS",
        "dos": "DoS/DDoS",
        " DoS": "DoS/DDoS",
        
        # Reconnaissance
        "Reconnaissance": "Reconnaissance",
        "reconnaissance": "Reconnaissance",
        " Reconnaissance": "Reconnaissance",
        
        # Exploits (including Shellcode - shell execution exploits)
        "Exploits": "Exploits",
        "exploits": "Exploits",
        " Exploits": "Exploits",
        "Shellcode": "Exploits",
        "shellcode": "Exploits",
        " Shellcode": "Exploits",
        
        # Brute Force (Fuzzers are input fuzzing attacks)
        "Fuzzers": "Brute_Force",
        "fuzzers": "Brute_Force",
        " Fuzzers": "Brute_Force",
        
        # Malware (Backdoors, Worms, Generic malicious)
        "Backdoor": "Malware",
        "backdoor": "Malware",
        " Backdoor": "Malware",
        "Backdoors": "Malware",
        "Worms": "Malware",
        "worms": "Malware",
        " Worms": "Malware",
        "Generic": "Malware",
        "generic": "Malware",
        " Generic": "Malware",
        
        # Analysis
        "Analysis": "Analysis",
        "analysis": "Analysis",
        " Analysis": "Analysis",
    })
    
    # CIC-IDS2017 -> Unified mapping
    _cic_ids2017_mapping: Dict[str, str] = field(default_factory=lambda: {
        # Normal
        "BENIGN": "Normal",
        "Benign": "Normal",
        "benign": "Normal",
        
        # DoS/DDoS
        "DDoS": "DoS/DDoS",
        "DoS slowloris": "DoS/DDoS",
        "DoS Slowhttptest": "DoS/DDoS",
        "DoS Hulk": "DoS/DDoS",
        "DoS GoldenEye": "DoS/DDoS",
        
        # Reconnaissance
        "PortScan": "Reconnaissance",
        
        # Exploits (Web attacks, Heartbleed)
        "Web Attack � Brute Force": "Exploits",
        "Web Attack � XSS": "Exploits",
        "Web Attack � Sql Injection": "Exploits",
        "Web Attack \x96 Brute Force": "Exploits",
        "Web Attack \x96 XSS": "Exploits",
        "Web Attack \x96 Sql Injection": "Exploits",
        "Heartbleed": "Exploits",
        
        # Brute Force (SSH/FTP Patator are brute force tools)
        "FTP-Patator": "Brute_Force",
        "SSH-Patator": "Brute_Force",
        
        # Malware
        "Bot": "Malware",
        "Infiltration": "Malware",
    })
    
    # Ordered category names for consistent label encoding
    _category_order: List[str] = field(default_factory=lambda: [
        "Normal",
        "DoS/DDoS",
        "Reconnaissance", 
        "Exploits",
        "Brute_Force",
        "Malware",
        "Analysis",
    ])
    
    def __post_init__(self):
        """Initialize reverse mappings and ID lookups."""
        self._category_to_id = {name: idx for idx, name in enumerate(self._category_order)}
        self._id_to_category = {idx: name for idx, name in enumerate(self._category_order)}
        
        # Add fallback mappings for encoding issues in CIC-IDS2017
        self._add_encoding_fallbacks()
    
    def _add_encoding_fallbacks(self):
        """Add fallback mappings for common encoding issues."""
        # Web attack variants with different encodings
        web_attack_patterns = [
            "Web Attack",
            "Brute Force",
            "XSS",
            "Sql Injection",
            "SQL Injection",
        ]
        
        for key in list(self._cic_ids2017_mapping.keys()):
            if "Web Attack" in key:
                # Create variants without special characters
                clean_key = key.replace("�", "-").replace("\x96", "-")
                self._cic_ids2017_mapping[clean_key] = "Exploits"
    
    @property
    def num_classes(self) -> int:
        """Number of unified attack categories."""
        return len(self._category_order)
    
    @property
    def category_names(self) -> List[str]:
        """Ordered list of category names."""
        return self._category_order.copy()
    
    @property
    def attack_categories(self) -> List[str]:
        """List of attack categories (excluding Normal)."""
        return [c for c in self._category_order if c != "Normal"]
    
    def map_unsw_nb15(self, label: str) -> str:
        """
        Map UNSW-NB15 attack label to unified category.
        
        Args:
            label: Original UNSW-NB15 attack_cat value.
        
        Returns:
            Unified category name.
        
        Raises:
            ValueError: If label is not recognized.
        """
        label_clean = str(label).strip()
        
        if label_clean in self._unsw_nb15_mapping:
            return self._unsw_nb15_mapping[label_clean]
        
        # Try case variations
        for key, value in self._unsw_nb15_mapping.items():
            if key.lower().strip() == label_clean.lower():
                return value
        
        logger.warning(f"Unknown UNSW-NB15 label: '{label}', defaulting to 'Malware'")
        return "Malware"
    
    def map_cic_ids2017(self, label: str) -> str:
        """
        Map CIC-IDS2017 attack label to unified category.
        
        Args:
            label: Original CIC-IDS2017 Label value.
        
        Returns:
            Unified category name.
        
        Raises:
            ValueError: If label is not recognized.
        """
        label_clean = str(label).strip()
        
        if label_clean in self._cic_ids2017_mapping:
            return self._cic_ids2017_mapping[label_clean]
        
        # Handle encoding issues - check if any known pattern is in the label
        label_lower = label_clean.lower()
        
        if "benign" in label_lower:
            return "Normal"
        if "ddos" in label_lower or "dos" in label_lower:
            return "DoS/DDoS"
        if "portscan" in label_lower or "port scan" in label_lower:
            return "Reconnaissance"
        if "web attack" in label_lower or "sql" in label_lower or "xss" in label_lower:
            return "Exploits"
        if "heartbleed" in label_lower:
            return "Exploits"
        if "patator" in label_lower or "brute" in label_lower:
            return "Brute_Force"
        if "bot" in label_lower or "infiltration" in label_lower:
            return "Malware"
        
        logger.warning(f"Unknown CIC-IDS2017 label: '{label}', defaulting to 'Malware'")
        return "Malware"
    
    def get_category_id(self, category: str) -> int:
        """
        Get numeric ID for a unified category.
        
        Args:
            category: Unified category name.
        
        Returns:
            Integer ID (0 to num_classes-1).
        """
        if category not in self._category_to_id:
            raise ValueError(f"Unknown category: {category}")
        return self._category_to_id[category]
    
    def get_category_name(self, category_id: int) -> str:
        """
        Get category name from numeric ID.
        
        Args:
            category_id: Integer ID.
        
        Returns:
            Category name string.
        """
        if category_id not in self._id_to_category:
            raise ValueError(f"Invalid category ID: {category_id}")
        return self._id_to_category[category_id]
    
    def get_category_info(self, category: str) -> CategoryInfo:
        """
        Get detailed information about a category.
        
        Args:
            category: Unified category name.
        
        Returns:
            CategoryInfo with description, MITRE mapping, severity.
        """
        unified_cat = UnifiedCategory(category)
        return CATEGORY_INFO[unified_cat]
    
    def get_binary_label(self, category: str) -> int:
        """
        Convert unified category to binary (Normal=0, Attack=1).
        
        Args:
            category: Unified category name.
        
        Returns:
            0 for Normal, 1 for any attack.
        """
        return 0 if category == "Normal" else 1
    
    def get_severity(self, category: str) -> int:
        """
        Get severity level for a category.
        
        Args:
            category: Unified category name.
        
        Returns:
            Severity level (0=Normal, 1=Low, 2=Medium, 3=High, 4=Critical).
        """
        try:
            return self.get_category_info(category).severity
        except (ValueError, KeyError):
            return 2  # Default to medium
    
    def get_mitre_techniques(self, category: str) -> List[str]:
        """
        Get MITRE ATT&CK technique IDs for a category.
        
        Args:
            category: Unified category name.
        
        Returns:
            List of MITRE technique IDs (e.g., ["T1498", "T1499"]).
        """
        try:
            return self.get_category_info(category).mitre_techniques
        except (ValueError, KeyError):
            return []
    
    def print_mapping_summary(self):
        """Print a summary of the unified taxonomy mappings."""
        print("=" * 70)
        print("UNIFIED ATTACK TAXONOMY")
        print("=" * 70)
        
        for category in self._category_order:
            print(f"\n{category}:")
            print("-" * 40)
            
            # UNSW-NB15 sources
            unsw_sources = [k for k, v in self._unsw_nb15_mapping.items() 
                          if v == category and not k.startswith(" ")]
            if unsw_sources:
                print(f"  UNSW-NB15: {', '.join(sorted(set(unsw_sources)))}")
            
            # CIC-IDS2017 sources
            cic_sources = [k for k, v in self._cic_ids2017_mapping.items() 
                         if v == category]
            if cic_sources:
                # Clean up for display
                clean_sources = []
                for s in cic_sources:
                    if any(x in s for x in ["�", "\x96"]):
                        continue
                    clean_sources.append(s)
                if clean_sources:
                    print(f"  CIC-IDS2017: {', '.join(sorted(set(clean_sources)))}")
            
            # MITRE mapping
            try:
                info = self.get_category_info(category)
                if info.mitre_techniques:
                    print(f"  MITRE: {', '.join(info.mitre_techniques)}")
                print(f"  Severity: {info.severity}")
            except (ValueError, KeyError):
                pass


# Singleton instance for convenience
_default_taxonomy = None

def get_taxonomy() -> UnifiedTaxonomy:
    """Get the default UnifiedTaxonomy instance."""
    global _default_taxonomy
    if _default_taxonomy is None:
        _default_taxonomy = UnifiedTaxonomy()
    return _default_taxonomy


if __name__ == "__main__":
    # Demo
    taxonomy = UnifiedTaxonomy()
    taxonomy.print_mapping_summary()
    
    print("\n" + "=" * 70)
    print("EXAMPLES")
    print("=" * 70)
    
    # Test mappings
    test_unsw = ["DoS", "Exploits", "Generic", "Backdoor", "Fuzzers", "Normal"]
    test_cic = ["BENIGN", "DDoS", "PortScan", "Bot", "FTP-Patator"]
    
    print("\nUNSW-NB15 mappings:")
    for label in test_unsw:
        unified = taxonomy.map_unsw_nb15(label)
        cat_id = taxonomy.get_category_id(unified)
        print(f"  {label:20} -> {unified:15} (ID: {cat_id})")
    
    print("\nCIC-IDS2017 mappings:")
    for label in test_cic:
        unified = taxonomy.map_cic_ids2017(label)
        cat_id = taxonomy.get_category_id(unified)
        print(f"  {label:20} -> {unified:15} (ID: {cat_id})")
