"""
Build Optimized Knowledge Base from UNSW-NB15 Training Dataset

This script creates:
1. unsw_knowledge_base.json - Statistical attack signatures
2. unsw_vector_docs.jsonl - Pre-formatted docs for vector embedding
3. enhanced_kb_cache.json - Threat intelligence cache

Covers ALL attack categories with discriminative feature profiles.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# Configuration
TRAINING_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "UNSW_NB15" / "UNSW_NB15_training-set.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

# Feature groups for signature generation
PACKET_FEATURES = ['spkts', 'dpkts', 'sbytes', 'dbytes']
TIMING_FEATURES = ['dur', 'rate', 'sttl', 'dttl', 'sinpkt', 'dinpkt', 'sjit', 'djit']
FLOW_FEATURES = ['sload', 'dload', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 
                 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
TCP_FEATURES = ['tcprtt', 'synack', 'ackdat', 'swin', 'dwin']
CATEGORICAL_FEATURES = ['proto', 'service', 'state']
CONTENT_FEATURES = ['trans_depth', 'response_body_len', 'ct_flw_http_mthd', 'is_sm_ips_ports']

# Max samples per attack category for vector docs (balance representation)
MAX_SAMPLES_PER_CATEGORY = 300
MAX_SIGNATURES_PER_CATEGORY = 50


def load_training_data() -> pd.DataFrame:
    """Load and preprocess training data."""
    print(f"[INFO] Loading training data from {TRAINING_DATA_PATH}...")
    df = pd.read_csv(TRAINING_DATA_PATH, low_memory=False)
    
    # Normalize attack_cat
    df['attack_cat'] = df['attack_cat'].fillna('Normal').str.strip()
    df['attack_cat'] = df['attack_cat'].replace('', 'Normal')
    
    print(f"[INFO] Loaded {len(df)} records")
    return df


def compute_feature_stats(group: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    """Compute statistical summary for numeric features."""
    stats = {}
    for feat in features:
        if feat in group.columns:
            values = pd.to_numeric(group[feat], errors='coerce').dropna()
            if len(values) > 0:
                stats[f"{feat}_mean"] = float(np.round(values.mean(), 4))
                stats[f"{feat}_std"] = float(np.round(values.std(), 4))
                stats[f"{feat}_min"] = float(np.round(values.min(), 4))
                stats[f"{feat}_max"] = float(np.round(values.max(), 4))
                stats[f"{feat}_median"] = float(np.round(values.median(), 4))
                # Percentiles for outlier detection
                stats[f"{feat}_p25"] = float(np.round(values.quantile(0.25), 4))
                stats[f"{feat}_p75"] = float(np.round(values.quantile(0.75), 4))
                stats[f"{feat}_p95"] = float(np.round(values.quantile(0.95), 4))
    return stats


def get_top_categorical_values(group: pd.DataFrame, feature: str, top_n: int = 5) -> List[Dict]:
    """Get most common categorical values with their frequencies."""
    if feature not in group.columns:
        return []
    
    value_counts = group[feature].fillna('-').astype(str).value_counts()
    total = len(group)
    
    result = []
    for val, count in value_counts.head(top_n).items():
        result.append({
            "value": val,
            "count": int(count),
            "frequency": float(np.round(count / total, 4))
        })
    return result


def generate_attack_description(attack_cat: str, stats: Dict, categorical_profile: Dict) -> str:
    """Generate human-readable attack description."""
    descriptions = {
        "Normal": "Benign network traffic with typical communication patterns",
        "Generic": "Generic malicious activity with anomalous traffic patterns",
        "Exploits": "Exploitation attempts targeting vulnerabilities in services or protocols",
        "Fuzzers": "Fuzzing attacks sending random/malformed data to discover vulnerabilities",
        "DoS": "Denial of Service attacks aimed at disrupting service availability",
        "Reconnaissance": "Network scanning and probing to discover hosts, services, and vulnerabilities",
        "Analysis": "Advanced analysis and profiling attacks gathering system information",
        "Backdoor": "Backdoor installation or communication for persistent unauthorized access",
        "Shellcode": "Shellcode injection attacks attempting to execute arbitrary code",
        "Worms": "Self-propagating malware spreading across network systems"
    }
    
    base_desc = descriptions.get(attack_cat, f"Unknown attack category: {attack_cat}")
    
    # Add characteristic features
    proto_info = categorical_profile.get('proto', [])
    service_info = categorical_profile.get('service', [])
    state_info = categorical_profile.get('state', [])
    
    details = []
    if proto_info:
        top_protos = [p['value'] for p in proto_info[:3] if p['frequency'] > 0.1]
        if top_protos:
            details.append(f"Common protocols: {', '.join(top_protos)}")
    
    if service_info:
        top_services = [s['value'] for s in service_info[:3] if s['frequency'] > 0.05 and s['value'] != '-']
        if top_services:
            details.append(f"Target services: {', '.join(top_services)}")
    
    if state_info:
        top_states = [s['value'] for s in state_info[:3] if s['frequency'] > 0.1]
        if top_states:
            details.append(f"Connection states: {', '.join(top_states)}")
    
    if details:
        return f"{base_desc}. {'; '.join(details)}."
    return base_desc


def create_signature_key(row: pd.Series) -> str:
    """Create a unique key for signature grouping."""
    proto = str(row.get('proto', '-')).strip() or '-'
    service = str(row.get('service', '-')).strip() or '-'
    state = str(row.get('state', '-')).strip() or '-'
    return f"{proto}|{service}|{state}"


def build_attack_signatures(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Build detailed attack signatures per category and protocol/service/state combination."""
    print("[INFO] Building attack signatures...")
    
    attack_signatures = defaultdict(list)
    attack_categories = df['attack_cat'].unique()
    
    for attack_cat in attack_categories:
        print(f"  Processing: {attack_cat}")
        cat_df = df[df['attack_cat'] == attack_cat]
        
        # Group by proto/service/state for fine-grained signatures
        cat_df['sig_key'] = cat_df.apply(create_signature_key, axis=1)
        
        sig_groups = cat_df.groupby('sig_key')
        signatures_for_cat = []
        
        for sig_key, group in sig_groups:
            if len(group) < 3:  # Skip very rare combinations
                continue
            
            parts = sig_key.split('|')
            proto, service, state = parts[0], parts[1], parts[2]
            
            # Compute statistics
            packet_stats = compute_feature_stats(group, PACKET_FEATURES)
            timing_stats = compute_feature_stats(group, TIMING_FEATURES)
            flow_stats = compute_feature_stats(group, FLOW_FEATURES)
            tcp_stats = compute_feature_stats(group, TCP_FEATURES)
            content_stats = compute_feature_stats(group, CONTENT_FEATURES)
            
            # Calculate confidence based on sample size and feature variance
            sample_count = len(group)
            # Higher confidence for more samples, lower variance
            confidence = min(0.95, 0.3 + (np.log10(sample_count + 1) / 10))
            
            signature = {
                "attack_category": attack_cat,
                "protocol": proto,
                "service": service,
                "connection_state": state,
                "sample_count": sample_count,
                "confidence_score": float(np.round(confidence, 2)),
                "packet_stats": packet_stats,
                "timing_stats": timing_stats,
                "flow_stats": flow_stats,
                "tcp_stats": tcp_stats,
                "content_stats": content_stats,
                "description": generate_attack_description(
                    attack_cat, 
                    {**packet_stats, **timing_stats}, 
                    {'proto': [{'value': proto, 'frequency': 1.0}]}
                )
            }
            signatures_for_cat.append(signature)
        
        # Sort by sample count and take top signatures
        signatures_for_cat.sort(key=lambda x: x['sample_count'], reverse=True)
        attack_signatures[attack_cat] = signatures_for_cat[:MAX_SIGNATURES_PER_CATEGORY]
    
    return dict(attack_signatures)


def build_category_profiles(df: pd.DataFrame) -> Dict[str, Dict]:
    """Build high-level profiles per attack category."""
    print("[INFO] Building category profiles...")
    
    profiles = {}
    for attack_cat in df['attack_cat'].unique():
        cat_df = df[df['attack_cat'] == attack_cat]
        
        # Categorical distributions
        categorical_profile = {}
        for feat in CATEGORICAL_FEATURES:
            categorical_profile[feat] = get_top_categorical_values(cat_df, feat, top_n=10)
        
        # Numeric feature statistics
        all_numeric_stats = {}
        for feat_group in [PACKET_FEATURES, TIMING_FEATURES, FLOW_FEATURES, TCP_FEATURES, CONTENT_FEATURES]:
            all_numeric_stats.update(compute_feature_stats(cat_df, feat_group))
        
        profiles[attack_cat] = {
            "total_samples": len(cat_df),
            "label_distribution": {
                "attack": int((cat_df['label'] == 1).sum()),
                "normal": int((cat_df['label'] == 0).sum())
            },
            "categorical_profile": categorical_profile,
            "numeric_stats": all_numeric_stats,
            "description": generate_attack_description(attack_cat, all_numeric_stats, categorical_profile)
        }
    
    return profiles


def build_vector_docs(df: pd.DataFrame) -> List[Dict]:
    """Build pre-formatted documents for vector embedding."""
    print("[INFO] Building vector documents...")
    
    docs = []
    attack_categories = df['attack_cat'].unique()
    
    for attack_cat in attack_categories:
        cat_df = df[df['attack_cat'] == attack_cat]
        
        # Sample records for this category
        n_samples = min(MAX_SAMPLES_PER_CATEGORY, len(cat_df))
        
        # Stratified sampling: prioritize diverse proto/service/state combinations
        if len(cat_df) > n_samples:
            # Sample with diversity
            sampled = cat_df.groupby(['proto', 'service', 'state'], group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), max(1, n_samples // 50)), random_state=42)
            )
            if len(sampled) < n_samples:
                remaining = n_samples - len(sampled)
                additional = cat_df[~cat_df.index.isin(sampled.index)].sample(
                    n=min(remaining, len(cat_df) - len(sampled)), random_state=42
                )
                sampled = pd.concat([sampled, additional])
            sampled = sampled.head(n_samples)
        else:
            sampled = cat_df
        
        for _, row in sampled.iterrows():
            # Create text representation
            text_parts = [f"UNSW_NB15: attack_cat={attack_cat}"]
            text_parts.append(f"label={int(row.get('label', 0))}")
            
            # Key features for embedding
            key_features = ['proto', 'service', 'state', 'spkts', 'dpkts', 
                          'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
                          'sload', 'dload', 'tcprtt', 'synack', 'ackdat',
                          'ct_srv_src', 'ct_state_ttl', 'is_sm_ips_ports']
            
            for feat in key_features:
                val = row.get(feat)
                if pd.notna(val):
                    str_val = str(val).strip()
                    if str_val and str_val != '-' and str_val != 'nan':
                        # Round floats for cleaner text
                        try:
                            float_val = float(val)
                            if float_val == int(float_val):
                                str_val = str(int(float_val))
                            else:
                                str_val = str(round(float_val, 4))
                        except:
                            pass
                        text_parts.append(f"{feat}={str_val}")
            
            docs.append({
                "text": " ".join(text_parts),
                "attack_category": attack_cat,
                "label": int(row.get('label', 0))
            })
    
    print(f"[INFO] Generated {len(docs)} vector documents")
    return docs


def build_threat_intelligence(df: pd.DataFrame, signatures: Dict) -> Dict:
    """Build threat intelligence cache for fast lookups."""
    print("[INFO] Building threat intelligence cache...")
    
    threat_intel = {}
    
    for attack_cat, sigs in signatures.items():
        if attack_cat == 'Normal':
            continue  # Skip normal traffic for threat intel
        
        cat_intel = []
        for sig in sigs[:20]:  # Top 20 signatures per category
            intel_entry = {
                "signature_id": f"{attack_cat}_{sig['protocol']}_{sig['service']}_{sig['connection_state']}".replace(' ', '_'),
                "attack_type": attack_cat,
                "confidence_score": sig['confidence_score'],
                "sample_count": sig['sample_count'],
                "indicators": {
                    "protocol": sig['protocol'],
                    "service": sig['service'],
                    "connection_state": sig['connection_state'],
                    # Key discriminative features
                    "sbytes_range": [
                        sig['packet_stats'].get('sbytes_p25', 0),
                        sig['packet_stats'].get('sbytes_p75', 0)
                    ],
                    "rate_range": [
                        sig['timing_stats'].get('rate_p25', 0),
                        sig['timing_stats'].get('rate_p75', 0)
                    ],
                    "spkts_typical": sig['packet_stats'].get('spkts_median', 0),
                    "dpkts_typical": sig['packet_stats'].get('dpkts_median', 0)
                },
                "description": sig['description']
            }
            cat_intel.append(intel_entry)
        
        threat_intel[attack_cat] = cat_intel
    
    return threat_intel


def build_discriminative_features(df: pd.DataFrame) -> Dict[str, Dict]:
    """Identify most discriminative features per attack category."""
    print("[INFO] Computing discriminative features...")
    
    discriminative = {}
    normal_df = df[df['attack_cat'] == 'Normal']
    
    numeric_features = PACKET_FEATURES + TIMING_FEATURES + FLOW_FEATURES + TCP_FEATURES
    
    for attack_cat in df['attack_cat'].unique():
        if attack_cat == 'Normal':
            continue
        
        attack_df = df[df['attack_cat'] == attack_cat]
        feature_diffs = {}
        
        for feat in numeric_features:
            if feat not in df.columns:
                continue
            
            attack_vals = pd.to_numeric(attack_df[feat], errors='coerce').dropna()
            normal_vals = pd.to_numeric(normal_df[feat], errors='coerce').dropna()
            
            if len(attack_vals) > 0 and len(normal_vals) > 0:
                attack_mean = attack_vals.mean()
                normal_mean = normal_vals.mean()
                
                # Calculate effect size (Cohen's d approximation)
                pooled_std = np.sqrt((attack_vals.std()**2 + normal_vals.std()**2) / 2)
                if pooled_std > 0:
                    effect_size = abs(attack_mean - normal_mean) / pooled_std
                else:
                    effect_size = 0
                
                feature_diffs[feat] = {
                    "attack_mean": float(np.round(attack_mean, 4)),
                    "normal_mean": float(np.round(normal_mean, 4)),
                    "effect_size": float(np.round(effect_size, 4)),
                    "direction": "higher" if attack_mean > normal_mean else "lower"
                }
        
        # Sort by effect size and keep top discriminative features
        sorted_features = sorted(feature_diffs.items(), key=lambda x: x[1]['effect_size'], reverse=True)
        discriminative[attack_cat] = {
            "top_features": dict(sorted_features[:10]),
            "sample_count": len(attack_df)
        }
    
    return discriminative


def main():
    """Main function to build all KB components."""
    print("=" * 60)
    print("Building Optimized UNSW-NB15 Knowledge Base")
    print("=" * 60)
    
    # Load data
    df = load_training_data()
    
    # Print attack category distribution
    print("\n[INFO] Attack Category Distribution:")
    for cat, count in df['attack_cat'].value_counts().items():
        print(f"  {cat}: {count} ({count/len(df)*100:.2f}%)")
    
    # Build components
    signatures = build_attack_signatures(df)
    profiles = build_category_profiles(df)
    vector_docs = build_vector_docs(df)
    threat_intel = build_threat_intelligence(df, signatures)
    discriminative = build_discriminative_features(df)
    
    # Count total signatures
    total_sigs = sum(len(sigs) for sigs in signatures.values())
    print(f"\n[INFO] Generated {total_sigs} attack signatures across {len(signatures)} categories")
    
    # Build main knowledge base
    knowledge_base = {
        "version": "2.0",
        "source": "UNSW-NB15 Training Set",
        "total_records": len(df),
        "attack_categories": list(df['attack_cat'].unique()),
        "attack_signatures": signatures,
        "category_profiles": profiles,
        "discriminative_features": discriminative
    }
    
    # Build enhanced cache
    enhanced_cache = {
        "version": "2.0",
        "threat_intelligence": threat_intel,
        "category_quick_lookup": {
            cat: {
                "total_samples": prof['total_samples'],
                "top_proto": prof['categorical_profile']['proto'][0]['value'] if prof['categorical_profile']['proto'] else '-',
                "top_service": prof['categorical_profile']['service'][0]['value'] if prof['categorical_profile']['service'] else '-',
                "description": prof['description']
            }
            for cat, prof in profiles.items()
        }
    }
    
    # Save files
    print("\n[INFO] Saving knowledge base files...")
    
    # 1. Main knowledge base
    kb_path = OUTPUT_DIR / "unsw_knowledge_base.json"
    with open(kb_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2)
    print(f"  Saved: {kb_path} ({kb_path.stat().st_size / 1024:.1f} KB)")
    
    # 2. Vector docs (JSONL format)
    vector_path = OUTPUT_DIR / "unsw_vector_docs.jsonl"
    with open(vector_path, 'w', encoding='utf-8') as f:
        for doc in vector_docs:
            f.write(json.dumps(doc) + '\n')
    print(f"  Saved: {vector_path} ({vector_path.stat().st_size / 1024:.1f} KB)")
    
    # 3. Enhanced cache
    cache_path = OUTPUT_DIR / "enhanced_kb_cache.json"
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_cache, f, indent=2)
    print(f"  Saved: {cache_path} ({cache_path.stat().st_size / 1024:.1f} KB)")
    
    print("\n" + "=" * 60)
    print("Knowledge Base Generation Complete!")
    print("=" * 60)
    
    # Print summary
    print("\nSummary:")
    print(f"  - Attack Categories: {len(df['attack_cat'].unique())}")
    print(f"  - Total Signatures: {total_sigs}")
    print(f"  - Vector Documents: {len(vector_docs)}")
    print(f"  - Threat Intel Entries: {sum(len(v) for v in threat_intel.values())}")
    
    return knowledge_base, vector_docs, enhanced_cache


if __name__ == "__main__":
    main()
