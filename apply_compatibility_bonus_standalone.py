"""
Standalone compatibility bonus application script (fully independent version)
Can be copied to any folder for use. Apply compatibility bonus to screening result files and re-sort.

Usage:
    python apply_compatibility_bonus_standalone.py [input_file] [output_file]
    
Examples:
    python apply_compatibility_bonus_standalone.py agent_recommended_results.csv agent_recommended_results_with_bonus.csv
    Or overwrite the original file:
    python apply_compatibility_bonus_standalone.py agent_recommended_results.csv
"""

import pandas as pd
import sys
import os
import json
import re
import logging
from typing import Dict, Any, Optional

# Configure logging - output to both console and file
import datetime
log_filename = f"compatibility_bonus_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Clear existing handlers
logger.handlers.clear()

# Create file handler
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Console only shows INFO and above
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent log propagation to root logger
logger.propagate = False

logger.info(f"Log saved to file: {log_filename}")

# Compatibility bonus step (score weight reduction per satisfied condition)
COMPATIBILITY_BONUS_STEP = 0.3

# Main group mapping
_GROUPS = {
    'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
    'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
    'Al': 13, 'Ga': 13,
    'Zn': 12, 'Cd': 12,
    'Fe': 8, 'Co': 9, 'Ni': 10, 'Mn': 7, 'Cr': 6, 'Cu': 11, 'Tb': 3, 'La': 3, 'Pr': 3, 'Bi': 15, 'Sn': 14
}

def _same_group(cat1: Optional[str], cat2: Optional[str]) -> Optional[bool]:
    """Check if two cations are in the same main group"""
    if not cat1 or not cat2:
        return None
    return _GROUPS.get(cat1) == _GROUPS.get(cat2) if (cat1 in _GROUPS and cat2 in _GROUPS) else None


def _norm_formula_key(formula: str) -> str:
    """Normalize salt name as key"""
    if formula is None:
        return ""
    
    try:
        if pd.isna(formula):
            return ""
    except (TypeError, ValueError):
        pass
    
    if not isinstance(formula, str):
        try:
            import math
            if isinstance(formula, float) and math.isnan(formula):
                return ""
            formula = str(formula)
            if formula.lower() in ['nan', 'none', 'null', '']:
                return ""
        except Exception:
            return ""
    
    if not isinstance(formula, str):
        return ""
    
    if not formula:
        return ""
    
    try:
        return formula.strip().replace(' ', '').replace('_', '').lower()
    except AttributeError:
        return ""


def _parse_charge(anion_str: str) -> Optional[int]:
    """Parse charge from anion string"""
    if not anion_str:
        return None
    
    # Strategy: match from right to left, prioritize the last charge digit
    # For "S2O32-", should match "2-" not "32-"
    
    # 1. If ends with "-", try to match last 1-2 digits (charge is typically 1-6)
    if anion_str.endswith('-'):
        # Try matching the last single digit first (most precise)
        m = re.search(r'(\d)(?!\d)([-+])$', anion_str)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 6:  # Charge is typically in the 1-6 range
                return -val
        
        # If the last single digit is not in the 1-6 range, try matching the last digit from a 2-digit sequence
        # e.g., in "S2O32-", match "32-" but only take "2"
        m = re.search(r'(\d)(\d)([-+])$', anion_str)
        if m:
            val = int(m.group(2))
            if 1 <= val <= 6:
                return -val
        
        # Default to -1 if no match
        return -1
    
    # 2. If ends with "+"
    if anion_str.endswith('+'):
        m = re.search(r'(\d)([+])$', anion_str)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 6:
                return val
        m = re.search(r'(\d)(\d)([+])$', anion_str)
        if m:
            val = int(m.group(2))
            if 1 <= val <= 6:
                return val
    
    # 3. Match charge format in the middle: -1, -2, 2-, etc. (not at end)
    # Only match single-digit charges (safer)
    m = re.search(r'([+-]?)(\d)([-+])', anion_str)
    if m:
        val = int(m.group(2))
        if 1 <= val <= 6:
            sign = -1 if m.group(1) == '-' or m.group(3) == '-' else 1
            return sign * val
    
    return None


def _anion_is_polyatomic(anion_str: str) -> bool:
    """Check if anion is polyatomic"""
    if not anion_str:
        return False
    
    # Remove charge symbols and digits, keep only element symbols
    # First remove trailing charge (e.g., "2-", "-2", "3-", etc.)
    clean = re.sub(r'[-+]?\d*[-+]?$', '', anion_str).strip()
    
    # If length > 2 after removing charge, it's definitely polyatomic
    # Monatomic anions: Cl-, F-, Br-, I-, O2-, S2-, etc. (typically 1-2 chars after removing charge)
    # Polyatomic anions: NO3-, SO4-, CO3-, S2O32-, etc. (typically 3+ chars after removing charge)
    
    # Special case: if only 1-2 chars after removing charge and is a single element symbol, it's monatomic
    if len(clean) <= 2:
        if re.match(r'^[A-Z][a-z]?$', clean):
            return False
    
    return len(clean) > 2


def _cation_symbol(cation: str) -> Optional[str]:
    """Extract element symbol from cation name"""
    if not cation:
        return None
    # Match the leading element symbol (1-2 letters)
    m = re.match(r'^([A-Z][a-z]?)', cation)
    return m.group(1) if m else None


def load_structured_meta(data_dir: str = 'data') -> Dict[str, Dict[str, Any]]:
    """Load structured metadata"""
    meta = {}
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory not found: {data_dir}")
        return meta
    
    for fn in os.listdir(data_dir):
        if not fn.endswith('_structured.json'):
            continue
        path = os.path.join(data_dir, fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                j = json.load(f)
            feat = j.get('Salt_Features', {})
            therm = j.get('thermal_properties', {})
            formula = feat.get('Salt_Formula') or feat.get('Compound_Name') or ''
            key = _norm_formula_key(formula)
            anion_str = feat.get('Anion') or ''
            
            # Prefer charge and polyatomic info from JSON file; fall back to parsing functions
            anion_charge = feat.get('Anion_Charge')
            if anion_charge is None:
                anion_charge = _parse_charge(anion_str)
            
            anion_poly = feat.get('Anion_Is_Polyatomic')
            if anion_poly is None:
                anion_poly = _anion_is_polyatomic(anion_str)
            
            meta[key] = {
                'formula': formula,
                'Tm': float(therm.get('Melting_temperature_C')) if therm.get('Melting_temperature_C') is not None else None,
                'dH': float(therm.get('Enthalpy_value_J_per_g')) if therm.get('Enthalpy_value_J_per_g') is not None else None,
                'nH2O': feat.get('Hydration_Number'),
                'cation': feat.get('Cation'),
                'cation_symbol': _cation_symbol(feat.get('Cation') or ''),
                'cation_radius': feat.get('Cation_Radius_Angstrom'),
                'anion': anion_str,
                'anion_charge': anion_charge,
                'anion_poly': anion_poly
            }
            if meta[key]['nH2O'] is not None:
                try:
                    meta[key]['nH2O'] = float(meta[key]['nH2O'])
                except Exception:
                    meta[key]['nH2O'] = None
            if meta[key]['cation_radius'] is not None:
                try:
                    meta[key]['cation_radius'] = float(meta[key]['cation_radius'])
                except Exception:
                    meta[key]['cation_radius'] = None
        except Exception as e:
            logger.debug(f"Failed to load file {fn}: {e}")
            continue
    return meta


def match_meta(meta_map: Dict[str, Dict[str, Any]], salt_name: str) -> Optional[Dict[str, Any]]:
    """Match salt name to metadata"""
    if pd.isna(salt_name):
        return None
    if salt_name is None:
        return None
    
    if not isinstance(salt_name, str):
        try:
            import math
            if isinstance(salt_name, float) and math.isnan(salt_name):
                return None
            salt_name = str(salt_name)
            if salt_name.lower() in ['nan', 'none', 'null', '']:
                return None
        except Exception:
            return None
    
    if not isinstance(salt_name, str):
        return None
    
    if not salt_name or not salt_name.strip():
        return None
    
    key = _norm_formula_key(salt_name)
    if not key:
        return None
    if key in meta_map:
        return meta_map[key]
    key2 = key.replace('·', '.')
    for k in meta_map.keys():
        if _norm_formula_key(k) == key or _norm_formula_key(k).replace('·', '.') == key2:
            return meta_map[k]
    return None


def _check_cation_bonus(meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> bool:
    """Check cation bonus condition"""
    cation_a = meta_a.get('cation')
    cation_b = meta_b.get('cation')
    
    if not cation_a or not cation_b:
        return False
    
    # Condition 1: Same cation
    if cation_a == cation_b:
        return True
    
    # Condition 2: Cations in the same main group
    cation_symbol_a = meta_a.get('cation_symbol')
    cation_symbol_b = meta_b.get('cation_symbol')
    
    if cation_symbol_a and cation_symbol_b:
        same_group = _same_group(cation_symbol_a, cation_symbol_b)
        if same_group is True:
            return True
    
    # Condition 3: Ionic radius difference < 15%
    radius_a = meta_a.get('cation_radius')
    radius_b = meta_b.get('cation_radius')
    
    if radius_a is not None and radius_b is not None:
        try:
            radius_a = float(radius_a)
            radius_b = float(radius_b)
            if radius_a > 0 and radius_b > 0:
                max_radius = max(radius_a, radius_b)
                diff_ratio = abs(radius_a - radius_b) / max_radius
                if diff_ratio < 0.15:
                    return True
        except (ValueError, TypeError):
            pass
    
    return False


def _check_anion_bonus(meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> bool:
    """Check anion bonus condition"""
    anion_a = meta_a.get('anion')
    anion_b = meta_b.get('anion')
    
    if not anion_a or not anion_b:
        return False
    
    # Condition 1: Same charge
    charge_a = meta_a.get('anion_charge')
    charge_b = meta_b.get('anion_charge')
    
    if charge_a is not None and charge_b is not None:
        if charge_a == charge_b:
            return True
    
    # Condition 2: Same atomicity
    poly_a = meta_a.get('anion_poly')
    poly_b = meta_b.get('anion_poly')
    
    if poly_a is not None and poly_b is not None:
        if poly_a == poly_b:
            return True
    
    return False


def _check_hydration_bonus(meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> bool:
    """Check hydration number bonus condition"""
    nH2O_a = meta_a.get('nH2O')
    nH2O_b = meta_b.get('nH2O')
    
    if nH2O_a is not None and nH2O_b is not None:
        try:
            nH2O_a = float(nH2O_a)
            nH2O_b = float(nH2O_b)
            diff = abs(nH2O_a - nH2O_b)
            if diff < 5:
                return True
        except (ValueError, TypeError):
            pass
    
    return False


def _calculate_pair_bonus(meta_a: Dict[str, Any], meta_b: Dict[str, Any], debug: bool = False, salt_a_name: str = "", salt_b_name: str = "") -> int:
    """Calculate compatibility bonus count for salt pair (N_bonus)"""
    n_bonus = 0
    
    cation_bonus = _check_cation_bonus(meta_a, meta_b)
    anion_bonus = _check_anion_bonus(meta_a, meta_b)
    hydration_bonus = _check_hydration_bonus(meta_a, meta_b)
    
    if cation_bonus:
        n_bonus += 1
    if anion_bonus:
        n_bonus += 1
    if hydration_bonus:
        n_bonus += 1
    
    if debug:
        logger.debug(f"  Salt pair {salt_a_name} vs {salt_b_name}:")
        logger.debug(f"    Cation bonus: {cation_bonus} (A: {meta_a.get('cation')}, B: {meta_b.get('cation')})")
        logger.debug(f"    Anion bonus: {anion_bonus} (A: {meta_a.get('anion')} (charge:{meta_a.get('anion_charge')}, polyatomic:{meta_a.get('anion_poly')}), B: {meta_b.get('anion')} (charge:{meta_b.get('anion_charge')}, polyatomic:{meta_b.get('anion_poly')}))")
        logger.debug(f"    Hydration number bonus: {hydration_bonus} (A: {meta_a.get('nH2O')}, B: {meta_b.get('nH2O')})")
        logger.debug(f"    N_bonus = {n_bonus}")
    
    return n_bonus


def calculate_compatibility_bonus(candidate: Dict[str, Any], meta_map: Dict[str, Dict[str, Any]]) -> tuple:
    """Calculate compatibility bonus for single candidate combination
    
    Returns:
        tuple: (w_compat, n_bonus) - compatibility weight and bonus count
    """
    salt_a_name = candidate.get('salt_a', '')
    salt_b_name = candidate.get('salt_b', '')
    salt_c_name = candidate.get('salt_c', None)
    
    def is_valid_salt(salt):
        if pd.isna(salt):
            return False
        if salt is None:
            return False
        if not salt or str(salt).strip().lower() in ['nan', 'none', 'null', '']:
            return False
        return True
    
    if not is_valid_salt(salt_a_name) or not is_valid_salt(salt_b_name):
        return (1.0, 0)
    
    meta_a = match_meta(meta_map, salt_a_name)
    meta_b = match_meta(meta_map, salt_b_name)
    
    if not meta_a or not meta_b:
        return (1.0, 0)
    
    if is_valid_salt(salt_c_name):
        # Ternary combination
        meta_c = match_meta(meta_map, salt_c_name)
        if not meta_c:
            return (1.0, 0)
        
        logger.debug(f"\n{'='*60}")
        logger.debug("Debug info: Calculating N_bonus for ternary combination")
        logger.debug(f"Salt A: {salt_a_name}")
        logger.debug(f"Salt B: {salt_b_name}")
        logger.debug(f"Salt C: {salt_c_name}")
        logger.debug(f"{'='*60}")
        
        n_bonus_ab = _calculate_pair_bonus(meta_a, meta_b, debug=True, salt_a_name=salt_a_name, salt_b_name=salt_b_name)
        n_bonus_ac = _calculate_pair_bonus(meta_a, meta_c, debug=True, salt_a_name=salt_a_name, salt_b_name=salt_c_name)
        n_bonus_bc = _calculate_pair_bonus(meta_b, meta_c, debug=True, salt_a_name=salt_b_name, salt_b_name=salt_c_name)
        
        logger.debug("\nN_bonus for each salt pair:")
        logger.debug(f"  AB (Salt A vs Salt B): {n_bonus_ab}")
        logger.debug(f"  AC (Salt A vs Salt C): {n_bonus_ac}")
        logger.debug(f"  BC (Salt B vs Salt C): {n_bonus_bc}")
        logger.debug(f"Final N_bonus = min({n_bonus_ab}, {n_bonus_ac}, {n_bonus_bc}) = {min(n_bonus_ab, n_bonus_ac, n_bonus_bc)}")
        logger.debug(f"{'='*60}\n")
        
        n_bonus = min(n_bonus_ab, n_bonus_ac, n_bonus_bc)
    else:
        # Binary combination
        logger.debug(f"\n{'='*60}")
        logger.debug("Debug info: Calculating N_bonus for binary combination")
        logger.debug(f"Salt A: {salt_a_name}")
        logger.debug(f"Salt B: {salt_b_name}")
        logger.debug(f"{'='*60}")
        n_bonus = _calculate_pair_bonus(meta_a, meta_b, debug=True, salt_a_name=salt_a_name, salt_b_name=salt_b_name)
        logger.debug(f"Final N_bonus = {n_bonus}")
        logger.debug(f"{'='*60}\n")
    
    w_compat = 1.0 - COMPATIBILITY_BONUS_STEP * n_bonus
    w_compat = max(0.0, min(1.0, w_compat))
    
    return (w_compat, n_bonus)


def apply_compatibility_bonus_to_dataframe(df: pd.DataFrame, meta_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Apply compatibility bonus to DataFrame"""
    if df.empty:
        logger.warning("DataFrame is empty, no processing needed")
        return df
    
    original_count = len(df)
    logger.info(f"Starting compatibility bonus application, {original_count} combinations total")
    
    df = df.reset_index(drop=True)
    df['_original_row_order'] = range(len(df))
    
    # Determine grouping column (prefer compliance_level, fall back to compliance_desc)
    group_col = None
    if 'compliance_level' in df.columns:
        group_col = 'compliance_level'
    elif 'compliance_desc' in df.columns:
        group_col = 'compliance_desc'
    
    if group_col:
        if group_col == 'compliance_level':
            # Sort by numeric value descending: 1.0 > 0.9 > 0.8 > others
            def get_group_priority(value):
                if pd.isna(value):
                    return 999999
                try:
                    num_value = float(value)
                    return -num_value
                except (ValueError, TypeError):
                    return 999999
            
            unique_groups = df[group_col].dropna().unique()
            sorted_groups = sorted(unique_groups, key=get_group_priority)
            group_order = {group: i for i, group in enumerate(sorted_groups)}
            df['_group_order'] = df[group_col].map(group_order)
            df['_group_order'] = df['_group_order'].fillna(999999)
            logger.info(f"Detected grouping column: {group_col}, Group order (by priority): {sorted_groups}")
        elif group_col == 'compliance_desc':
            priority_map = {
                'Fully compliant': 0,
                'Perfect': 0,
                '完全达标': 0,
                '90% compliant': 1,
                '90% Compliance': 1,
                '90%符合': 1,
                '80% compliant': 2,
                '80% Compliance': 2,
                '80%符合': 2,
                'Low compliance': 3,
                'Low Compliance': 3,
                '低符合度': 3,
                '低符合': 3
            }
            
            def get_desc_priority(value):
                if pd.isna(value):
                    return 999999
                return priority_map.get(str(value), 999)
            
            unique_groups = df[group_col].dropna().unique()
            sorted_groups = sorted(unique_groups, key=get_desc_priority)
            group_order = {group: i for i, group in enumerate(sorted_groups)}
            df['_group_order'] = df[group_col].map(group_order)
            df['_group_order'] = df['_group_order'].fillna(999999)
            logger.info(f"Detected grouping column: {group_col}, Group order (by priority): {sorted_groups}")
        else:
            first_occurrence = {}
            for idx, row in df.iterrows():
                group_value = row[group_col]
                if pd.notna(group_value) and group_value not in first_occurrence:
                    first_occurrence[group_value] = idx
            
            sorted_groups = sorted(first_occurrence.keys(), key=lambda x: first_occurrence[x])
            group_order = {group: i for i, group in enumerate(sorted_groups)}
            df['_group_order'] = df[group_col].map(group_order)
            df['_group_order'] = df['_group_order'].fillna(999999)
            logger.info(f"Detected grouping column: {group_col}, Group order: {sorted_groups}")
    else:
        logger.warning("No grouping column found (compliance_level or compliance_desc), will sort globally by score")
        df['_group_order'] = 0
    
    if 'score' not in df.columns:
        logger.warning("Score column not found, creating default score column")
        if 'tm_pred' in df.columns and 'dh_pred' in df.columns:
            if 'target_tm' in df.columns and 'target_dh' in df.columns:
                df['score'] = abs(df['tm_pred'] - df['target_tm']) + 0.02 * abs(df['dh_pred'] - df['target_dh'])
            else:
                df['score'] = 0.0
        else:
            df['score'] = 0.0
    
    if 'score_original' in df.columns:
        logger.info("score_original column exists, using existing values")
    else:
        df['score_original'] = df['score'].copy()
        logger.info("Saved original score as score_original")
    
    compatibility_weights = []
    compatibility_bonuses = []
    n_bonus_list = []
    new_scores = []
    
    for idx, row in df.iterrows():
        candidate = row.to_dict()
        original_score = candidate.get('score_original', candidate.get('score', float('inf')))
        
        if pd.isna(original_score) or original_score == float('inf'):
            compatibility_weights.append(1.0)
            compatibility_bonuses.append(0.0)
            n_bonus_list.append(0)
            new_scores.append(original_score)
            continue
        
        w_compat, n_bonus = calculate_compatibility_bonus(candidate, meta_map)
        
        new_score = float(original_score) * w_compat
        new_scores.append(new_score)
        
        compatibility_weights.append(w_compat)
        compatibility_bonuses.append(1.0 - w_compat)
        n_bonus_list.append(n_bonus)
    
    logger.info(f"Processed compatibility bonus for {len(compatibility_weights)} combinations")
    
    df['compatibility_weight'] = compatibility_weights
    df['compatibility_bonus'] = compatibility_bonuses
    df['N_bonus'] = n_bonus_list
    
    if 'score' in df.columns:
        df = df.drop(columns=['score'])
    
    df['score'] = new_scores
    
    valid_scores = df[~pd.isna(df['score']) & (df['score'] != float('inf'))]
    if len(valid_scores) > 0:
        if 'score_original' in df.columns:
            updated_count = len(df[(df['score'] != df['score_original']) & 
                                  (~pd.isna(df['score'])) & 
                                  (~pd.isna(df['score_original'])) &
                                  (df['score'] != float('inf')) &
                                  (df['score_original'] != float('inf'))])
            logger.info(f"Updated score for {updated_count} valid combinations")
            
            if updated_count > 0:
                sample = df[(df['score'] != df['score_original']) & 
                           (~pd.isna(df['score'])) & 
                           (~pd.isna(df['score_original'])) &
                           (df['score'] != float('inf')) &
                           (df['score_original'] != float('inf'))].head(3)
                logger.info("Score update examples (top 3):")
                for idx, row in sample.iterrows():
                    orig = row['score_original']
                    new = row['score']
                    weight = row['compatibility_weight']
                    logger.info(f"  Original: {orig:.6f} -> New: {new:.6f} (Weight: {weight:.2f})")
    
    if '_group_order' in df.columns:
        df = df.sort_values(['_group_order', 'score'], ascending=[True, True])
        logger.info("Maintained group order, sorted within groups by score")
        df = df.drop(columns=['_original_row_order', '_group_order'])
    else:
        df = df.sort_values('score', ascending=True)
        if '_original_row_order' in df.columns:
            df = df.drop(columns=['_original_row_order'])
    
    if len(df) != original_count:
        logger.error(f"Error: Combination count changed after bonus! Original: {original_count}, Current: {len(df)}")
    else:
        logger.info(f"Compatibility bonus applied, count unchanged: {len(df)}")
    
    bonus_stats = {
        'No bonus (N=0, w=1.0)': len(df[df['N_bonus'] == 0]),
        f'1 condition (N=1, w={max(0.0, 1.0 - COMPATIBILITY_BONUS_STEP * 1):.1f})': len(df[df['N_bonus'] == 1]),
        f'2 conditions (N=2, w={max(0.0, 1.0 - COMPATIBILITY_BONUS_STEP * 2):.1f})': len(df[df['N_bonus'] == 2]),
        f'3 conditions (N=3, w={max(0.0, 1.0 - COMPATIBILITY_BONUS_STEP * 3):.1f})': len(df[df['N_bonus'] == 3])
    }
    logger.info(f"Compatibility bonus statistics: {bonus_stats}")
    
    return df


def main():
    """Main function"""
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    
    if len(sys.argv) < 2:
        print("=" * 60, flush=True)
        print("Compatibility Bonus Application Script (Standalone)", flush=True)
        print("=" * 60, flush=True)
        print("\nUsage:", flush=True)
        print("  python apply_compatibility_bonus_standalone.py <input_file> [output_file]", flush=True)
        print("\nExamples:", flush=True)
        print("  python apply_compatibility_bonus_standalone.py agent_recommended_results.csv", flush=True)
        print("  python apply_compatibility_bonus_standalone.py agent_recommended_results.csv output_with_bonus.csv", flush=True)
        print("\nNotes:", flush=True)
        print("  - If no output file specified, overwrites original (auto-backup)", flush=True)
        print("  - Requires data folder with *_structured.json files", flush=True)
        print("=" * 60, flush=True)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    print(f"\nProcessing: {input_file}", flush=True)
    print("=" * 60, flush=True)
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        backup_file = f"{base_name}_backup{ext}"
        output_file = input_file
        
        if os.path.exists(output_file):
            import shutil
            shutil.copy2(output_file, backup_file)
            print(f"Backup created: {backup_file}", flush=True)
            logger.info(f"Backup created: {backup_file}")
    
    if not os.path.exists(input_file):
        error_msg = f"Input file not found: {input_file}"
        print(f"[Error] {error_msg}", flush=True)
        logger.error(error_msg)
        sys.exit(1)
    
    print(f"Reading input file: {input_file}", flush=True)
    logger.info(f"Reading input file: {input_file}")
    
    try:
        df = pd.read_csv(input_file, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file, encoding='gbk')
        except Exception as e:
            logger.error(f"Cannot read file: {e}")
            sys.exit(1)
    
    print(f"Successfully read {len(df)} rows", flush=True)
    logger.info(f"Successfully read {len(df)} rows")
    logger.info(f"Columns: {list(df.columns)}")
    
    required_cols = ['salt_a', 'salt_b']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        print(f"[Error] {error_msg}", flush=True)
        logger.error(error_msg)
        sys.exit(1)
    
    print("Loading metadata...", flush=True)
    logger.info("Loading metadata...")
    meta_map = load_structured_meta('data')
    if len(meta_map) == 0:
        error_msg = "No metadata loaded. Ensure data folder exists with *_structured.json files"
        print(f"[Error] {error_msg}", flush=True)
        logger.error(error_msg)
        sys.exit(1)
    print(f"Successfully loaded metadata for {len(meta_map)} salts", flush=True)
    logger.info(f"Successfully loaded metadata for {len(meta_map)} salts")
    
    print("Applying compatibility bonus...", flush=True)
    logger.info("Applying compatibility bonus...")
    df_result = apply_compatibility_bonus_to_dataframe(df, meta_map)
    
    if 'score' not in df_result.columns:
        logger.error("Error: score column missing!")
    else:
        valid_scores = df_result[~pd.isna(df_result['score']) & (df_result['score'] != float('inf'))]
        logger.info(f"Valid score count: {len(valid_scores)}/{len(df_result)}")
        
        if 'score_original' in df_result.columns:
            updated = df_result[(df_result['score'] != df_result['score_original']) & 
                               (~pd.isna(df_result['score'])) & 
                               (~pd.isna(df_result['score_original'])) &
                               (df_result['score'] != float('inf')) &
                               (df_result['score_original'] != float('inf'))]
            logger.info(f"Updated score count: {len(updated)}/{len(valid_scores)}")
    
    print(f"\nSaving results to: {output_file}", flush=True)
    logger.info(f"Saving results to: {output_file}")
    try:
        if 'score' not in df_result.columns:
            print("Warning: score column missing, creating default", flush=True)
            logger.warning("Warning: score column missing, creating default")
            df_result['score'] = df_result.get('score_original', 0.0)
        
        cols = [col for col in df_result.columns if col != 'score']
        cols.append('score')
        df_result = df_result[cols]
        
        df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Successfully saved {len(df_result)} rows to {output_file}", flush=True)
        logger.info(f"Successfully saved {len(df_result)} rows to {output_file}")
        logger.info(f"  Column order: {list(df_result.columns)}")
        logger.info(f"  Last column: {df_result.columns[-1]}")
    except Exception as e:
        error_msg = f"Failed to save file: {e}"
        print(f"[Error] {error_msg}", flush=True)
        logger.error(error_msg)
        sys.exit(1)
    
    logger.info("\n" + "="*60)
    logger.info("Top 5 results (sorted by new score):")
    logger.info("="*60)
    
    display_cols = ['salt_a', 'salt_b']
    if 'salt_c' in df_result.columns:
        display_cols.append('salt_c')
    display_cols.extend(['score_original', 'N_bonus', 'compatibility_weight', 'compatibility_bonus', 'score'])
    available_cols = [col for col in display_cols if col in df_result.columns]
    
    if 'score' in df_result.columns and 'score_original' in df_result.columns:
        print("\nScore comparison (top 5, score as last column):", flush=True)
        print(df_result[available_cols].head().to_string())
        print("\nExample calculation (1st combination):", flush=True)
        if len(df_result) > 0:
            first_row = df_result.iloc[0]
            if 'score_original' in first_row and 'compatibility_weight' in first_row:
                orig = first_row['score_original']
                weight = first_row['compatibility_weight']
                new = first_row['score']
                n_bonus_val = first_row.get('N_bonus', 0)
                print(f"  Original score (score_original): {orig:.6f}", flush=True)
                print(f"  N_bonus: {n_bonus_val}", flush=True)
                print(f"  Compatibility weight: {weight:.2f}", flush=True)
                print(f"  New score (last column): {new:.6f} = {orig:.6f} x {weight:.2f}", flush=True)
                print(f"  Score reduction: {(1-weight)*100:.1f}%", flush=True)
    else:
        print(df_result[available_cols].head().to_string())
        if 'score' not in df_result.columns:
            logger.warning("Warning: score column missing in output!")
    
    print("\n" + "=" * 60, flush=True)
    print("Compatibility bonus applied!", flush=True)
    print("=" * 60, flush=True)
    logger.info("\n" + "=" * 60)
    logger.info("Compatibility bonus applied!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
