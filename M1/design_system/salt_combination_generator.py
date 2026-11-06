import pandas as pd
import numpy as np
import os
import joblib
from typing import List, Tuple, Dict, Any, Optional
import json
import re

def _normalize_name(s):
    return ''.join(str(s).strip().lower().replace('_', '').replace(' ', '').replace('\u3000', ''))

def _detect_salt_column(df: pd.DataFrame):
    cols = list(df.columns)
    norm_map = {col: _normalize_name(col) for col in cols}
    # Priority synonyms (common column names)
    priority_raw = ['salt_type', 'salt', 'salttype', 'saltname', 'saltcategory']
    priority = [_normalize_name(x) for x in priority_raw]
    # Prefer exact synonym matches
    for col, norm in norm_map.items():
        if norm in priority:
            return col
    # Fallback: columns containing "salt"
    candidates = []
    for col in cols:
        cname = _normalize_name(col)
        if 'salt' in cname:
            candidates.append(col)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # If multiple, take the first by appearance order
        return candidates[0]
    return None

def _detect_id_column(df: pd.DataFrame):
    cols = list(df.columns)
    # Common ID column names
    priority_raw = ['ID', 'Id', 'id', 'sample_id', 'sampleid', 'row_id', 'rowid']
    priority = [_normalize_name(x) for x in priority_raw]
    for col in cols:
        if _normalize_name(col) in priority:
            return col
    return None

# === Agent search additions start ===

# Base feature order consistent with training (kept consistent across functions in this file)
BASE_FEATURES = [
    'Tm (°C)', 'ΔHm (J/g)', 'Molecular Weight', 'Hydrogen Bond Donor Count',
    'Hydrogen Bond Acceptor Count', 'Exact Mass', 'Topological Polar Surface Area',
    'Heavy Atom Count', 'Complexity', 'Covalently-Bonded Unit Count', 'Hydration Number',
    'Elem_H', 'Elem_O', 'Hydration Density', 'Hydrogen Bond Density', 'Polarity per Atom'
]

def _build_feature_columns():
    cols = []
    for k in (1, 2, 3):
        for feat in BASE_FEATURES:
            cols.append(f"{k}-{feat}")
    cols += ['ID1', 'ID2', 'ID3']
    return cols

FEATURE_COLUMNS = _build_feature_columns()

def _extract_model_feature_names(model: Any) -> Optional[List[str]]:
    """Extract feature names used during training from a model object (if available)."""
    if model is None:
        return None

    # Common sklearn attribute (usually available when trained with a DataFrame)
    names = getattr(model, 'feature_names_in_', None)
    if names is not None and len(names) > 0:
        return [str(x) for x in list(names)]

    # Feature names stored in an XGBoost booster
    try:
        booster = model.get_booster() if hasattr(model, 'get_booster') else None
        if booster is not None:
            bnames = booster.feature_names
            if bnames is not None and len(bnames) > 0:
                return [str(x) for x in list(bnames)]
    except Exception:
        pass

    return None

def _extract_model_feature_count(model: Any) -> Optional[int]:
    """Extract the expected number of input features (fallback when feature names are unavailable)."""
    if model is None:
        return None
    n = getattr(model, 'n_features_in_', None)
    if n is None:
        return None
    try:
        return int(n)
    except Exception:
        return None

def resolve_model_feature_columns(models: Dict[str, Any]) -> List[str]:
    """
    Resolve the actual feature columns required by the models.

    Priority:
    1) Use feature names stored in the model (feature_names_in_ / booster.feature_names)
    2) If feature names are unavailable, fall back to slicing the default feature list by n_features_in_
    """
    cached = models.get('__resolved_feature_columns__')
    if isinstance(cached, list) and len(cached) > 0:
        return cached

    model_order = ['clf', 'tm', 'dh']
    named_columns = {}
    for key in model_order:
        cols = _extract_model_feature_names(models.get(key))
        if cols:
            named_columns[key] = cols

    if named_columns:
        # Prefer the classifier's feature order; otherwise use the first available model
        ref_key = 'clf' if 'clf' in named_columns else next(iter(named_columns.keys()))
        resolved = named_columns[ref_key]
    else:
        # Fall back by feature count when names are unavailable
        fallback_cols = list(FEATURE_COLUMNS)
        n_features = None
        for key in model_order:
            n_features = _extract_model_feature_count(models.get(key))
            if n_features:
                break
        if n_features is None:
            resolved = fallback_cols
        else:
            resolved = fallback_cols[:n_features] if n_features <= len(fallback_cols) else fallback_cols

    models['__resolved_feature_columns__'] = resolved
    return resolved

# --- structured meta loading for compatibility ---
def _norm_formula_key(s: str) -> str:
    s = str(s).strip()
    s = s.replace('·', '·')
    s = s.replace(' ', '')
    return s

def _parse_charge(ion: str) -> Optional[int]:
    if not ion:
        return None
    m = re.search(r'([+-])\s*$', ion)
    if m:
        sign = -1 if m.group(1) == '-' else 1
        m2 = re.search(r'(\d+)[+-]\s*$', ion)
        val = int(m2.group(1)) if m2 else 1
        return sign * val
    m = re.search(r'(\d+)\s*([+-])', ion)
    if m:
        val = int(m.group(1))
        sign = -1 if m.group(2) == '-' else 1
        return sign * val
    return None

def _anion_is_polyatomic(anion: str) -> Optional[bool]:
    if not anion:
        return None
    base = re.sub(r'[\d\+\-]', '', anion)
    return len(base) > 2 or any(ch.islower() for ch in base[1:])

def _cation_symbol(cation: str) -> Optional[str]:
    if not cation:
        return None
    m = re.match(r'([A-Za-z]+)', cation)
    return m.group(1) if m else None

_GROUPS = {
    'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
    'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
    'Al': 13, 'Ga': 13,
    'Zn': 12, 'Cd': 12,
    'Fe': 8, 'Co': 9, 'Ni': 10, 'Mn': 7, 'Cr': 6, 'Cu': 11, 'Tb': 3, 'La': 3, 'Pr': 3, 'Bi': 15, 'Sn': 14
}

def _same_group(cat1: Optional[str], cat2: Optional[str]) -> Optional[bool]:
    if not cat1 or not cat2:
        return None
    return _GROUPS.get(cat1) == _GROUPS.get(cat2) if (cat1 in _GROUPS and cat2 in _GROUPS) else None

def _check_chemical_compatibility(m1: Dict[str, Any], m2: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check chemical compatibility: precipitation, strong hydrolysis, gas release, etc.

    Returns:
        Tuple[bool, str]: (compatible, incompatibility_reason)
    """
    cation1 = m1.get('cation', '')
    cation2 = m2.get('cation', '')
    anion1 = m1.get('anion', '')
    anion2 = m2.get('anion', '')

    # Check potential precipitation
    if _check_precipitation_risk(cation1, anion2) or _check_precipitation_risk(cation2, anion1):
        return False, 'precipitation_risk'

    # Check hydrolysis/acid-base reaction risk (based on the union of ions)
    if _check_hydrolysis_risk(cation1, anion1, cation2, anion2):
        return False, 'hydrolysis_risk'

    # Check gas release risk
    if _check_gas_release_risk(cation1, anion1, cation2, anion2):
        return False, 'gas_release_risk'

    return True, 'compatible'

def _check_precipitation_risk(cation: str, anion: str) -> bool:
    """Check precipitation risk."""
    if not cation or not anion:
        return False

    # Common insoluble combinations
    insoluble_combinations = {
        # Insoluble sulfates
        ('Ba', 'SO4'): True, ('Sr', 'SO4'): True, ('Ca', 'SO4'): True, ('Pb', 'SO4'): True,
        # Insoluble chlorides
        ('Ag', 'Cl'): True, ('Pb', 'Cl'): True, ('Hg', 'Cl'): True,
        # Insoluble carbonates (extended)
        ('Ca', 'CO3'): True, ('Ba', 'CO3'): True, ('Sr', 'CO3'): True, ('Pb', 'CO3'): True,
        ('Mg', 'CO3'): True, ('Mn', 'CO3'): True, ('Fe', 'CO3'): True, ('Co', 'CO3'): True,
        ('Ni', 'CO3'): True, ('Cu', 'CO3'): True, ('Zn', 'CO3'): True,
        # Insoluble phosphates / phosphorus-containing salts
        ('Ca', 'PO4'): True, ('Ba', 'PO4'): True, ('Al', 'PO4'): True, ('Fe', 'PO4'): True,
        ('Ca', 'HPO4'): True, ('Ba', 'HPO4'): True, ('Al', 'HPO4'): True, ('Fe', 'HPO4'): True,
        ('Ca', 'P2O7'): True, ('Ba', 'P2O7'): True, ('Al', 'P2O7'): True, ('Fe', 'P2O7'): True,
        # Insoluble hydroxides (except alkali metals and Ba)
        ('Al', 'OH'): True, ('Fe', 'OH'): True, ('Cr', 'OH'): True, ('Zn', 'OH'): True,
        ('Cu', 'OH'): True, ('Ni', 'OH'): True, ('Co', 'OH'): True, ('Mn', 'OH'): True,
        # Optional extension: fluorides
        ('Ca', 'F'): True, ('Mg', 'F'): True, ('Fe', 'F'): True, ('Co', 'F'): True,
        ('Ni', 'F'): True, ('Cu', 'F'): True, ('Zn', 'F'): True, ('Al', 'F'): True,
        # Optional extension: silicates
        ('Ca', 'SiO3'): True, ('Mg', 'SiO3'): True, ('Fe', 'SiO3'): True, ('Co', 'SiO3'): True,
        ('Ni', 'SiO3'): True, ('Cu', 'SiO3'): True, ('Zn', 'SiO3'): True, ('Al', 'SiO3'): True,
    }

    # Extract cation symbol
    cation_symbol = _cation_symbol(cation)
    # Extract anion group (compatible with notations like CO32- / CO3^2- / HPO4^2- / P2O7^4- / SiO3^2-)
    anion_text = str(anion).strip().replace('−', '-').replace(' ', '')
    if '^' in anion_text:
        anion_text = anion_text.split('^', 1)[0]
    anion_compact = re.sub(r'[\+\-]', '', anion_text)
    anion_upper = anion_compact.upper()
    if anion_upper.startswith('HPO4'):
        anion_base = 'HPO4'
    elif anion_upper.startswith('P2O7'):
        anion_base = 'P2O7'
    elif anion_upper.startswith('CO3') or anion_upper.startswith('HCO3'):
        anion_base = 'CO3'
    elif anion_upper.startswith('PO4'):
        anion_base = 'PO4'
    elif anion_upper.startswith('SO4'):
        anion_base = 'SO4'
    elif anion_upper.startswith('SIO3'):
        anion_base = 'SiO3'
    elif anion_upper.startswith('OH'):
        anion_base = 'OH'
    elif anion_upper.startswith('CL'):
        anion_base = 'Cl'
    elif anion_upper.startswith('F'):
        anion_base = 'F'
    else:
        anion_base = re.sub(r'[\d\+\-]', '', anion_compact)

    return insoluble_combinations.get((cation_symbol, anion_base), False)

def _check_hydrolysis_risk(cation1: str, anion1: str, cation2: str, anion2: str) -> bool:
    """Check hydrolysis/acid-base reaction risk (based on the union of ions)."""
    def _compact_cation(c: str) -> str:
        s = str(c or "").strip().replace("−", "-").replace(" ", "")
        # Normalize to the Al3+ style (compatible with Al+3 / Al^3+ / Al3+)
        s = s.replace("^", "")
        m = re.match(r"^([A-Za-z][A-Za-z0-9()]*)([+-])(\d+)$", s)
        if m:
            return f"{m.group(1)}{m.group(3)}{m.group(2)}"
        m = re.match(r"^([A-Za-z][A-Za-z0-9()]*)(\d+)([+-])$", s)
        if m:
            return f"{m.group(1)}{m.group(2)}{m.group(3)}"
        m = re.match(r"^([A-Za-z][A-Za-z0-9()]*)([+-])$", s)
        if m:
            return f"{m.group(1)}1{m.group(2)}"
        return s

    def _compact_anion(a: str) -> str:
        s = str(a or "").strip().replace("−", "-").replace(" ", "")
        # Normalize to the CO32- style (remove ^)
        return s.replace("^", "")

    cations = {_compact_cation(cation1), _compact_cation(cation2)}
    anions = {_compact_anion(anion1), _compact_anion(anion2)}

    # Strongly hydrolyzing / strongly acidifying cations
    strong_hydrolysis_cations = {'Al3+', 'Fe3+', 'Cr3+', 'Bi3+', 'Sn4+'}
    # Strongly basic / acid-base convertible anions
    strong_hydrolysis_anions = {
        'CO32-', 'HCO3-', 'PO43-', 'HPO42-',
        'S2-', 'SO32-', 'HSO3-', 'SiO32-'
    }

    # Reject only when the union of cations intersects the strong-hydrolysis set AND
    # the union of anions intersects the strong-basic anion set
    return bool(cations & strong_hydrolysis_cations) and bool(anions & strong_hydrolysis_anions)

def _check_gas_release_risk(cation1: str, anion1: str, cation2: str, anion2: str) -> bool:
    """Check gas release risk (based on the union of ions)."""
    # Examples:
    # - Carbonate + acidic salt → CO2
    # - Sulfide + acid → H2S
    # - Sulfite + acid → SO2

    def _compact_anion(a: str) -> str:
        s = str(a or "").strip().replace("−", "-").replace(" ", "")
        # Normalize to the CO32- style (remove ^)
        return s.replace("^", "")

    def _compact_cation(c: str) -> str:
        s = str(c or "").strip().replace("−", "-").replace(" ", "")
        # Normalize to the Al3+ style (compatible with Al+3 / Al^3+ / Al3+)
        s = s.replace("^", "")
        m = re.match(r"^([A-Za-z][A-Za-z0-9()]*)([+-])(\d+)$", s)
        if m:
            return f"{m.group(1)}{m.group(3)}{m.group(2)}"
        m = re.match(r"^([A-Za-z][A-Za-z0-9()]*)(\d+)([+-])$", s)
        if m:
            return f"{m.group(1)}{m.group(2)}{m.group(3)}"
        m = re.match(r"^([A-Za-z][A-Za-z0-9()]*)([+-])$", s)
        if m:
            return f"{m.group(1)}1{m.group(2)}"
        return s

    gas_producing_anions = {'CO32-', 'HCO3-', 'S2-', 'SO32-', 'HSO3-'}
    acidic_cations = {'Al3+', 'Fe3+', 'Cr3+', 'Bi3+', 'Sn4+'}

    # Simplified rule: one salt has a gas-producing anion, the other has an acidic cation
    anions = {_compact_anion(anion1), _compact_anion(anion2)}
    cations = {_compact_cation(cation1), _compact_cation(cation2)}

    return bool(anions & gas_producing_anions) and bool(cations & acidic_cations)

def _check_ion_compatibility(m1: Dict[str, Any], m2: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Check ion structural compatibility

    Returns:
        Tuple[Dict, List]: (compatibility_details, warnings)
    """
    compatibility = {}
    warnings = []

    # Cation compatibility
    cation1_symbol = m1.get('cation_symbol')
    cation2_symbol = m2.get('cation_symbol')

    if cation1_symbol and cation2_symbol:
        # 1) Identical or same-group cations
        if cation1_symbol == cation2_symbol:
            compatibility['cation_similarity'] = 'identical'
        elif _same_group(cation1_symbol, cation2_symbol):
            compatibility['cation_similarity'] = 'same_group'
        else:
            compatibility['cation_similarity'] = 'different'

        # 2) Cation radius difference
        r1, r2 = m1.get('cation_radius'), m2.get('cation_radius')
        if r1 is not None and r2 is not None:
            radius_diff_percent = abs(r1 - r2) / ((r1 + r2) / 2.0)
            compatibility['cation_radius_diff_percent'] = radius_diff_percent * 100

            if radius_diff_percent < 0.15:
                compatibility['cation_radius_compatibility'] = 'good'
            else:
                compatibility['cation_radius_compatibility'] = 'reduced'
                warnings.append(f'Large cation radius difference ({radius_diff_percent*100:.1f}%), reduced compatibility')
        else:
            compatibility['cation_radius_compatibility'] = 'unknown'
            warnings.append('Missing cation radius data')

        # 3) Possibility of cation lattice-site alternation
        if compatibility['cation_similarity'] in ['identical', 'same_group']:
            compatibility['cation_lattice_substitution'] = 'supported'
        elif compatibility.get('cation_radius_compatibility') == 'good':
            compatibility['cation_lattice_substitution'] = 'possible'
        else:
            compatibility['cation_lattice_substitution'] = 'needs_verification'
            warnings.append('Cation lattice site alternation needs verification')
    else:
        compatibility['cation_similarity'] = 'unknown'
        warnings.append('Missing cation data')

    # Anion compatibility
    anion1_charge = m1.get('anion_charge')
    anion2_charge = m2.get('anion_charge')
    anion1_poly = m1.get('anion_poly')
    anion2_poly = m2.get('anion_poly')

    # 1) Anion charge consistency
    if anion1_charge is not None and anion2_charge is not None:
        if anion1_charge == anion2_charge:
            compatibility['anion_charge_compatibility'] = 'ideal'
        else:
            compatibility['anion_charge_compatibility'] = 'reduced'
            warnings.append(f'Anion charge mismatch ({anion1_charge} vs {anion2_charge}), reduced compatibility')
    else:
        compatibility['anion_charge_compatibility'] = 'unknown'
        warnings.append('Missing anion charge data')

    # 2) Anion structural type (monoatomic vs polyatomic)
    if anion1_poly is not None and anion2_poly is not None:
        if anion1_poly == anion2_poly:
            if anion1_poly:
                compatibility['anion_structure_compatibility'] = 'ideal_polyatomic'
            else:
                compatibility['anion_structure_compatibility'] = 'ideal_monoatomic'
        else:
            compatibility['anion_structure_compatibility'] = 'reduced'
            warnings.append('One monoatomic, one polyatomic anion, reduced compatibility')
    else:
        compatibility['anion_structure_compatibility'] = 'unknown'
        warnings.append('Missing anion structural data')

    return compatibility, warnings

def load_structured_meta(data_dir: str = 'data') -> Dict[str, Dict[str, Any]]:
    meta = {}
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
            meta[key] = {
                'formula': formula,
                'Tm': float(therm.get('Melting_temperature_C')) if therm.get('Melting_temperature_C') is not None else None,
                'dH': float(therm.get('Enthalpy_value_J_per_g')) if therm.get('Enthalpy_value_J_per_g') is not None else None,
                'nH2O': feat.get('Hydration_Number'),
                'cation': feat.get('Cation'),
                'cation_symbol': _cation_symbol(feat.get('Cation')),
                'cation_radius': feat.get('Cation_Radius_Angstrom'),
                'anion': feat.get('Anion'),
                'anion_charge': _parse_charge(feat.get('Anion') or ''),
                'anion_poly': _anion_is_polyatomic(feat.get('Anion') or '')
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
        except Exception:
            continue
    return meta

def match_meta(meta_map: Dict[str, Dict[str, Any]], salt_name: str) -> Optional[Dict[str, Any]]:
    key = _norm_formula_key(salt_name)
    if key in meta_map:
        return meta_map[key]
    key2 = key.replace('·', '.')
    for k in meta_map.keys():
        if _norm_formula_key(k) == key or _norm_formula_key(k).replace('·','.') == key2:
            return meta_map[k]
    return None

def compatibility_decision(m1: Dict[str, Any], m2: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluate compatibility of two salts based on eutectic screening criteria.

    Dimensions:
    1) Thermodynamic compatibility (ΔTm)
    2) Chemical compatibility (precipitation/hydrolysis/gas reaction)
    3) Hydration matching (Δn(H2O))
    4) Ion structural compatibility (cation + anion)

    Returns:
        Tuple[bool, Dict]: (compatible, detailed_reasons_and_score)
    """
    reasons = {}
    compatibility_score = 0
    warnings = []

    # Check required data availability
    if m1.get('Tm') is None or m2.get('Tm') is None:
        return False, {'reason': 'Missing Tm data', 'decision': 'reject_missing_data'}

    # 1) Thermodynamic compatibility (ΔTm)
    dTm = abs(m1['Tm'] - m2['Tm'])
    reasons['ΔTm'] = dTm

    if dTm < 30.0:
        # Ideal (+2)
        compatibility_score += 2
        reasons['thermal_compatibility'] = 'ideal'
    elif dTm <= 100.0:
        # Acceptable (+1)
        compatibility_score += 1
        reasons['thermal_compatibility'] = 'acceptable'
    else:
        # Reject
        return False, {**reasons, 'decision': 'reject_ΔTm>100', 'thermal_compatibility': 'rejected'}

    # 2) Chemical compatibility
    chemical_compatible, chemical_reason = _check_chemical_compatibility(m1, m2)
    if not chemical_compatible:
        return False, {**reasons, 'decision': f'reject_chemical_{chemical_reason}', 'chemical_compatibility': 'rejected'}
    reasons['chemical_compatibility'] = 'compatible'

    # 3) Hydration matching (Δn(H2O))
    n1, n2 = m1.get('nH2O'), m2.get('nH2O')
    if n1 is not None and n2 is not None:
        dn = abs(n1 - n2)
        reasons['Δn(H2O)'] = dn

        if dn <= 2:
            # Ideal
            reasons['hydration_compatibility'] = 'ideal'
        elif dn <= 5:
            # Acceptable
            reasons['hydration_compatibility'] = 'acceptable'
        else:
            # Not recommended (treated as reduced compatibility rather than exclusion)
            reasons['hydration_compatibility'] = 'reduced'
            warnings.append('Large hydration water difference, reduced compatibility')
    else:
        reasons['hydration_compatibility'] = 'unknown'
        warnings.append('Missing hydration water data')

    # 4) Ion structural compatibility
    ion_compatibility, ion_warnings = _check_ion_compatibility(m1, m2)
    reasons.update(ion_compatibility)
    warnings.extend(ion_warnings)

    # Summarize
    reasons['compatibility_score'] = compatibility_score
    reasons['warnings'] = warnings
    reasons['decision'] = 'compatible' if compatibility_score > 0 else 'marginal'

    return True, reasons

def _row_to_component_features(s: pd.Series) -> List[float]:
    out = []
    for feat in BASE_FEATURES:
        out.append(float(s[feat]) if feat in s.index and pd.notna(s[feat]) else 0.0)
    return out

def build_feature_row(samples: List[pd.Series], ratios: List[float]) -> Dict[str, Any]:
    # samples: a row list like [A, B] or [A, B, C]
    # ratios: ratio list like [r1, r2] or [r1, r2, r3]
    data = {}
    data['row_id'] = samples[0].name if samples[0].name is not None else 0

    # Components 1/2/3
    for idx in range(3):
        if idx < len(samples):
            feats = _row_to_component_features(samples[idx])
        else:
            feats = [0.0] * len(BASE_FEATURES)
        for j, feat in enumerate(BASE_FEATURES):
            data[f"{idx+1}-{feat}"] = feats[j]

    r1 = round(ratios[0], 2) if len(ratios) > 0 else 0.0
    r2 = round(ratios[1], 2) if len(ratios) > 1 else 0.0
    r3 = round(ratios[2], 2) if len(ratios) > 2 else 0.0
    data['ID1'] = r1
    data['ID2'] = r2
    data['ID3'] = r3
    return data

def load_models() -> Dict[str, Any]:
    # M1 models are stored in ../models
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(base_dir, "..", "models"))
    clf_path = os.path.join(model_dir, "GBDT_best_model.pkl")
    dh_path = os.path.join(model_dir, "XGBoost_best_model.pkl")
    tm_path = os.path.join(model_dir, "RF_best_model.pkl")
    models = {}
    if os.path.exists(clf_path):
        models['clf'] = joblib.load(clf_path)
    else:
        raise FileNotFoundError(f"Single peak model not found: {clf_path}")
    if os.path.exists(dh_path):
        models['dh'] = joblib.load(dh_path)
    else:
        raise FileNotFoundError(f"Enthalpy model not found: {dh_path}")
    if os.path.exists(tm_path):
        models['tm'] = joblib.load(tm_path)
    else:
        raise FileNotFoundError(f"Melting point model not found: {tm_path}")
    return models

def predict_with_models(models: Dict[str, Any], feat_row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Use model-resolved feature columns to avoid train/predict mismatch errors
        model_feature_cols = resolve_model_feature_columns(models)
        x = pd.DataFrame([feat_row]).reindex(columns=model_feature_cols, fill_value=0.0)
        X = x.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # Data integrity check
        if X.isnull().any().any():
            print("Warning: Feature data contains NaN, filling with 0")
            X = X.fillna(0)

        res = {}
        clf = models['clf']

        # Single-peak prediction
        pred = clf.predict(X)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[:, 1]
        else:
            # Fallback probability placeholder
            proba = np.array([0.5])

        res['is_single_peak'] = int(pred[0])
        res['single_peak_proba'] = float(proba[0])

        # Only predict Tm and ΔH for single-peak samples
        if res['is_single_peak'] == 1:
            try:
                res['tm_pred'] = float(models['tm'].predict(X)[0])
                res['dh_pred'] = float(models['dh'].predict(X)[0])
            except Exception as e:
                print(f"Melting point/enthalpy prediction failed: {e}")
                res['tm_pred'] = np.nan
                res['dh_pred'] = np.nan
        else:
            # Do not predict Tm and ΔH for non-single-peak samples
            res['tm_pred'] = np.nan
            res['dh_pred'] = np.nan

        return res

    except Exception as e:
        print(f"Model prediction failed: {e}")
        # Return default values
        return {
            'is_single_peak': 0,
            'single_peak_proba': 0.0,
            'tm_pred': np.nan,
            'dh_pred': np.nan
        }

# === Agent search additions end ===

