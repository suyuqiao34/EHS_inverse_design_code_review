"""
Multi-agent salt screening system
Implements multi-agent collaborative screening based on DashScope API
"""

import pandas as pd
import numpy as np
import json
import logging
import os
import re
from typing import Dict, List, Tuple, Any, Optional, Set
from dashscope import Generation
import time

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_base_salt_type(salt_name: str) -> str:
    """
    Extract base salt type (remove hydration info)
    
    Examples:
    - "NaCl·2H2O" -> "NaCl"
    - "CaCl2·4H2O" -> "CaCl2"
    - "NaOH·H2O" -> "NaOH"
    - "NaCl" -> "NaCl"
    
    Args:
        salt_name: Salt name, may contain hydration info
        
    Returns:
        Base salt type name (without hydration part)
    """
    if not salt_name:
        return salt_name
    
    salt_name = str(salt_name).strip()
    
    # Match hydration patterns: ·nH2O, .nH2O, nH2O etc. (n can be digit or empty)
    # Supports multiple formats:
    # - ·2H2O, ·4H2O, ·H2O (middle dot)
    # - .2H2O, .4H2O, .H2O (period)
    # - 2H2O, 4H2O, H2O (direct concatenation, less common)
    
    # Try matching ·nH2O or .nH2O format first (most common)
    # Pattern: · or . + optional digits + H2O (at end of string)
    pattern1 = r'[·.](?:\d+)?H2O$'
    base_type = re.sub(pattern1, '', salt_name, flags=re.IGNORECASE)
    
    # If no match, try matching trailing nH2O format (less common, direct concatenation)
    if base_type == salt_name:
        pattern2 = r'(?:\d+)?H2O$'
        base_type = re.sub(pattern2, '', salt_name, flags=re.IGNORECASE)
    
    # Remove trailing spaces, periods, or middle dots
    base_type = base_type.rstrip()
    # Remove trailing middle dot (U+00B7) and period
    while base_type and (base_type[-1] in ['·', '.', '\u00B7']):
        base_type = base_type[:-1]
    base_type = base_type.rstrip()
    
    return base_type if base_type else salt_name

def are_same_salt_type(salt1: str, salt2: str) -> bool:
    """
    Check if two salts are same type (ignoring hydration)
    
    Args:
        salt1: First salt name
        salt2: Second salt name
        
    Returns:
        True if same salt type (only hydration differs), False if different types
    """
    base1 = extract_base_salt_type(salt1)
    base2 = extract_base_salt_type(salt2)
    
    # Normalized comparison (remove spaces, unify case)
    base1_norm = base1.strip().replace(' ', '').replace('_', '')
    base2_norm = base2.strip().replace(' ', '').replace('_', '')
    
    return base1_norm.lower() == base2_norm.lower()


def _infer_ions_from_salt_name(salt_name: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Conservative inference from salt name when structured data lacks ion info.
    Returns: (cation, anion, anion_charge)
    """
    if not salt_name:
        return None, None, None

    name = str(salt_name).strip()
    core = re.split(r"[·\.]", name)[0]
    core = core.replace(" ", "")

    anion_specs = [
        ("CH3COO", ("CH3COO^-", -1)),
        ("P2O7", ("P2O7^4-", -4)),
        ("H2PO4", ("H2PO4^-", -1)),
        ("HPO4", ("HPO4^2-", -2)),
        ("PO4", ("PO4^3-", -3)),
        ("S2O3", ("S2O3^2-", -2)),
        ("SO4", ("SO4^2-", -2)),
        ("CO3", ("CO3^2-", -2)),
        ("SiO3", ("SiO3^2-", -2)),
        ("NO3", ("NO3^-", -1)),
        ("ClO3", ("ClO3^-", -1)),
        ("OH", ("OH^-", -1)),
        ("Cl", ("Cl^-", -1)),
        ("Br", ("Br^-", -1)),
        ("I", ("I^-", -1)),
        ("F", ("F^-", -1)),
    ]

    cation_candidates = [
        "Na", "K", "Li", "Ca", "Mg", "Mn", "Fe", "Co", "Ni", "Cr", "Zn",
        "Cd", "La", "Bi", "Ba", "Sr", "Al", "Cu", "Sn", "Ag", "Pb", "Rb", "Cs"
    ]
    cation_guess = None
    for cat in cation_candidates:
        if core.startswith(cat):
            cation_guess = cat
            break
    if cation_guess is None:
        for cat in cation_candidates:
            if core.endswith(cat):
                cation_guess = cat
                break

    anion_guess = None
    anion_charge_guess = None
    for token, (anion_label, charge) in anion_specs:
        if token in core:
            anion_guess = anion_label
            anion_charge_guess = charge
            break

    if cation_guess:
        cation_guess = f"{cation_guess}+"

    return cation_guess, anion_guess, anion_charge_guess


def _resolve_meta_with_data_priority(
    meta_map: Dict[str, Dict[str, Any]],
    salt_name: str,
    row: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Unified metadata source rules:
    1) Prioritize ion fields from structured metadata under data/;
    2) Only perform automatic inference when data lacks ion fields.
    """
    from salt_combination_generator import match_meta

    raw = match_meta(meta_map, salt_name) if meta_map else None
    meta = dict(raw) if raw else {}
    meta["formula"] = meta.get("formula") or salt_name

    if "Tm" not in meta or meta.get("Tm") in (None, ""):
        meta["Tm"] = row.get("Tm (°C)", None) if row is not None else None
    if "nH2O" not in meta or meta.get("nH2O") in (None, ""):
        meta["nH2O"] = row.get("Hydration Number", None) if row is not None else None

    has_data_cation = meta.get("cation") not in (None, "")
    has_data_anion = meta.get("anion") not in (None, "")
    has_data_anion_charge = meta.get("anion_charge") not in (None, "")

    if not (has_data_cation and has_data_anion and has_data_anion_charge):
        inf_cation, inf_anion, inf_anion_charge = _infer_ions_from_salt_name(salt_name)
        if not has_data_cation and inf_cation:
            meta["cation"] = inf_cation
        if not has_data_anion and inf_anion:
            meta["anion"] = inf_anion
        if not has_data_anion_charge and inf_anion_charge is not None:
            meta["anion_charge"] = inf_anion_charge

    return meta


def call_dashscope_agent(prompt: str, system_role: str = "You are a professional phase change material screening assistant") -> str:
    """Call DashScope API"""
    if not DASHSCOPE_API_KEY:
        logger.error("DASHSCOPE_API_KEY is not set. Please configure it as an environment variable.")
        return None

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = Generation.call(
                api_key=DASHSCOPE_API_KEY,
                model='qwen-plus',
                prompt=prompt,
                system=system_role
            )

            if response.status_code == 200:
                return response.output.text

            logger.error(f"DashScope API call failed (attempt {attempt}/{max_retries}): {response.code} - {response.message}")
        except Exception as e:
            logger.error(f"DashScope API call exception (attempt {attempt}/{max_retries}): {e}")

        if attempt < max_retries:
            # Exponential backoff for transient network/SSL issues
            time.sleep(1.5 ** attempt)

    return None


HARD_FILTER_SYSTEM_PROMPT = """You are a "Eutectic Salt Combination Compatibility Hard-filter Expert (Generalized Robust Version)".

Task:
Perform hard-filtering on input salt combinations. The goal is to eliminate systems with "high probability of irreversible incompatibility under default experimental conditions", while avoiding false rejection of systems that "only involve reversible equilibria with no clear irreversible driving force".

Default Experimental Conditions:
- Atmospheric pressure
- Contains crystal water and small amount of free water
- 20-150°C (locally may be higher)
- High ionic strength, concentrated salt environment
- No externally added strong acid/base/complexing agent/redox agent
- Does not assume open system with continuous removal of gas or products

Evidence Standards (must comply):
1) REJECT only when a "clear, self-consistent, actionable" main reaction pathway exists with realistic driving force.
2) Must not use "may/perhaps/speculation" as sole basis; must not fabricate database values or constants.
3) If only reversible acid-base equilibrium, coordination rearrangement, or ion association exists, and does not point to new phase/gas release/decomposition, must not REJECT.
4) For charge, prioritize `anion_charge` from input JSON; do not second-guess.

Hard Veto Triggers (REJECT if any is matched):
- ΔTm > 100°C
- Clear precipitation/metathesis (must identify specific new phase)
- Strong hydrolysis or significant acid-base reaction leading to stable new phase
- Clear gas release/decomposition (must have sustained net gas-producing driving force)
- Clear redox incompatibility
- In ternary systems, any sub-pair (AB/AC/BC) is incompatible, or new incompatibility pathway emerges when all three coexist

Anti-False-Kill Constraints (universal):
A) Must not equate "reversible proton transfer/buffer equilibrium" directly with hard incompatibility.
B) Must not declare "possible gas production" as hard veto without strong driving force evidence.
C) Precipitation conclusion must satisfy three elements: specific phase + formation pathway + driving force explanation.
D) For ternary, first evaluate AB/AC/BC; REJECT only when sub-pair is incompatible or new hard triggers emerge from all three.
E) For mild inorganic salt systems of similar type, with no polyvalent metal precipitation risk, no redox pair, and no strong acid/base addition, default to PASS.
F) For anion-anion acid-base equilibria, without externally added strong acid, no net acid source, and no continuous degassing boundary conditions, must not use "possible CO2 production" as REJECT basis.
G) For coexistence of amphoteric anions and weakly basic anions, if only reversible proton transfer equilibrium is involved with no clear new phase/net gas release/decomposition pathway, must determine as PASS (or continue gathering evidence, but must not REJECT based on speculation).
H) If REJECT with reason_code=gas_or_decomposition, reason must specify "reactant source and boundary conditions for net gas production"; both are required, otherwise the conclusion is invalid and overridden to PASS.

Judgment Rules:
- No hard veto trigger matched => PASS
- Any hard veto trigger matched => REJECT

Output Format (strict):
{
  "decision": "PASS | REJECT",
  "reason_code": "compatible | missing_critical_data | dTm_too_large | precipitation | hydrolysis_or_acid_base | gas_or_decomposition | redox | ternary_cross_reaction | llm_unavailable | invalid_llm_output",
  "reason": "One-sentence reason (including trigger and evidence)"
}

Strict Requirements:
- Output only one JSON object, no additional text.
- If REJECT, reason must specify "trigger + minimum sufficient evidence"."""


def _clean_for_llm(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, str, bool)):
        return value
    if isinstance(value, float):
        return None if pd.isna(value) else value
    try:
        return None if pd.isna(value) else value
    except Exception:
        return value


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = raw.find('{')
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = raw[start:idx + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return None
    return None


def _normalize_salt_meta_for_llm(meta: Optional[Dict[str, Any]], fallback_name: str = "") -> Dict[str, Any]:
    m = meta or {}
    return {
        "name": _clean_for_llm(m.get("formula") or m.get("name") or fallback_name),
        "Tm": _clean_for_llm(m.get("Tm")),
        "nH2O": _clean_for_llm(m.get("nH2O")),
        "cation": _clean_for_llm(m.get("cation")),
        "anion": _clean_for_llm(m.get("anion")),
        "anion_charge": _clean_for_llm(m.get("anion_charge"))
    }


def evaluate_compatibility_hard_filter(
    salt_a_meta: Optional[Dict[str, Any]],
    salt_b_meta: Optional[Dict[str, Any]],
    salt_c_meta: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[bool, Dict[str, Any]]:
    MAX_REASON_LEN = 180

    valid_reason_codes = {
        "compatible",
        "missing_critical_data",
        "dtm_too_large",
        "precipitation",
        "hydrolysis_or_acid_base",
        "gas_or_decomposition",
        "redox",
        "ternary_cross_reaction",
        "llm_unavailable",
        "invalid_llm_output",
    }
    decision_aliases = {
        "PASS": "PASS",
        "REJECT": "REJECT",
        "COMPATIBLE": "PASS",
        "INCOMPATIBLE": "REJECT",
    }

    def _infer_reason_code_from_text(text: str) -> str:
        t = (text or "").lower()
        if ("dtm" in t and ("100" in t or "too large" in t)):
            return "dtm_too_large"
        if "precip" in t or "insoluble" in t:
            return "precipitation"
        if "hydrolysis" in t or "acid" in t or "base" in t:
            return "hydrolysis_or_acid_base"
        if "gas" in t or "decomposition" in t:
            return "gas_or_decomposition"
        if "redox" in t:
            return "redox"
        if "ternary" in t or "ab/ac/bc" in t:
            return "ternary_cross_reaction"
        if "charge mismatch" in t:
            return "missing_critical_data"
        if "missing" in t or "insufficient" in t:
            return "missing_critical_data"
        return "missing_critical_data"

    def _list_missing_critical_fields(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
        missing = []
        for k in ("Tm", "cation", "anion", "anion_charge"):
            if a.get(k) in (None, ""):
                missing.append(f"A.{k}")
            if b.get(k) in (None, ""):
                missing.append(f"B.{k}")
        return missing

    def _clip_reason_text(text: Any, max_len: int = MAX_REASON_LEN) -> str:
        s = str(text or "").strip()
        if len(s) <= max_len:
            return s
        return s[:max_len].rstrip() + "...(truncated)"

    def _has_minimum_critical_data(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        required_keys = ("Tm", "cation", "anion", "anion_charge")
        for k in required_keys:
            av = a.get(k)
            bv = b.get(k)
            if av in (None, "") or bv in (None, ""):
                return False
        return True

    def _has_required_gas_evidence(reason_text: str) -> bool:
        t = (reason_text or "").lower()
        source_keywords = (
            "strong acid", "acid source", "net acid", "external acid", "proton source"
        )
        boundary_keywords = (
            "open system", "degassing", "vent", "continuous removal", "boundary condition"
        )
        has_source = any(k in t for k in source_keywords)
        has_boundary = any(k in t for k in boundary_keywords)
        return has_source and has_boundary

    def _is_speculative_reject(reason_text: str) -> bool:
        t = (reason_text or "").lower()
        speculative_keywords = (
            "may ", "might ", "prone to", "possible", "possibly", "lack documented", "no documented"
        )
        evidence_keywords = (
            "insoluble", "precipitate", "ksp",
            "strong acid", "acid source", "net acid", "external acid", "open system", "degassing",
            "decomposition temperature", "redox pair"
        )
        has_speculative = any(k in t for k in speculative_keywords)
        has_evidence = any(k in t for k in evidence_keywords)
        return has_speculative and not has_evidence

    def _has_concrete_precipitate_evidence(reason_text: str) -> bool:
        """
        Minimum evidence for precipitation-based rejection:
        must include both a specific precipitate phase (not a generic term)
        and a formation pathway/basis.
        Generic statements (possible/less soluble/complex formation/species)
        alone do not count as valid evidence.
        """
        t = (reason_text or "").lower()
        generic_clues = (
            "mixed salts", "possible", "possibly", "may ", "might ",
            "less soluble", "complex formation", "species", "potential precipitation"
        )
        path_or_basis = (
            "ksp", "solubility", "insoluble", "driven by", "reaction pathway"
        )
        # Count only neutral, specific product phases as formula evidence;
        # exclude charged-ion-only forms (e.g., x2-/x-).
        has_neutral_formula = bool(re.search(r"\b[a-z][a-z0-9()]*\d[a-z0-9()]*\b(?!\s*[+-])", t))
        has_charged_ion_only = bool(re.search(r"\b[a-z0-9()]+[+-]\b", t)) and not has_neutral_formula
        has_basis = any(k in t for k in path_or_basis)
        # Stronger evidence: requires explicit solid-phase marker or Ksp-level basis,
        # avoiding false rejection from generic statements.
        phase_markers = ("(s)", " solid ", "solid phase", "precipitated phase")
        has_phase_marker = any(k in t for k in phase_markers)
        has_ksp_like = any(k in t for k in ("ksp", "solubility product"))
        only_generic = any(k in t for k in generic_clues)
        if has_charged_ion_only:
            return False
        return has_neutral_formula and has_basis and (has_phase_marker or has_ksp_like) and not only_generic

    def _apply_antifalse_reject_guard(res: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(res, dict):
            return res
        if res.get("decision") != "REJECT":
            return res

        reason_code = str(res.get("reason_code", "")).strip().lower()
        reason = str(res.get("reason", ""))
        t = reason.lower()
        co2_related = any(k in t for k in ("co2", "carbon dioxide", "hco3", "co3", "hpo4", "po4"))

        # Unified gas evidence threshold: any reason involving CO2/gas claims must provide minimum evidence regardless of reason_code
        if co2_related:
            if not _has_required_gas_evidence(reason):
                return {
                    "decision": "PASS",
                    "reason_code": "compatible",
                    "reason": "Anti-false-kill guard: Gas-related REJECT lacks minimum evidence of net gas source plus boundary conditions, overriding to PASS per rules"
                }

        if reason_code in {"ternary_cross_reaction", "precipitation", "hydrolysis_or_acid_base", "gas_or_decomposition"}:
            if _is_speculative_reject(reason):
                return {
                    "decision": "PASS",
                    "reason_code": "compatible",
                    "reason": "Anti-false-kill guard: REJECT based on speculative statements without minimum evidence (e.g., possible phase separation or lack of literature), overriding to PASS per rules"
                }

        if reason_code == "precipitation":
            if not _has_concrete_precipitate_evidence(reason):
                return {
                    "decision": "PASS",
                    "reason_code": "compatible",
                    "reason": "Anti-false-kill guard: Precipitation REJECT lacks evidence of a specific precipitate phase plus formation pathway, overriding to PASS per rules"
                }

        if _has_minimum_critical_data(a_norm, b_norm):
            data_error_clues = (
                "anion_charge", "charge mismatch", "incorrectly specified", "invalid data", "missing"
            )
            if reason_code == "missing_critical_data" or any(k in t for k in data_error_clues):
                return {
                    "decision": "PASS",
                    "reason_code": "compatible",
                    "reason": "Anti-false-kill guard: A/B critical fields complete, REJECT reason is data error/missing, overriding to PASS per rules"
                }

        pseudo_rule_clues = (
            "at least two distinct cations", "cation diversity", "distinct ion pairing"
        )
        if any(k in t for k in pseudo_rule_clues):
            return {
                "decision": "PASS",
                "reason_code": "compatible",
                "reason": "Anti-false-kill guard: REJECT used pseudo-conditions not defined in the rules (e.g., requiring at least two cation types), overriding to PASS per rules"
            }

        malformed_charge_clues = ("-42", "-32", "-43", "charge -42", "charge -32", "charge -43")
        if any(k in t for k in malformed_charge_clues):
            return {
                "decision": "PASS",
                "reason_code": "compatible",
                "reason": "Anti-false-kill guard: Detected charge concatenation/format hallucination (e.g. -42/-32), REJECT invalid, overriding to PASS"
            }
        return res

    def _fallback_when_llm_invalid(a: Dict[str, Any], b: Dict[str, Any], c: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        def _tm(v: Dict[str, Any]) -> Optional[float]:
            try:
                x = v.get("Tm")
                if x in (None, ""):
                    return None
                return float(x)
            except Exception:
                return None

        def _canonical_ion_notation(ion: Any) -> str:
            s = str(ion or "").strip()
            if not s:
                return s
            s = s.replace("−", "-")
            if "^" in s:
                return s
            m = re.match(r"^([A-Za-z0-9()]+?)(\d+)([+-])$", s)
            if m:
                return f"{m.group(1)}^{m.group(2)}{m.group(3)}"
            m2 = re.match(r"^([A-Za-z0-9()]+?)([+-])$", s)
            if m2:
                return f"{m2.group(1)}^{m2.group(2)}"
            return s

        def _legacy_chemical_veto(m1: Dict[str, Any], m2: Dict[str, Any]) -> Optional[Tuple[str, str]]:
            try:
                from salt_combination_generator import _check_chemical_compatibility
            except Exception:
                return None

            mm1 = {"cation": str(m1.get("cation", "") or ""), "anion": _canonical_ion_notation(m1.get("anion", ""))}
            mm2 = {"cation": str(m2.get("cation", "") or ""), "anion": _canonical_ion_notation(m2.get("anion", ""))}
            ok, why = _check_chemical_compatibility(mm1, mm2)
            if ok:
                return None
            mapping = {
                "precipitation_risk": ("precipitation", "Legacy rule fallback: precipitation risk"),
                "hydrolysis_risk": ("hydrolysis_or_acid_base", "Legacy rule fallback: strong hydrolysis/acid-base risk"),
                "gas_release_risk": ("gas_or_decomposition", "Legacy rule fallback: gas release risk"),
            }
            return mapping.get(str(why), ("hydrolysis_or_acid_base", f"Legacy rule fallback: {why}"))

        if not _has_minimum_critical_data(a, b):
            missing_fields = _list_missing_critical_fields(a, b)
            return {
                "decision": "REJECT",
                "reason_code": "missing_critical_data",
                "reason": f"LLM output invalid, critical fields insufficient, fallback cannot determine. Missing: {', '.join(missing_fields) if missing_fields else 'unknown'}"
            }

        tm_a, tm_b = _tm(a), _tm(b)
        if tm_a is not None and tm_b is not None and abs(tm_a - tm_b) > 100.0:
            return {
                "decision": "REJECT",
                "reason_code": "dtm_too_large",
                "reason": f"LLM output invalid, fallback matched deterministic hard trigger: ΔTm={abs(tm_a - tm_b):.2f}>100"
            }

        ab_veto = _legacy_chemical_veto(a, b)
        if ab_veto is not None:
            code, text = ab_veto
            return {"decision": "REJECT", "reason_code": code, "reason": f"LLM output invalid, {text}(AB)"}

        if c is not None:
            tm_c = _tm(c)
            if tm_a is not None and tm_c is not None and abs(tm_a - tm_c) > 100.0:
                return {
                    "decision": "REJECT",
                    "reason_code": "ternary_cross_reaction",
                    "reason": f"LLM output invalid, fallback matched ternary sub-pair hard trigger: ΔTm(AC)={abs(tm_a - tm_c):.2f}>100"
                }
            if tm_b is not None and tm_c is not None and abs(tm_b - tm_c) > 100.0:
                return {
                    "decision": "REJECT",
                    "reason_code": "ternary_cross_reaction",
                    "reason": f"LLM output invalid, fallback matched ternary sub-pair hard trigger: ΔTm(BC)={abs(tm_b - tm_c):.2f}>100"
                }

            ac_veto = _legacy_chemical_veto(a, c)
            if ac_veto is not None:
                code, text = ac_veto
                return {"decision": "REJECT", "reason_code": "ternary_cross_reaction", "reason": f"LLM output invalid, {text}(AC)"}
            bc_veto = _legacy_chemical_veto(b, c)
            if bc_veto is not None:
                code, text = bc_veto
                return {"decision": "REJECT", "reason_code": "ternary_cross_reaction", "reason": f"LLM output invalid, {text}(BC)"}

        return {
            "decision": "PASS",
            "reason_code": "compatible",
            "reason": "LLM output invalid, deterministic rule fallback: critical fields complete and no hard trigger (ΔTm>100) matched, tentatively compatible"
        }

    a_norm = _normalize_salt_meta_for_llm(salt_a_meta, "A")
    b_norm = _normalize_salt_meta_for_llm(salt_b_meta, "B")
    c_norm = _normalize_salt_meta_for_llm(salt_c_meta, "C") if salt_c_meta else None

    def _canonical_ion_notation_for_legacy(ion: Any) -> str:
        s = str(ion or "").strip()
        if not s:
            return s
        s = s.replace("−", "-")
        if "^" in s:
            return s
        m = re.match(r"^([A-Za-z0-9()]+?)(\d+)([+-])$", s)
        if m:
            return f"{m.group(1)}^{m.group(2)}{m.group(3)}"
        m2 = re.match(r"^([A-Za-z0-9()]+?)([+-])$", s)
        if m2:
            return f"{m2.group(1)}^{m2.group(2)}"
        return s

    def _legacy_gate_reject(a: Dict[str, Any], b: Dict[str, Any], c: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Run legacy rules first; reject immediately if matched."""
        try:
            from salt_combination_generator import _check_chemical_compatibility
        except Exception:
            return None

        def _pair_reject(m1: Dict[str, Any], m2: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
            mm1 = {
                "cation": str(m1.get("cation", "") or ""),
                "anion": _canonical_ion_notation_for_legacy(m1.get("anion", ""))
            }
            mm2 = {
                "cation": str(m2.get("cation", "") or ""),
                "anion": _canonical_ion_notation_for_legacy(m2.get("anion", ""))
            }
            ok, why = _check_chemical_compatibility(mm1, mm2)
            if ok:
                return None
            mapping = {
                "precipitation_risk": ("precipitation", "Legacy rule hit: precipitation risk"),
                "hydrolysis_risk": ("hydrolysis_or_acid_base", "Legacy rule hit: strong hydrolysis/acid-base risk"),
                "gas_release_risk": ("gas_or_decomposition", "Legacy rule hit: gas release risk"),
            }
            code, text = mapping.get(str(why), ("hydrolysis_or_acid_base", f"Legacy rule hit: {why}"))
            return {
                "decision": "REJECT",
                "reason_code": code,
                "reason": f"{text}({label})"
            }

        ab_reject = _pair_reject(a, b, "AB")
        if ab_reject is not None:
            return ab_reject
        if c is not None:
            ac_reject = _pair_reject(a, c, "AC")
            if ac_reject is not None:
                return ac_reject
            bc_reject = _pair_reject(b, c, "BC")
            if bc_reject is not None:
                return bc_reject
        return None

    payload = {
        "mode": "ternary" if c_norm else "binary",
        "salt_A": a_norm,
        "salt_B": b_norm,
        "salt_C": c_norm,
        "experimental_context": {
            "pressure": "1atm",
            "water_presence": "crystal_water_plus_trace_free_water",
            "temperature_range_C": "20-100_or_locally_higher",
            "ionic_strength": "high",
            "extra_buffer_or_ligand": False
        }
    }

    cache_key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if cache is not None and cache_key in cache:
        result = cache[cache_key]
        return result.get("decision") == "PASS", result

    # New flow: legacy rules first, then LLM for blind spots if passed
    legacy_reject = _legacy_gate_reject(a_norm, b_norm, c_norm)
    if legacy_reject is not None:
        result = legacy_reject
        if cache is not None:
            cache[cache_key] = result
        return False, result

    user_prompt = (
        "Please evaluate the following salt combination compatibility according to hard-filter rules, output JSON only, no other text.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    response = call_dashscope_agent(prompt=user_prompt, system_role=HARD_FILTER_SYSTEM_PROMPT)

    if response is None:
        result = {
            "decision": "PASS",
            "reason_code": "compatible",
            "reason": "Legacy rules passed; LLM supplementary service unavailable, proceeding with legacy rule result"
        }
    else:
        def _parse_llm_response(text: str) -> Optional[Dict[str, str]]:
            raw_text = (text or "").strip()
            if not raw_text:
                return None

            if raw_text.upper() == "PASS":
                return {"decision": "PASS", "reason_code": "compatible", "reason": "LLM returned PASS (non-JSON)"}
            if raw_text.upper() == "REJECT":
                return {"decision": "REJECT", "reason_code": "missing_critical_data", "reason": "LLM returned REJECT (non-JSON)"}

            # Keep only first JSON object, discard all subsequent text
            parsed_obj = _extract_first_json_object(raw_text)
            if not isinstance(parsed_obj, dict):
                return None

            if "decision" not in parsed_obj and (
                "compatible" in parsed_obj or "is_compatible" in parsed_obj or "compatibility" in parsed_obj
            ):
                comp_val = parsed_obj.get("compatible", parsed_obj.get("is_compatible", parsed_obj.get("compatibility")))
                if isinstance(comp_val, bool):
                    decision = "PASS" if comp_val else "REJECT"
                else:
                    comp_str = str(comp_val).strip().lower()
                    if comp_str in {"true", "yes", "1", "compatible", "pass"}:
                        decision = "PASS"
                    elif comp_str in {"false", "no", "0", "incompatible", "reject"}:
                        decision = "REJECT"
                    else:
                        return None

                reason = str(parsed_obj.get("reason", "")).strip()
                if not reason:
                    reasons = parsed_obj.get("incompatibility_reasons")
                    if not isinstance(reasons, list):
                        reasons = parsed_obj.get("incompatible_reasons")
                    if isinstance(reasons, list) and len(reasons) > 0:
                        reason = "; ".join([str(x) for x in reasons[:3]])
                if not reason:
                    reason = "No reason provided, handling conservatively"

                reason_code = "compatible" if decision == "PASS" else _infer_reason_code_from_text(reason)
                return {"decision": decision, "reason_code": reason_code, "reason": _clip_reason_text(reason)}

            raw_decision = str(parsed_obj.get("decision", parsed_obj.get("result", ""))).strip()
            decision = decision_aliases.get(raw_decision.upper(), decision_aliases.get(raw_decision, ""))
            if decision not in {"PASS", "REJECT"}:
                return None

            default_reason_code = "compatible" if decision == "PASS" else "missing_critical_data"
            reason_code = str(parsed_obj.get("reason_code", "")).strip().lower() or default_reason_code
            reason_code = reason_code.replace("-", "_")
            reason_code = reason_code.replace("dtmtoolarge", "dtm_too_large")
            if reason_code not in valid_reason_codes:
                reason_code = _infer_reason_code_from_text(str(parsed_obj.get("reason", "")))
                if decision == "PASS":
                    reason_code = "compatible"

            reason = str(parsed_obj.get("reason", "")).strip() or "No reason provided, handling conservatively"
            return {"decision": decision, "reason_code": reason_code, "reason": _clip_reason_text(reason)}

        parsed_result = _parse_llm_response(response)
        if parsed_result is None:
            preview = response.strip()[:200].replace("\n", "\\n")
            logger.warning(f"LLM hard-filter returned abnormal format, attempting one LLM retry. response_preview={preview}")
            repair_prompt = (
                "Your last output did not match the required format. Please strictly re-answer now.\n"
                "Requirement: output only one JSON object, fields must be decision/reason_code/reason.\n"
                "decision can only be PASS or REJECT; do not output any other text.\n\n"
                f"Please re-evaluate based on the same input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
            )
            repaired_response = call_dashscope_agent(prompt=repair_prompt, system_role=HARD_FILTER_SYSTEM_PROMPT)
            if repaired_response is not None:
                parsed_result = _parse_llm_response(repaired_response)
                if parsed_result is None:
                    repaired_preview = repaired_response.strip()[:200].replace("\n", "\\n")
                    logger.warning(f"LLM hard-filter retry still returned abnormal format. response_preview={repaired_preview}")

        if parsed_result is None:
            result = {
                "decision": "PASS",
                "reason_code": "compatible",
                "reason": "Legacy rules passed; LLM output format abnormal and retry failed, proceeding with legacy rule result"
            }
        else:
            result = parsed_result
            if result.get("reason_code") == "missing_critical_data":
                chem_prompt = (
                    "Please do not use missing_critical_data.\n"
                    "You must directly determine salt pair compatibility based on chemistry knowledge only: PASS or REJECT (choose one).\n"
                    "If REJECT, reason_code can only be chosen from:"
                    "precipitation, hydrolysis_or_acid_base, gas_or_decomposition, redox, "
                    "dTm_too_large, ternary_cross_reaction.\n"
                    "If using gas_or_decomposition, the reason must include both net gas source and boundary conditions;"
                    "If either is missing, must not REJECT.\n"
                    "Must not REJECT by claiming the provided anion_charge is incorrect or missing;"
                    "If claiming charge issues, must cite specific field values from input JSON and be self-consistent.\n"
                    "Must not use speculative statements such as possible, tendency, lack of reported stable ternary eutectic, or possible phase separation as the sole REJECT basis.\n"
                    "If reason_code=precipitation, the reason must include both specific precipitate phase (not generic) and formation pathway/basis;"
                    "If only generic terms (e.g. mixed salts/basic phosphates), must not REJECT.\n"
                    "If PASS, reason_code=compatible.\n"
                    "Output must be JSON: decision/reason_code/reason, no other text.\n\n"
                    f"Input data:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
                    f"Previous round result:\n{json.dumps(result, ensure_ascii=False, indent=2)}"
                )
                chem_response = call_dashscope_agent(prompt=chem_prompt, system_role=HARD_FILTER_SYSTEM_PROMPT)
                if chem_response is not None:
                    chem_result = _parse_llm_response(chem_response)
                    if chem_result is not None:
                        result = chem_result

                if result.get("reason_code") == "missing_critical_data":
                    result = {
                        "decision": "PASS",
                        "reason_code": "compatible",
                        "reason": "Legacy rules passed; LLM supplementary returned missing_critical_data, proceeding with legacy rule result"
                    }

            if result.get("decision") == "REJECT" and result.get("reason_code") == "missing_critical_data":
                missing_fields = _list_missing_critical_fields(a_norm, b_norm)
                if missing_fields:
                    result["reason"] = f"{result.get('reason', '')} | Actually missing fields: {', '.join(missing_fields)}"
                else:
                    result["reason"] = f"{result.get('reason', '')} | Actually all critical fields present (A/B contain Tm,cation,anion,anion_charge)"

            result = _apply_antifalse_reject_guard(result)

    # Clip reason length to prevent overly long text affecting identification/logging
    if isinstance(result, dict):
        result["reason"] = _clip_reason_text(result.get("reason", ""))

    if cache is not None:
        cache[cache_key] = result
    return result["decision"] == "PASS", result


class AnalysisAgent:
    """Analysis Agent - Parse user input and formulate screening strategy"""
    
    def __init__(self, api_key: str = DASHSCOPE_API_KEY):
        self.api_key = api_key
        logger.info("✓ Analysis Agent initialized")
    
    def analyze(self, tm_input: str, dh_input: str) -> Dict[str, Any]:
        """
        Parse target parameters from user input and formulate strategy
        
        Return format:
        {
            'tm_target': float,
            'dh_target': float,
            'tm_tolerance': float,
            'dh_tolerance': float,
            'strategy_level': 'strict' | 'moderate' | 'relaxed'
        }
        """
        logger.info(f"Analysis Agent parsing input: temperature={tm_input}, enthalpy={dh_input}")
        
        # Build prompt
        prompt = f"""
Please parse the target parameters for the following PCM and formulate a screening strategy.

User input:
- Target melting point (Tm): {tm_input}
- Target enthalpy (ΔH): {dh_input}

Please extract the following information and return in JSON format:
1. tm_target: Target melting point value (unit: °C)
2. dh_target: Target enthalpy value (unit: J/g)
3. tm_tolerance: Melting point tolerance range (recommended 2-5°C)
4. dh_tolerance: Enthalpy tolerance range (recommended 5-15 J/g)
5. strategy_level: Screening strategy level (strict/moderate/relaxed)

Note:
- If input is "greater than X", target is X, tolerance can be relaxed
- If input is "X~Y" range, target is (X+Y)/2
- If input is exact value "X", target is X, tolerance is small

Please return JSON only, no other text:
{{
    "tm_target": <value>,
    "dh_target": <value>,
    "tm_tolerance": <value>,
    "dh_tolerance": <value>,
    "strategy_level": "<strict/moderate/relaxed>"
}}
"""
        
        # Call LLM
        response = call_dashscope_agent(
            prompt=prompt,
            system_role="You are a professional PCM parameter parsing expert, skilled at extracting and normalizing physicochemical parameters."
        )
        
        if response:
            try:
                # Extract JSON content
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    strategy = json.loads(json_match.group(0))
                    strategy = self._apply_numeric_targets_from_input(strategy, tm_input, dh_input)
                    logger.info(f"✓ LLM parsing successful: {strategy}")
                    return strategy
                else:
                    logger.warning("LLM format incorrect, using rule-based parsing")
            except Exception as e:
                logger.warning(f"LLM parsing failed: {e}, using rule-based parsing")
        
        # Fallback: use rule-based parsing
        return self._rule_based_parse(tm_input, dh_input)

    def _normalize_numeric_text(self, raw_text: Any) -> str:
        """Normalize minus signs and common punctuation for numeric parsing."""
        return (
            str(raw_text)
            .replace('\u2103', '')
            .replace('°C', '')
            .replace('J/g', '')
            .replace('J·g-1', '')
            .replace('J·g⁻¹', '')
            .replace('−', '-')  # unicode minus
            .replace('–', '-')  # en dash
            .replace('—', '-')  # em dash
            .strip()
        )

    def _extract_target_value(self, raw_text: Any, allow_range_avg: bool = False) -> Optional[float]:
        """Extract one signed numeric target from input text."""
        clean = self._normalize_numeric_text(raw_text)

        if allow_range_avg:
            range_match = re.search(
                r'(-?\d+(?:\.\d+)?)\s*(?:~|\uff5e|to)\s*(-?\d+(?:\.\d+)?)',
                clean,
                flags=re.IGNORECASE,
            )
            if range_match:
                low = float(range_match.group(1))
                high = float(range_match.group(2))
                return (low + high) / 2.0

        nums = re.findall(r'-?\d+(?:\.\d+)?', clean)
        return float(nums[0]) if nums else None

    def _apply_numeric_targets_from_input(
        self, strategy: Dict[str, Any], tm_input: str, dh_input: str
    ) -> Dict[str, Any]:
        """Guarantee numeric targets are consistent with raw user input (including minus sign)."""
        tm_val = self._extract_target_value(tm_input, allow_range_avg=True)
        if tm_val is not None:
            strategy['tm_target'] = tm_val

        dh_val = self._extract_target_value(dh_input, allow_range_avg=False)
        if dh_val is not None:
            strategy['dh_target'] = dh_val

        return strategy
    
    def _rule_based_parse(self, tm_input: str, dh_input: str) -> Dict[str, Any]:
        """Rule-based fallback parsing"""
        logger.info("Using rule engine to parse parameters...")
        
        strategy = {
            'tm_target': 30.0,
            'dh_target': 180.0,
            'tm_tolerance': 5.0,
            'dh_tolerance': 15.0,
            'strategy_level': 'moderate'
        }
        
        # Parse temperature
        tm_clean = self._normalize_numeric_text(tm_input)
        tm_target = self._extract_target_value(tm_input, allow_range_avg=True)
        if tm_target is not None:
            strategy['tm_target'] = tm_target

        if strategy.get('tm_target') is not None:
            if '>' in tm_clean:
                strategy['tm_tolerance'] = 10.0
                strategy['strategy_level'] = 'relaxed'
        
        # Parse enthalpy
        dh_clean = self._normalize_numeric_text(dh_input)
        dh_target = self._extract_target_value(dh_input, allow_range_avg=False)
        if dh_target is not None:
            strategy['dh_target'] = dh_target
            if '>' in dh_clean:
                strategy['dh_tolerance'] = 20.0
        
        logger.info(f"✓ Rule-based parsing completed: {strategy}")
        return strategy


class RetrievalAgent:
    """Retrieval Agent - Screen candidate salts (rules + LLM enhanced)"""
    
    def __init__(self, data_file: str, api_key: str = DASHSCOPE_API_KEY):
        self.api_key = api_key
        self.data_file = data_file
        self.df = None
        self.salt_col = None
        self._load_data()
        # Import strategy selector (initialize after data loading)
        from strategy_selector import StrategySelector, create_strategy_config
        self.strategy_selector = StrategySelector(self.df, self.salt_col)
        self.create_strategy_config = create_strategy_config
        logger.info("✓ Retrieval Agent initialized")
    
    def _load_data(self):
        """Load salt database"""
        try:
            self.df = pd.read_csv(self.data_file, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.data_file, encoding='gbk')
        
        # Detect salt column
        from salt_combination_generator import _detect_salt_column
        self.salt_col = _detect_salt_column(self.df)
        if not self.salt_col:
            self.salt_col = self.df.columns[0]
        
        logger.info(f"Data loaded: {len(self.df)} records, salt column: {self.salt_col}")
    
    def retrieve_salts(self, strategy: Dict[str, Any], attempt: int = 0) -> Dict[str, List[pd.Series]]:
        """
        Retrieve candidate salts (using LLM selection)
        
        Args:
            strategy: Strategy configuration dict
            attempt: Fallback attempt count (0, 1, 2, 3)
        
        Return format:
        {
            'salt_a_candidates': [salt_A1, salt_A2, ...],  # High-Tm salts (temperature-tuning)
            'salt_b_candidates': [salt_B1, salt_B2, ...]   # High-ΔH salts (enthalpy-tuning)
        }
        """
        logger.info(f"Retrieval Agent screening candidate salts (attempt={attempt})...")
        
        tm_target = strategy['tm_target']
        dh_target = strategy['dh_target']
        
        # Use LLM to select salt A and salt B
        try:
            salt_a_candidates = self._llm_select_salts(strategy, 'salt_a', attempt)
            salt_b_candidates = self._llm_select_salts(strategy, 'salt_b', attempt)
            
            logger.info(f"LLM screening yielded {len(salt_a_candidates)} salt A candidates, {len(salt_b_candidates)} salt B candidates")
        except Exception as e:
            logger.warning(f"LLM selection failed: {e}, falling back to rule-based screening")
            # Fallback to rule-based screening
            config = self.create_strategy_config(tm_target, dh_target, attempt)
            salt_a_df = self.strategy_selector.select_salt_a_candidates(config)
            salt_b_df = self.strategy_selector.select_salt_b_candidates(config)
            salt_a_candidates = [row for _, row in salt_a_df.iterrows()]
            salt_b_candidates = [row for _, row in salt_b_df.iterrows()]
            logger.info(f"Rule-based screening yielded {len(salt_a_candidates)} salt A candidates, {len(salt_b_candidates)} salt B candidates")
        
        return {
            'salt_a_candidates': salt_a_candidates,
            'salt_b_candidates': salt_b_candidates
        }
    
    def _llm_select_salts(self, strategy: Dict[str, Any], role: str, attempt: int = 0) -> List[pd.Series]:
        """
        Use LLM to intelligently select candidate salts from full salt database
        
        Args:
            strategy: Strategy configuration dict containing tm_target, dh_target, etc.
            role: Role, 'salt_a' (temperature-tuning) or 'salt_b' (enthalpy-tuning)
            attempt: Fallback attempt count (0, 1, 2, 3)
        
        Returns:
            List[pd.Series]: Candidate salt list, each Series contains all original info and llm_reason
        """
        tm_target = strategy.get('tm_target', 30.0)
        dh_target = strategy.get('dh_target', 180.0)
        
        # Prepare full salt database (filter valid data)
        df_clean = self.df[
            pd.notna(self.df['Tm (°C)']) & 
            pd.notna(self.df['ΔHm (J/g)'])
        ].copy()
        
        if len(df_clean) == 0:
            logger.warning("No valid salt data")
            return []
        
        # Build salt list string
        salt_list_str = ""
        for enum_idx, (idx, row) in enumerate(df_clean.iterrows(), start=1):
            salt_name = row.get(self.salt_col, 'Unknown')
            tm = row.get('Tm (°C)', 0)
            dh = row.get('ΔHm (J/g)', 0)
            salt_list_str += f"{enum_idx}. {salt_name} - Tm: {tm:.1f}°C, ΔH: {dh:.1f} J/g\n"
        
        total_count = len(df_clean)
        
        # Build eutectic physics knowledge based on role
        if role == 'salt_a':
            role_desc = "Temperature-tuning (controls eutectic melting point)"
            eutectic_knowledge = f"""**Eutectic Temperature Rule**: Eutectic point temperature is typically lower than the phase change temperature of individual salts (especially lower than the system minimum melting point).

Therefore, to design a eutectic system with target melting point of {tm_target}°C:

- Individual salt melting points typically need to be higher than the target eutectic point

- Temperature-tuning salt Tm should be equal to or greater than target; the goal is not to find the optimal solution in one pass, but to maximize candidate salt/combination coverage

- But how much higher depends on the degree of eutectic formation and the salt combination

- Please judge based on your materials science knowledge which salts have melting points favorable for forming eutectic points near the target temperature"""
            
            role_specific_guidance = """**For temperature-tuning salt (salt_a):**
- Temperature-tuning salt Tm should be equal to or greater than target; the goal is not to find the optimal solution in one pass, but to maximize candidate salt/combination coverage
- Is this salt's melting point favorable for reaching the target? When combined with other salts, can the eutectic point approach the target? Please judge based on the rule that "eutectic point temperature is typically lower than individual salt phase change temperatures"
- **Note: salt_a only needs to evaluate melting point matching, no need to consider enthalpy**"""
            
            role_specific_note = "salt_a only needs to evaluate melting point matching, no need to consider enthalpy."
        else:  # salt_b
            role_desc = "Enthalpy-tuning (enhances eutectic enthalpy)"
            eutectic_knowledge = f"""**Eutectic Enthalpy Rule**:

- Some eutectics have higher ΔH than the low-enthalpy component (enhanced eutectic effect)

- Some eutectics have lower ΔH than both individual salts (diminished eutectic effect)

To design a eutectic system with target enthalpy of {dh_target} J/g:

- Enthalpy-tuning salt ΔH should be equal to or greater than target; the goal is not to find the optimal solution in one pass, but to maximize candidate salt/combination coverage

- Individual salt enthalpy should be able to reach or approach the target through eutectic combination

- Please judge based on your materials science knowledge and eutectic enthalpy rules which salts are favorable for reaching the target enthalpy"""
            
            role_specific_guidance = """**For enthalpy-tuning salt (salt_b):**
- Enthalpy-tuning salt ΔH should be equal to or greater than target; the goal is not to find the optimal solution in one pass, but to maximize candidate salt/combination coverage
- Is this salt's enthalpy favorable for reaching the target? Please judge based on the rule that "some eutectics have higher ΔH than the low-enthalpy component, while others have lower ΔH than both individual salts"
- **Note: salt_b only needs to evaluate enthalpy matching, no need to consider melting point**"""
            
            role_specific_note = "salt_b only needs to evaluate enthalpy matching, no need to consider melting point."
        
        # Build prompt
        prompt = f"""You are a senior expert in the PCM field with extensive eutectic hydrated salt design experience. You need to **directly select** candidate salts from the complete salt database based on materials science knowledge.

## Design Objectives
- Target melting point (Tm): {tm_target}°C
- Target enthalpy (ΔH): {dh_target} J/g
- Current selection role: {role_desc}

## Eutectic Physics Rules (Knowledge Guidance)
{eutectic_knowledge}

## Complete Salt Database ({total_count} salts)
{salt_list_str}

## Selection Task
Based on your materials science knowledge and the above eutectic physics rules, **directly select** the most suitable candidate salts from the above complete database. This is the **core knowledge judgment step**, requiring rigorous assessment with professional expertise.

**Key Principle**: The goal is not to find the optimal solution in one pass, but to maximize candidate salt/combination coverage. Therefore, all salts meeting the criteria should be selected without artificial quantity limits.

## Screening Criteria (evaluate rigorously)

### 1. Eutectic Formation Ability (must evaluate)
- Does this salt's crystal structure readily form eutectics with other salts? What is the crystal structure compatibility?
- Are this salt's ionic characteristics (cation, anion) favorable for eutectic formation?

### 2. Target Matching (must evaluate rigorously, combining eutectic physics rules)
{role_specific_guidance}


## Output Requirements
Please return in JSON format (no quantity limit, select as many as qualified):

{{
    "selected_salts": [
        {{
            "name": "Full salt name (must exactly match database name)",
            "reason": "<Detailed selection reason, explaining why this salt was chosen, what materials knowledge was used, how eutectic physics rules were applied>"
        }},
        ...
    ],
    "selection_summary": "<Brief summary: why these salts were selected, what the selection strategy is, why others were excluded>"
}}

**Important Notes**:
1. **No quantity limit**: Select all qualifying hydrated salts without artificial limits
2. **Maximize coverage**: The goal is not optimal solution in one pass, but maximum candidate salt/combination coverage. All qualifying salts should be selected
3. **Knowledge-based judgment**: Judge by combining eutectic physics rules (eutectic point typically lower than individual salts, eutectic enthalpy may be higher or lower), do not rely on rigid numerical ranges
4. **Role distinction**: {role_specific_note}
5. Must select from the above database only, cannot select non-existent salts
6. Salt names must **exactly match** database names (including hydration number, format, etc.)
7. Selection reasons must be specific, demonstrating materials science knowledge and eutectic physics reasoning
8. Return JSON only, no other text"""
        
        system_role = "You are a senior expert in the PCM field with extensive eutectic hydrated salt design experience. You are familiar with eutectic formation rules, literature cases, and experimental characteristics of various hydrated salts. You can intelligently select the most suitable salts from a large number of candidates."
        
        # Call LLM API
        response_text = call_dashscope_agent(prompt, system_role)
        
        if not response_text:
            logger.warning(f"LLM API call failed (role={role}), returning empty list")
            return []
        
        # Parse JSON result
        try:
            # Use regex to extract JSON content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                logger.warning(f"Cannot extract JSON from LLM response (role={role})")
                return []
        except Exception as e:
            logger.warning(f"JSON parsing failed (role={role}): {e}")
            return []
        
        # Validate and match salt names
        selected_salts = result.get('selected_salts', [])
        matched_candidates = []
        
        for salt_info in selected_salts:
            salt_name_llm = salt_info.get('name', '').strip()
            reason = salt_info.get('reason', '')
            
            if not salt_name_llm:
                continue
            
            # Find matching salt in database (exact or fuzzy match)
            matched_row = None
            
            # Try exact match first (strip spaces)
            for idx, row in df_clean.iterrows():
                db_name = str(row.get(self.salt_col, '')).strip()
                if db_name == salt_name_llm:
                    matched_row = row.copy()
                    matched_row['llm_reason'] = reason
                    break
            
            # If exact match fails, try case-insensitive match
            if matched_row is None:
                for idx, row in df_clean.iterrows():
                    db_name = str(row.get(self.salt_col, '')).strip()
                    if db_name.lower() == salt_name_llm.lower():
                        matched_row = row.copy()
                        matched_row['llm_reason'] = reason
                        break
            
            if matched_row is not None:
                matched_candidates.append(matched_row)
            else:
                logger.debug(f"Cannot match salt name in database: {salt_name_llm} (role={role})")
        
        logger.info(f"LLM selected {len(selected_salts)} salts, successfully matched {len(matched_candidates)} (role={role})")
        
        # Return candidate salt list (no quantity limit, no score sorting, as LLM already made intelligent selection)
        return matched_candidates
    
    def _llm_rank_salts(self, candidates: pd.DataFrame, strategy: Dict[str, Any], 
                        role: str, target_value: float) -> pd.DataFrame:
        """Use LLM to rank candidate salts (simplified: directly return rule-sorted results)"""
        
        if len(candidates) == 0:
            return candidates
        
        # Simplified implementation: directly return rule-sorted results
        # Full implementation could call LLM to analyze salt chemistry and re-rank
        logger.info(f"✓ Screened {len(candidates)} {role} candidate salts")
        return candidates  # Return all candidates


class CompatibilityAgent:
    """Compatibility Agent - Evaluate eutectic structural compatibility"""
    
    def __init__(self, data_file: str = None, api_key: str = DASHSCOPE_API_KEY):
        self.api_key = api_key
        self.data_file = data_file
        self.df = None
        self.salt_col = None
        self._hard_filter_cache: Dict[str, Dict[str, Any]] = {}
        if data_file:
            self._load_data()
        logger.info("✓ Compatibility Agent initialized")
    
    def _load_data(self):
        """Load salt database"""
        if not self.data_file:
            return
        try:
            self.df = pd.read_csv(self.data_file, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.data_file, encoding='gbk')
        
        # Detect salt column
        from salt_combination_generator import _detect_salt_column
        self.salt_col = _detect_salt_column(self.df)
        if not self.salt_col:
            self.salt_col = self.df.columns[0]
        
        logger.info(f"Compatibility Agent data loaded: {len(self.df)} records, salt column: {self.salt_col}")
    
    def filter_compatible_pairs(self, candidates: Dict[str, List[pd.Series]], 
                                relax_round: int = 0) -> Tuple[Dict[str, List[pd.Series]], Set[Tuple[str, str]]]:
        """
        Check compatibility of all salt pairs, return set of compatible pairs
        
        Args:
            candidates: Candidate salt dict {'salt_a_candidates': [...], 'salt_b_candidates': [...]}
            relax_round: Fallback round (0=strict, >1=relaxed)
        
        Returns:
            (candidates, compatible_pairs_set): 
            - candidates: All candidate salts retained (unfiltered)
            - compatible_pairs_set: Set of compatible salt pairs {(salt_a_name, salt_b_name), ...}
        """
        from salt_combination_generator import load_structured_meta, match_meta
        
        salt_a_list = candidates.get('salt_a_candidates', [])
        salt_b_list = candidates.get('salt_b_candidates', [])
        
        if len(salt_a_list) == 0 or len(salt_b_list) == 0:
            return candidates, set()
        
        logger.info(f"Starting compatibility check, salt A candidates: {len(salt_a_list)}, salt B candidates: {len(salt_b_list)}")
        
        # Load structured metadata (continue on failure, use inline fields as supplement)
        try:
            meta_map = load_structured_meta('data')
        except Exception as e:
            logger.warning(f"Failed to load metadata, will use inline fields for LLM hard-filter: {e}")
            meta_map = {}
        
        # Check all salt pairs for compatibility
        compatible_pairs = set()  # Compatible salt pairs set {(salt_a_name, salt_b_name), ...}
        incompatible_count = 0
        skipped_same_type_count = 0
        skipped_empty_name_count = 0
        skipped_duplicate_count = 0
        total_pairs = 0
        llm_reject_count = 0
        llm_pass_count = 0
        llm_request_fail_count = 0
        chemical_reject_count = 0
        reject_reason_counts: Dict[str, int] = {}
        seen_pair_keys: Set[Tuple[str, str]] = set()
        pair_full_meta_count = 0
        pair_partial_meta_count = 0
        pair_no_meta_count = 0
        
        # Check compatibility pair by pair
        for salt_a in salt_a_list:
            salt_a_name = salt_a.get(self.salt_col, '')
            if not salt_a_name:
                skipped_empty_name_count += len(salt_b_list)  # This salt_a will skip all salt_b
                continue
            
            mA_raw = match_meta(meta_map, salt_a_name)
            has_mA = mA_raw is not None
            mA = _resolve_meta_with_data_priority(meta_map, salt_a_name, salt_a)
            
            for salt_b in salt_b_list:
                salt_b_name = salt_b.get(self.salt_col, '')
                if not salt_b_name:
                    skipped_empty_name_count += 1
                    continue
                
                # Check that salt A and salt B are not same type (ignoring hydration)
                # Consistent with binary prediction stage logic
                if are_same_salt_type(salt_a_name, salt_b_name):
                    skipped_same_type_count += 1
                    continue
                
                pair_key = (salt_a_name, salt_b_name)
                
                # Deduplicate (avoid redundant LLM requests)
                if pair_key in seen_pair_keys:
                    skipped_duplicate_count += 1
                    continue
                seen_pair_keys.add(pair_key)
                
                total_pairs += 1  # Only count unchecked valid salt pairs
                
                mB_raw = match_meta(meta_map, salt_b_name)
                has_mB = mB_raw is not None
                mB = _resolve_meta_with_data_priority(meta_map, salt_b_name, salt_b)
                if has_mA and has_mB:
                    pair_full_meta_count += 1
                elif has_mA or has_mB:
                    pair_partial_meta_count += 1
                else:
                    pair_no_meta_count += 1

                # Single LLM hard-filter (per requirements, no batch API)
                compatible, detail = evaluate_compatibility_hard_filter(
                    salt_a_meta=mA,
                    salt_b_meta=mB,
                    cache=self._hard_filter_cache
                )
                if compatible:
                    compatible_pairs.add(pair_key)
                    llm_pass_count += 1
                else:
                    incompatible_count += 1
                    llm_reject_count += 1
                    reason_code = str(detail.get("reason_code", "")).strip() or "unknown"
                    reject_reason_counts[reason_code] = reject_reason_counts.get(reason_code, 0) + 1
                    if reason_code == "llm_unavailable":
                        llm_request_fail_count += 1
                    else:
                        chemical_reject_count += 1
        
        compatible_count = len(compatible_pairs)
        total_skipped = skipped_same_type_count + skipped_empty_name_count + skipped_duplicate_count
        
        logger.info(f"✓ Compatibility check done: initial pairs={len(salt_a_list) * len(salt_b_list)}")
        logger.info(f"  Skip stats: same salt type={skipped_same_type_count}, empty name={skipped_empty_name_count}, duplicate pairs={skipped_duplicate_count}, total skipped={total_skipped}")
        logger.info(f"  Checked {total_pairs} valid pairs (verify: {len(salt_a_list) * len(salt_b_list)} - {total_skipped} = {len(salt_a_list) * len(salt_b_list) - total_skipped}, actual={total_pairs})")
        logger.info(f"  Compatibility results (LLM hard-filter): PASS={llm_pass_count}, REJECT={llm_reject_count}, final compatible pairs={compatible_count}")
        logger.info(f"  Request failure count: network/service failure={llm_request_fail_count}, chemical rejection={chemical_reject_count}")
        logger.info(f"  Metadata hits: both hit={pair_full_meta_count}, one hit={pair_partial_meta_count}, none hit={pair_no_meta_count}")
        if reject_reason_counts:
            reason_items = sorted(reject_reason_counts.items(), key=lambda x: x[1], reverse=True)
            reason_preview = ", ".join([f"{k}:{v}" for k, v in reason_items[:8]])
            logger.info(f"  REJECT reason distribution (Top): {reason_preview}")
            llm_unavailable_count = reject_reason_counts.get("llm_unavailable", 0)
            if llm_unavailable_count == total_pairs and total_pairs > 0:
                logger.warning("  All pairs in current round rejected due to LLM service unavailability, not chemical incompatibility")
        logger.info(f"  Verify: PASS({llm_pass_count}) + REJECT({llm_reject_count}) = {llm_pass_count + llm_reject_count} (should equal {total_pairs})")
        logger.info(f"  Note: Retaining all salt A candidates ({len(salt_a_list)}) and salt B candidates ({len(salt_b_list)}) for subsequent ternary combination lookup")
        logger.info("  Note: Salt C will be selected from full dataset (by ion matching strategy), will not be same as salt A or B")
        logger.info(f"  Next: binary prediction for {compatible_count} compatible pairs only, skipping {incompatible_count} incompatible pairs")
        
        # Return all candidate salts (unfiltered) and compatible pair set
        return candidates, compatible_pairs
    
    def judge(self, candidates: Dict[str, List[pd.Series]], 
              strategy: Dict[str, Any], 
              tri_reason: Dict[str, bool] = None) -> Dict[str, Any]:
        """
        Evaluate candidate salts and select potential salt C (for ternary combinations)
        
        Note: Regardless of compatibility, will attempt to select salt C candidates,
        as we try both binary and ternary combinations
        
        Args:
            candidates: Candidate salt dict
            strategy: Strategy configuration
            tri_reason: Ternary trigger reason (optional), format: {'no_monopeak': bool, 'perf_fail': bool, 'struct_fail': bool}
        
        Return format:
        {
            'compatible': bool,
            'need_salt_c': bool,  # deprecated, kept for compatibility only
            'salt_c_candidates': [salt_C1, salt_C2, ...],  # always try to provide
            'reason': str
        }
        """
        logger.info("Compatibility Agent evaluating and selecting salt C...")
        
        salt_a_list = candidates.get('salt_a_candidates', [])
        salt_b_list = candidates.get('salt_b_candidates', [])
        
        if len(salt_a_list) == 0 or len(salt_b_list) == 0:
            logger.warning("Insufficient candidate salts")
            return {
                'compatible': False,
                'need_salt_c': True,
                'salt_c_candidates': [],
                'reason': 'Insufficient candidate salt count'
            }
        
        # If no trigger reason provided, use default strategy (compatibility strategy)
        if tri_reason is None:
            tri_reason = {
                'no_monopeak': False,
                'perf_fail': False,
                'struct_fail': False
            }
            logger.info("No trigger reason provided, using default compatibility strategy to select salt C")
        
        # Select salt C candidates based on trigger reason
        salt_c_candidates = self._select_salt_c(candidates, strategy, tri_reason)
        
        result = {
            'compatible': True,
            'need_salt_c': len(salt_c_candidates) > 0,  # Try ternary if salt C available
            'salt_c_candidates': salt_c_candidates,
            'reason': f'Selected {len(salt_c_candidates)} salt C candidates, will try both binary and ternary combinations'
        }
        
        logger.info(f"✓ Evaluation done: will try both binary and ternary combinations ({len(salt_c_candidates)} salt C candidates)")
        
        return result
    
    def _get_salt_col(self, row: pd.Series) -> str:
        """Get salt column name (first column of the salt library)"""
        return row.index[0]
    
    def _select_salt_c(self, candidates: Dict[str, List[pd.Series]], 
                      strategy: Dict[str, Any],
                      tri_reason: Dict[str, bool],
                      limit: Optional[int] = 1) -> List[pd.Series]:
        """
        Select salt C candidates (intelligent strategy based on trigger reason and ion matching)
        
        Select salt C based on failure reason and ion matching logic:
        - If need lower Tm: select C with lower Tm sharing anion/cation with A or B
        - If need higher ΔH: select C with higher ΔH sharing anion/cation with A or B
        - If single peak failure: select C sharing anion/cation with A or B
        """
        logger.info(f"Selecting salt C, trigger reason: {tri_reason}")
        
        # If no data file, use simplified strategy
        if self.df is None or self.salt_col is None:
            logger.warning("Compatibility Agent data not loaded, using simplified strategy")
            salt_a_list = candidates.get('salt_a_candidates', [])
            if len(salt_a_list) > 0:
                return salt_a_list[5:10] if len(salt_a_list) > 10 else salt_a_list[2:5]
            return []
        
        # Filter valid data
        df_clean = self.df[pd.notna(self.df['Tm (°C)']) & pd.notna(self.df['ΔHm (J/g)'])].copy()
        
        if len(df_clean) == 0:
            logger.warning("No valid salt data for selecting salt C")
            return []
        
        # Get ion info from salt A and salt B
        salt_a_list = candidates.get('salt_a_candidates', [])
        salt_b_list = candidates.get('salt_b_candidates', [])
        
        # Exclude salts of same type as A/B (not just different hydration)
        salt_a_names = {row.get(self.salt_col, '') for row in salt_a_list if row.get(self.salt_col, '')}
        salt_b_names = {row.get(self.salt_col, '') for row in salt_b_list if row.get(self.salt_col, '')}
        
        # Filter out salts of same type as salt A or salt B
        filtered_indices = []
        for idx, row in df_clean.iterrows():
            salt_c_name = row.get(self.salt_col, '')
            if not salt_c_name:
                continue
            
            # Check if same type as salt A or B (ignoring hydration)
            is_same_as_a = any(are_same_salt_type(salt_c_name, salt_a_name) for salt_a_name in salt_a_names)
            is_same_as_b = any(are_same_salt_type(salt_c_name, salt_b_name) for salt_b_name in salt_b_names)
            
            if not is_same_as_a and not is_same_as_b:
                filtered_indices.append(idx)
        
        if filtered_indices:
            df_clean = df_clean.loc[filtered_indices].copy()
            logger.info(f"After excluding salts of same type as A/B, {len(df_clean)} candidates remain")
        else:
            logger.warning("After excluding salts of same type as A/B, no candidates remain")
            return []
        
        ab_ions = self._extract_ions_from_candidates(salt_a_list, salt_b_list)
        
        if not ab_ions['cations'] and not ab_ions['anions']:
            logger.warning("Cannot extract ion info from salt A/B, returning no salt C candidates per strict rules")
            candidates_df = df_clean.iloc[0:0].copy()
        else:
            # Select different strategies based on trigger reason (with ion matching)
            if tri_reason.get('no_monopeak', False):
                # Single peak failure: use ion matching as structural adjustment strategy
                candidates_df = self._filter_by_ion_match(df_clean, ab_ions)
                logger.info(f"Single peak failure: selecting salt C with shared ions as structural modifier, found {len(candidates_df)} candidates")
            elif tri_reason.get('perf_fail', False):
                # Performance not met: determine if need lower Tm or higher ΔH
                # Note: need_lower_tm and need_higher_dh should come from outside, but _select_salt_c lacks these params
                # So using default logic: judge based on median of salt A and B candidates
                tm_target = strategy.get('tm_target')
                dh_target = strategy.get('dh_target')
                
                # Check if need lower Tm or higher ΔH
                need_lower_tm = False
                need_higher_dh = False
                
                if tm_target is not None:
                    # If target Tm is lower than salt A candidate median, need lower Tm
                    salt_a_tm_values = [row.get('Tm (°C)') for row in salt_a_list if pd.notna(row.get('Tm (°C)', None))]
                    if salt_a_tm_values:
                        salt_a_tm_median = pd.Series(salt_a_tm_values).median()
                        need_lower_tm = tm_target < salt_a_tm_median
                
                if dh_target is not None:
                    # If target ΔH is higher than salt B candidate median, need higher ΔH
                    salt_b_dh_values = [row.get('ΔHm (J/g)') for row in salt_b_list if pd.notna(row.get('ΔHm (J/g)', None))]
                    if salt_b_dh_values:
                        salt_b_dh_median = pd.Series(salt_b_dh_values).median()
                        need_higher_dh = dh_target > salt_b_dh_median
                
                if need_lower_tm:
                    # If need lower Tm: select C with lower Tm sharing anion/cation with A or B
                    candidates_df = self._filter_by_ion_match(df_clean, ab_ions)
                    candidates_df = candidates_df.sort_values('Tm (°C)')  # Sort by Tm ascending
                    logger.info(f"Need lower Tm: selecting salt C with lower Tm and shared ions, found {len(candidates_df)} candidates")
                elif need_higher_dh:
                    # If need higher ΔH: select C with higher ΔH sharing anion/cation with A or B
                    candidates_df = self._filter_by_ion_match(df_clean, ab_ions)
                    candidates_df = candidates_df.sort_values('ΔHm (J/g)', ascending=False)  # Sort by ΔH descending
                    logger.info(f"Need higher ΔH: selecting salt C with higher ΔH and shared ions, found {len(candidates_df)} candidates")
                else:
                    # Default: select high-enthalpy salt to compensate performance
                    candidates_df = self._filter_by_ion_match(df_clean, ab_ions)
                    candidates_df = candidates_df.sort_values('ΔHm (J/g)', ascending=False)
                    logger.info(f"Performance not met: selecting salt C with higher ΔH and shared ions, found {len(candidates_df)} candidates")
            else:
                # Default compatibility strategy: maintain ion consistency with A/B
                candidates_df = self._filter_by_ion_match(df_clean, ab_ions)
                logger.info(f"Default compatibility strategy: selecting salt C with shared ions, found {len(candidates_df)} candidates")
        
        # Convert to Series list
        salt_c_list = [row for _, row in candidates_df.iterrows()]
        
        logger.info(f"✓ Selected {len(salt_c_list)} salt C candidates based on strategy")
        
        if limit is not None and len(salt_c_list) > limit:
            logger.info(f"  Keeping only top {limit} salt C candidates for subsequent combinations")
            salt_c_list = salt_c_list[:limit]
            logger.info(f"  Remaining {len(salt_c_list)} salt C candidates will be used for ternary combination generation")
        
        return salt_c_list
    
    def _extract_ions_from_candidates(self, salt_a_list: List[pd.Series], 
                                      salt_b_list: List[pd.Series]) -> Dict[str, set]:
        """Extract anion and cation info from salt A and B candidates"""
        cations = set()
        anions = set()
        
        try:
            from salt_combination_generator import load_structured_meta
            
            # Load structured metadata
            meta_map = load_structured_meta('data')
            
            # Extract ions from salt A candidates
            for row in salt_a_list:
                salt_name = row.get(self.salt_col, '')
                if not salt_name:
                    continue
                meta = _resolve_meta_with_data_priority(meta_map, salt_name, row)
                if meta:
                    if meta.get('cation'):
                        cations.add(meta['cation'])
                    if meta.get('anion'):
                        anions.add(meta['anion'])
            
            # Extract ions from salt B candidates
            for row in salt_b_list:
                salt_name = row.get(self.salt_col, '')
                if not salt_name:
                    continue
                meta = _resolve_meta_with_data_priority(meta_map, salt_name, row)
                if meta:
                    if meta.get('cation'):
                        cations.add(meta['cation'])
                    if meta.get('anion'):
                        anions.add(meta['anion'])
            
            logger.info(f"Extracted {len(cations)} cation types and {len(anions)} anion types from salt A/B candidates")
            
        except Exception as e:
            logger.warning(f"Failed to extract ion info: {e}")
        
        return {'cations': cations, 'anions': anions}
    
    def _filter_by_ion_match(self, df: pd.DataFrame, ab_ions: Dict[str, set]) -> pd.DataFrame:
        """Filter salt C by ion matching: select salts sharing anion or cation with A or B"""
        if not ab_ions['cations'] and not ab_ions['anions']:
            return df
        
        try:
            from salt_combination_generator import load_structured_meta
            
            # Load structured metadata
            meta_map = load_structured_meta('data')
            
            matched_indices = []
            
            for idx, row in df.iterrows():
                salt_name = row.get(self.salt_col, '')
                if not salt_name:
                    continue
                    
                meta = _resolve_meta_with_data_priority(meta_map, salt_name, row)
                
                if meta:
                    salt_cation = meta.get('cation')
                    salt_anion = meta.get('anion')
                    
                    # Check if sharing cation or anion
                    has_matching_cation = salt_cation and salt_cation in ab_ions['cations']
                    has_matching_anion = salt_anion and salt_anion in ab_ions['anions']
                    
                    if has_matching_cation or has_matching_anion:
                        matched_indices.append(idx)
            
            if matched_indices:
                matched_df = df.loc[matched_indices].copy()
                logger.info(f"Ion matching filter: screened {len(matched_df)} matching salt C candidates from {len(df)} salts")
                return matched_df
            else:
                logger.warning("Ion matching filter: no salt C found meeting shared cation or anion, returning empty candidates")
                return df.iloc[0:0].copy()
                
        except Exception as e:
            logger.warning(f"Ion matching filter failed: {e}, returning empty candidates per strict rules")
            return df.iloc[0:0].copy()
    
    def _select_compatible_salts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select salts with good compatibility (hydration number 2-6)"""
        # Check if hydration number column exists
        hydration_col = None
        for col in df.columns:
            if 'Hydration' in col or 'hydration' in col.lower():
                hydration_col = col
                break
        
        if hydration_col and hydration_col in df.columns:
            df_with_hydration = df[pd.notna(df[hydration_col])].copy()
            if len(df_with_hydration) > 0:
                # Select salts with hydration number 2-6 (typically better compatibility)
                compatible = df_with_hydration[
                    (df_with_hydration[hydration_col] >= 2) &
                    (df_with_hydration[hydration_col] <= 6)
                ]
                if len(compatible) > 0:
                    logger.info(f"Found {len(compatible)} salts with hydration number 2-6")
                    return compatible
        
        # Fallback: return all salts
        logger.info("No hydration number data found, returning all salts")
        return df


class RatioFunnelAgent:
    """Ratio Funnel Agent - Generate ratio schemes"""
    
    def __init__(self):
        logger.info("✓ Ratio Funnel Agent initialized")
    
    def generate_ratios(self, compatibility_result: Dict[str, Any], 
                       is_ternary: bool = None) -> List[Tuple[float, ...]]:
        """
        Generate ratio schemes
        
        Return format:
        - Binary: [(ratio_a, ratio_b), ...]
        - Ternary: [(ratio_a, ratio_b, ratio_c), ...]
        """
        logger.info("Ratio Funnel Agent generating ratio schemes...")
        
        # Determine binary or ternary
        if is_ternary is None:
            is_ternary = compatibility_result.get('need_salt_c', False)
        
        if is_ternary:
            ratios = self._generate_ternary_ratios()
            logger.info(f"✓ Generated {len(ratios)} ternary ratio schemes")
        else:
            ratios = self._generate_binary_ratios()
            logger.info(f"✓ Generated {len(ratios)} binary ratio schemes")
        
        return ratios
    
    def _generate_binary_ratios(self) -> List[Tuple[float, float]]:
        """Generate binary ratios"""
        ratios = []
        # Generate fine-grained ratios: 0.01:0.99, 0.02:0.98, ..., 0.99:0.01 (up to 99)
        for ratio_a in np.arange(0.01, 1.0, 0.01):
            ratio_b = 1.0 - ratio_a
            if abs(ratio_a + ratio_b - 1.0) < 0.01:  # Ensure sum equals 1
                ratios.append((round(ratio_a, 2), round(ratio_b, 2)))
        return ratios
    
    def _generate_ternary_ratios(self) -> List[Tuple[float, float, float]]:
        """Generate ternary ratios"""
        ratios = []
        # Salt C ratio: 10% to 50%, step 5%
        for c_ratio in np.arange(0.1, 0.51, 0.05):  # 10% to 50%, step 5%
            remaining = 1.0 - c_ratio
            # A and B ratio distribution in remaining portion: 20% to 80%, step 5%
            for ab_ratio in np.arange(0.2, 0.81, 0.05):  # 20% to 80%, step 5%
                ratio_a = remaining * ab_ratio
                ratio_b = remaining * (1 - ab_ratio)
                if abs(ratio_a + ratio_b + c_ratio - 1.0) < 0.01:
                    ratios.append((
                        round(ratio_a, 3), 
                        round(ratio_b, 3), 
                        round(c_ratio, 3)
                    ))
        return ratios


class TernaryAgent:
    """Ternary Agent - Prediction and validation using XGBoost"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.df = None
        self.models = None
        self.salt_col = None
        self._hard_filter_cache: Dict[str, Dict[str, Any]] = {}
        self._load_data()
        self._load_models()
        logger.info("✓ Ternary Agent initialized")
    
    def _load_data(self):
        """Load data"""
        try:
            self.df = pd.read_csv(self.data_file, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.data_file, encoding='gbk')
        
        from salt_combination_generator import _detect_salt_column
        self.salt_col = _detect_salt_column(self.df)
        if not self.salt_col:
            self.salt_col = self.df.columns[0]
    
    def _load_models(self):
        """Load XGBoost models"""
        from salt_combination_generator import load_models
        try:
            self.models = load_models()
            logger.info("XGBoost models loaded")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def predict_and_validate(self, candidates: Dict[str, List[pd.Series]], 
                            ratios: List[Tuple[float, ...]], 
                            strategy: Dict[str, Any],
                            compatibility_result: Dict[str, Any] = None,
                            try_ternary: bool = False,
                            compatible_pairs: Set[Tuple[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Predict and validate salt combination performance
        
        Args:
            try_ternary: Whether to try ternary combinations (default False, True only after binary failure)
        
        Return format:
        [
            {
                'salt_a': str,
                'salt_b': str,
                'salt_c': str (if ternary),
                'ratio_a': float,
                'ratio_b': float,
                'ratio_c': float (if ternary),
                'tm_pred': float,
                'dh_pred': float,
                'single_peak_proba': float,
                'is_single_peak': int,
                'score': float
            },
            ...
        ]
        """
        salt_a_list = candidates.get('salt_a_candidates', [])
        salt_b_list = candidates.get('salt_b_candidates', [])
        
        results = []
        tm_target = strategy['tm_target']
        dh_target = strategy['dh_target']
        
        # ===== Step 1: Predict binary combinations =====
        logger.info("Step 1: Predicting binary combinations...")
        binary_ratios = self._generate_binary_ratios()
        binary_results = self._predict_binary(
            salt_a_list, salt_b_list, binary_ratios, 
            tm_target, dh_target, compatible_pairs=compatible_pairs
        )
        
        # Separate qualified and unqualified binary combinations
        # Performance criteria: single peak passed + Tm error ≤6°C + ΔH error ≤30 J/g (consistent with 80% compliance)
        valid_binary = [r for r in binary_results if r.get('score') != float('inf') and 
                       abs(r.get('tm_pred', 0) - tm_target) <= 6.0 and 
                       abs(r.get('dh_pred', 0) - dh_target) <= 30.0]
        invalid_binary = [r for r in binary_results if r not in valid_binary]
        
        # Always return all prediction results (including unqualified) for subsequent processing
        results.extend(binary_results)  # Include all binary combinations (qualified and unqualified)
        logger.info(f"✓ Binary combination classification: qualified {len(valid_binary)}, unqualified {len(invalid_binary)}")
        
        # ===== Step 2: Add salt C to unqualified binary combinations for ternary =====
        # Only add salt C to unqualified binary combinations when try_ternary=True
        if try_ternary and len(invalid_binary) > 0:
            logger.info(f"Step 2: Selecting salt C for {len(invalid_binary)} unqualified binary combinations and generating ternary...")
            ternary_results = self._augment_invalid_binary_with_salt_c(
                invalid_binary, salt_a_list, salt_b_list, tm_target, dh_target, strategy
            )
            results.extend(ternary_results)
            logger.info(f"✓ Ternary prediction done: found {len(ternary_results)} valid ternary combinations")
        elif len(invalid_binary) > 0:
            logger.info(f"Skipping ternary generation (try_ternary=False), keeping {len(valid_binary)} qualified binary, {len(invalid_binary)} unqualified for later processing")
        
        # Sort by score (mixed binary and ternary)
        results.sort(key=lambda x: x['score'])
        
        total_binary = len([r for r in results if r['type'] == 'binary'])
        total_ternary = len([r for r in results if r['type'] == 'ternary'])
        
        logger.info(f"✓ XGBoost prediction done: found {len(results)} total prediction results")
        logger.info(f"  - Binary combinations: {total_binary}")
        if try_ternary:
            logger.info(f"  - Ternary combinations: {total_ternary}")
        
        return results  # Return all results
    
    def _generate_binary_ratios(self) -> List[Tuple[float, float]]:
        """Generate binary ratios"""
        ratios = []
        # Step 0.01: up to 99 ratio attempts per binary salt pair
        for ratio_a in np.arange(0.01, 1.0, 0.01):
            ratio_b = 1.0 - ratio_a
            if abs(ratio_a + ratio_b - 1.0) < 0.01:
                ratios.append((round(ratio_a, 2), round(ratio_b, 2)))
        return ratios
    
    def _generate_ternary_ratios(self) -> List[Tuple[float, float, float]]:
        """Generate ternary ratios"""
        ratios = []
        for c_ratio in np.arange(0.1, 0.51, 0.05):
            remaining = 1.0 - c_ratio
            for ab_ratio in np.arange(0.2, 0.81, 0.05):
                ratio_a = remaining * ab_ratio
                ratio_b = remaining * (1 - ab_ratio)
                if abs(ratio_a + ratio_b + c_ratio - 1.0) < 0.01:
                    ratios.append((
                        round(ratio_a, 3), 
                        round(ratio_b, 3), 
                        round(c_ratio, 3)
                    ))
        return ratios
    
    def _predict_binary(self, salt_a_list: List[pd.Series], salt_b_list: List[pd.Series],
                       ratios: List[Tuple[float, float]], tm_target: float, dh_target: float,
                       compatible_pairs: Set[Tuple[str, str]] = None) -> List[Dict[str, Any]]:
        """Predict binary combinations (save all results for failure analysis)"""
        from salt_combination_generator import build_feature_row, predict_with_models

        results = []
        skipped_same_type = 0
        skipped_empty_name = 0
        skipped_incompatible = 0
        skipped_duplicate = 0
        predicted_count = 0
        error_count = 0
        predicted_pairs = set()
        
        total_combinations = len(salt_a_list) * len(salt_b_list) * len(ratios)
        expected_predictions = len(compatible_pairs) * len(ratios) if compatible_pairs else total_combinations
        
        if compatible_pairs is not None:
            logger.info(f"Starting binary prediction: theoretical total={total_combinations}, expected predictions={expected_predictions} compatible combinations")
        
        for salt_a in salt_a_list:
            for salt_b in salt_b_list:
                # Check salt names first (before ratio loop, avoid redundant checks)
                salt_a_name = salt_a.get(self.salt_col, '')
                salt_b_name = salt_b.get(self.salt_col, '')
                
                # Check empty names
                if not salt_a_name or not salt_b_name:
                    skipped_empty_name += len(ratios)  # Skip all ratios for this salt pair
                    continue
                
                # Check salt A and B must not be same type (ignoring hydration)
                if are_same_salt_type(salt_a_name, salt_b_name):
                    skipped_same_type += len(ratios)  # Skip all ratios for this salt pair
                    continue
                
                # Build pair key (for compatibility check and deduplication)
                pair_key = (salt_a_name, salt_b_name)
                
                # Compatibility check: if compatible pairs set provided, only predict compatible pairs
                if compatible_pairs is not None:
                    if pair_key not in compatible_pairs:
                        skipped_incompatible += len(ratios)  # Skip all ratios for this salt pair
                        continue
                
                # Check if this salt pair was already predicted (avoid redundant prediction)
                if pair_key in predicted_pairs:
                    skipped_duplicate += len(ratios)  # Skip all ratios for this salt pair
                    continue
                
                # Record this salt pair as predicted
                predicted_pairs.add(pair_key)
                
                # Iterate all ratios for prediction
                for ratio_tuple in ratios:
                    try:
                        # At this point, the pair is compatible, proceed with prediction
                        predicted_count += 1
                        ratio_a, ratio_b = ratio_tuple
                        
                        # Build features and predict
                        feat = build_feature_row([salt_a, salt_b], [ratio_a, ratio_b])
                        pred = predict_with_models(self.models, feat)
                        
                        # Save all prediction results (including filtered), for failure analysis
                        result_item = {
                            'salt_a': salt_a.get(self.salt_col, 'Unknown'),
                            'salt_b': salt_b.get(self.salt_col, 'Unknown'),
                            'ratio_a': ratio_a,
                            'ratio_b': ratio_b,
                            'tm_pred': pred.get('tm_pred'),
                            'dh_pred': pred.get('dh_pred'),
                            'single_peak_proba': pred.get('single_peak_proba', 0),
                            'is_single_peak': pred.get('is_single_peak', 0),
                            'type': 'binary',
                            'target_tm': tm_target,
                            'target_dh': dh_target
                        }
                        
                        # Check single peak
                        if pred['is_single_peak'] != 1 or pred['single_peak_proba'] < 0.5:
                            # Non-single-peak combinations also saved, marked invalid
                            result_item['score'] = float('inf')
                            results.append(result_item)
                            continue
                        
                        # Check performance
                        if pd.isna(pred['tm_pred']) or pd.isna(pred['dh_pred']):
                            # Prediction failure also saved
                            result_item['score'] = float('inf')
                            results.append(result_item)
                            continue
                        
                        tm_diff = abs(pred['tm_pred'] - tm_target)
                        dh_diff = abs(pred['dh_pred'] - dh_target)
                        score = tm_diff + dh_diff * 0.02
                        result_item['score'] = score
                        
                        # Relax conditions to get more candidates (for subsequent compliance evaluation)
                        # But all results are saved, including those not meeting this condition
                        results.append(result_item)
                        
                        if predicted_count % 500 == 0:
                            valid_count = len([r for r in results if r['score'] != float('inf') and 
                                             abs(r.get('tm_pred', 0) - tm_target) <= 6.0 and 
                                             abs(r.get('dh_pred', 0) - dh_target) <= 30.0])
                            logger.info(f"  Predicted {predicted_count} compatible combinations, found {valid_count} performance-qualified (80% compliance)")
                    
                    except Exception as e:
                        error_count += 1
                        logger.warning(f"Binary combination prediction failed: {e}")
                        continue
        
        # Output statistics
        # Statistics criteria:
        # - Single peak passed: score != inf (single peak model passed and predicted values not NaN)
        # - Performance qualified: single peak passed + Tm error ≤6°C + ΔH error ≤30 J/g (consistent with 80% compliance)
        single_peak_passed = [r for r in results if r.get('score') != float('inf')]
        performance_qualified = [r for r in results if r.get('score') != float('inf') and 
                                abs(r.get('tm_pred', 0) - tm_target) <= 6.0 and 
                                abs(r.get('dh_pred', 0) - dh_target) <= 30.0]
        
        # Calculate unique pair count
        unique_pairs_count = len(predicted_pairs)
        skipped_same_type_pairs = skipped_same_type // len(ratios) if len(ratios) > 0 else skipped_same_type
        skipped_empty_name_pairs = skipped_empty_name // len(ratios) if len(ratios) > 0 else skipped_empty_name
        skipped_incompatible_pairs = skipped_incompatible // len(ratios) if len(ratios) > 0 else skipped_incompatible
        skipped_duplicate_pairs = skipped_duplicate // len(ratios) if len(ratios) > 0 else skipped_duplicate
        
        logger.info("Binary prediction statistics:")
        logger.info(f"  - Total combinations: {total_combinations}")
        if skipped_same_type > 0:
            logger.info(f"  - Skipped same salt type: {skipped_same_type} combinations ({skipped_same_type_pairs} pairs × {len(ratios)} ratios)")
        if skipped_empty_name > 0:
            logger.info(f"  - Skipped empty name: {skipped_empty_name} combinations ({skipped_empty_name_pairs} pairs × {len(ratios)} ratios)")
        if skipped_incompatible > 0:
            logger.info(f"  - Skipped incompatible pairs: {skipped_incompatible} combinations ({skipped_incompatible_pairs} pairs × {len(ratios)} ratios)")
        if skipped_duplicate > 0:
            logger.info(f"  - Skipped duplicate pairs: {skipped_duplicate} combinations ({skipped_duplicate_pairs} pairs × {len(ratios)} ratios)")
        logger.info(f"  - Actual predictions: {predicted_count} combinations ({unique_pairs_count} unique compatible pairs × {len(ratios)} ratios)")
        
        # Verify statistics
        total_skipped = skipped_same_type + skipped_empty_name + skipped_incompatible + skipped_duplicate
        expected_predictions = unique_pairs_count * len(ratios) if len(ratios) > 0 else unique_pairs_count
        logger.info(f"  - Verify: {total_combinations} - {total_skipped} = {total_combinations - total_skipped}, actual={predicted_count}, expected={expected_predictions}")
        
        if compatible_pairs is not None:
            if unique_pairs_count != len(compatible_pairs):
                logger.warning(f"  - Warning: compatibility check had {len(compatible_pairs)} unique compatible pairs, but binary prediction only predicted {unique_pairs_count} unique pairs")
            else:
                logger.info(f"  - Verification passed: predicted unique pair count ({unique_pairs_count}) matches compatibility check")
        if error_count > 0:
            logger.info(f"  - Prediction errors: {error_count}")
        # Count unqualified combinations
        no_monopeak = len(results) - len(single_peak_passed)  # Single peak not passed
        perf_fail = len(single_peak_passed) - len(performance_qualified)  # Single peak passed but performance not met
        total_invalid = len(results) - len(performance_qualified)  # Total unqualified
        
        logger.info(f"  - Prediction results: {len(results)}")
        logger.info(f"    * Single peak passed: {len(single_peak_passed)}")
        logger.info(f"    * Performance qualified: {len(performance_qualified)} (single peak passed + Tm error ≤6°C + ΔH error ≤30 J/g, 80% compliance)")
        logger.info(f"    * Unqualified: {total_invalid} (single peak failed {no_monopeak} + passed but performance not met {perf_fail})")
        
        return results
    
    def _augment_invalid_binary_with_salt_c(self, invalid_binary: List[Dict[str, Any]], 
                                            salt_a_list: List[pd.Series], 
                                            salt_b_list: List[pd.Series],
                                            tm_target: float, dh_target: float,
                                            strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Select salt C for unqualified binary combinations and generate ternary combinations
        
        Args:
            invalid_binary: List of unqualified binary combinations
            salt_a_list: Salt A candidate list
            salt_b_list: Salt B candidate list
            tm_target: Target melting point
            dh_target: Target enthalpy
            strategy: Strategy configuration
        
        Returns:
            List of ternary combination prediction results
        
        Note: Different failed ratios of the same binary pair will select different salt C based on failure type
        """
        results = []
        
        # Create compatibility agent (reuse, avoid repeated initialization)
        compatibility_agent = CompatibilityAgent(self.data_file)
        
        # Group failed binary combinations by salt pair: {(salt_a_name, salt_b_name): [binary_result1, binary_result2, ...]}
        salt_pair_groups = {}
        for binary_result in invalid_binary:
            salt_a_name = binary_result.get('salt_a')
            salt_b_name = binary_result.get('salt_b')
            pair_key = (salt_a_name, salt_b_name)
            if pair_key not in salt_pair_groups:
                salt_pair_groups[pair_key] = []
            salt_pair_groups[pair_key].append(binary_result)
        
        logger.info(f"  Grouped by salt pair: {len(salt_pair_groups)} distinct pairs, {len(invalid_binary)} failed ratio combinations")
        
        ternary_count = 0
        total_generated_count = 0
        valid_count = 0
        
        # Select corresponding salt C for each failed ratio of each pair
        for pair_idx, (pair_key, binary_results) in enumerate(salt_pair_groups.items()):
            salt_a_name, salt_b_name = pair_key
            
            # Find corresponding salt A and B
            salt_a = next((s for s in salt_a_list if s.get(self.salt_col) == salt_a_name), None)
            salt_b = next((s for s in salt_b_list if s.get(self.salt_col) == salt_b_name), None)
            
            if salt_a is None or salt_b is None:
                continue
            
            logger.info(f"  Processing pair ({salt_a_name}, {salt_b_name}): {len(binary_results)} failed ratios")
            
            # Maintain used salt C set per pair, ensure different failed ratios select different salt C
            used_salt_c_names = set()
            
            # Select corresponding salt C for each failed ratio based on failure reason
            for binary_result in binary_results:
                single_peak_proba = binary_result.get('single_peak_proba', 0)
                is_single_peak = binary_result.get('is_single_peak', 0)
                tm_pred = binary_result.get('tm_pred')
                dh_pred = binary_result.get('dh_pred')
                ratio_a = binary_result.get('ratio_a', 0.5)
                ratio_b = binary_result.get('ratio_b', 0.5)
                
                # Determine failure reason and select corresponding salt C
                if is_single_peak != 1 or single_peak_proba < 0.5:
                    # Single peak failure: select salt with shared ions as structural modifier
                    tri_reason = {'no_monopeak': True, 'perf_fail': False, 'struct_fail': False}
                    salt_c_list = self._select_salt_c_for_binary_pair(
                        salt_a, salt_b, salt_a_list, salt_b_list,
                        tri_reason, strategy,
                        compatibility_agent=compatibility_agent,
                        exclude_salt_names=used_salt_c_names  # Only exclude used salt C within the same binary pair
                    )
                    failure_type = "single_peak_failure"
                else:
                    # Performance not met: determine if need lower Tm or higher ΔH
                    need_lower_tm = pd.notna(tm_pred) and tm_pred > tm_target
                    need_higher_dh = pd.notna(dh_pred) and dh_pred < dh_target
                    tri_reason = {'no_monopeak': False, 'perf_fail': True, 'struct_fail': False}
                    salt_c_list = self._select_salt_c_for_binary_pair(
                        salt_a, salt_b, salt_a_list, salt_b_list, tri_reason, strategy,
                        need_lower_tm=need_lower_tm, need_higher_dh=need_higher_dh,
                        compatibility_agent=compatibility_agent,
                        exclude_salt_names=used_salt_c_names  # Only exclude used salt C within the same binary pair
                    )
                    if need_lower_tm:
                        failure_type = "perf_fail(need_lower_Tm)"
                    elif need_higher_dh:
                        failure_type = "perf_fail(need_higher_ΔH)"
                    else:
                        failure_type = "perf_fail"
                
                if not salt_c_list:
                    logger.debug(f"    Ratio ({ratio_a:.2f}:{ratio_b:.2f}) [{failure_type}]: no available salt C (may all be used)")
                    continue
                
                # Record used salt C name (for subsequent ratio exclusion)
                salt_c_name = salt_c_list[0].get(self.salt_col, 'Unknown')
                used_salt_c_names.add(salt_c_name)
                logger.debug(f"    Ratio ({ratio_a:.2f}:{ratio_b:.2f}) [{failure_type}]: selected salt C {salt_c_name}")
                
                # Generate ternary combinations for this failed ratio (using corresponding salt C)
                ternary_results, generated_count = self._generate_ternary_for_binary_pair(
                    salt_a, salt_b, salt_c_list,
                    ratio_a, ratio_b,
                    tm_target, dh_target
                )
                ternary_count += len(ternary_results)
                total_generated_count += generated_count  # Accumulate theoretical generation count
                valid_count += len(ternary_results)  # All returned results are valid (passed single peak and performance check)
                results.extend(ternary_results)
            
            if (pair_idx + 1) % 10 == 0:
                logger.info(f"    Processed {pair_idx + 1}/{len(salt_pair_groups)} pairs, theoretically generated {total_generated_count} ternary combinations, found {ternary_count} valid")
        
        logger.info(f"  ✓ Ternary generation done: processed {len(salt_pair_groups)} pairs, theoretically generated {total_generated_count} ternary, found {ternary_count} valid")
        
        return results
    
    def _select_salt_c_for_binary_pair(self, salt_a: pd.Series, salt_b: pd.Series,
                                       salt_a_list: List[pd.Series],
                                       salt_b_list: List[pd.Series],
                                       tri_reason: Dict[str, bool],
                                       strategy: Dict[str, Any],
                                       need_lower_tm: bool = False,
                                       need_higher_dh: bool = False,
                                       compatibility_agent: 'CompatibilityAgent' = None,
                                       limit: int = 1,
                                       exclude_salt_names: Set[str] = None) -> List[pd.Series]:
        """
        Select salt C candidates for a specific binary combination (salt A + salt B)
        
        Args:
            salt_a: Salt A
            salt_b: Salt B
            salt_a_list: Salt A candidate list (for extracting ion info)
            salt_b_list: Salt B candidate list (for extracting ion info)
            tri_reason: Ternary trigger reason
            strategy: Strategy configuration
            need_lower_tm: Whether to lower melting point
            need_higher_dh: Whether to increase enthalpy
            exclude_salt_names: Set of salt names to exclude (ensure different ratios select different salt C)
        """
        # Use CompatibilityAgent logic to select salt C
        # Reuse passed CompatibilityAgent instance, avoid repeated initialization
        if compatibility_agent is None:
            compatibility_agent = CompatibilityAgent(self.data_file)
        
        # Build candidates dict (containing only this specific salt A and B)
        candidates = {
            'salt_a_candidates': [salt_a],
            'salt_b_candidates': [salt_b]
        }
        
        # If need_lower_tm or need_higher_dh specified, adjust tri_reason
        if need_lower_tm or need_higher_dh:
            tri_reason = tri_reason.copy()
            tri_reason['perf_fail'] = True
        
        # First filter salt C from full candidate pool, then trim to limit in this function
        # This ensures "1 salt C per failed ratio", but salt C comes from full pool not top-N
        temp_limit = None
        salt_c_candidates = compatibility_agent._select_salt_c(candidates, strategy, tri_reason, limit=temp_limit)
        
        # If need_lower_tm or need_higher_dh specified, re-sort
        if need_lower_tm:
            # Sort by melting point ascending
            salt_c_candidates = sorted(salt_c_candidates, 
                                     key=lambda x: x.get('Tm (°C)', float('inf')) if pd.notna(x.get('Tm (°C)', None)) else float('inf'))
        elif need_higher_dh:
            # Sort by enthalpy descending
            salt_c_candidates = sorted(salt_c_candidates, 
                                     key=lambda x: x.get('ΔHm (J/g)', 0) if pd.notna(x.get('ΔHm (J/g)', None)) else 0, 
                                     reverse=True)
        
        # Exclude used salt C (by name matching, including same type with different hydration)
        if exclude_salt_names:
            filtered_candidates = []
            for salt_c in salt_c_candidates:
                salt_c_name = salt_c.get(compatibility_agent.salt_col, 'Unknown')
                # Check if same type as used salt C (ignoring hydration)
                is_excluded = any(are_same_salt_type(salt_c_name, used_name) for used_name in exclude_salt_names)
                if not is_excluded:
                    filtered_candidates.append(salt_c)
            
            salt_c_candidates = filtered_candidates

        # Limit return count (default 1 salt C per failed ratio)
        if limit is not None and len(salt_c_candidates) > limit:
            salt_c_candidates = salt_c_candidates[:limit]
        
        return salt_c_candidates
    
    def _generate_ternary_for_binary_pair(self, salt_a: pd.Series, salt_b: pd.Series,
                                         salt_c_list: List[pd.Series],
                                         ratio_a: float, ratio_b: float,
                                         tm_target: float, dh_target: float) -> Tuple[List[Dict[str, Any]], int]:
        """
        Generate ternary combinations and predictions for a specific binary pair (salt A + B) and salt C candidates
        
        Args:
            salt_a: Salt A
            salt_b: Salt B
            salt_c_list: Salt C candidate list
            ratio_a: Salt A ratio in binary (for logging, no longer used to limit ratios)
            ratio_b: Salt B ratio in binary (for logging, no longer used to limit ratios)
            tm_target: Target melting point
            dh_target: Target enthalpy
        
        Returns:
            (results, total_generated): Valid results list and theoretical generation count
        
        Note: After adding salt C, generates all possible ternary ratios (9 C ratios × 13 A/B ratios = 117 ratios)
        """
        from salt_combination_generator import build_feature_row, predict_with_models, load_structured_meta
        import numpy as np
        
        results = []
        processed = 0
        total_generated = 0  # Total theoretical generation count (including all attempted combinations)
        
        # Load metadata for compatibility detection
        try:
            meta_map = load_structured_meta('data')
        except Exception as e:
            logger.warning(f"Failed to load metadata, skipping compatibility detection: {e}")
            meta_map = None
        
        # Get metadata for salt A and B (only need once)
        salt_a_name = salt_a.get(self.salt_col, 'Unknown')
        salt_b_name = salt_b.get(self.salt_col, 'Unknown')
        mA = _resolve_meta_with_data_priority(meta_map or {}, salt_a_name, salt_a)
        mB = _resolve_meta_with_data_priority(meta_map or {}, salt_b_name, salt_b)
        
        # Generate all ternary ratios: salt C 10% to 50%, step 5%
        for c_ratio in np.arange(0.1, 0.51, 0.05):  # 10% to 50%, step 5%
            remaining = 1.0 - c_ratio
            # A and B ratio distribution in remaining portion: 20% to 80%, step 5%
            for ab_ratio in np.arange(0.2, 0.81, 0.05):  # 20% to 80%, step 5%
                new_ratio_a = remaining * ab_ratio
                new_ratio_b = remaining * (1 - ab_ratio)
                
                # Ensure sum equals 1 (considering floating point error)
                if abs(new_ratio_a + new_ratio_b + c_ratio - 1.0) >= 0.01:
                    continue
                
                for salt_c in salt_c_list:  # Use limited salt C candidates (up to 10)
                    try:
                        # Ensure salt C is not same as salt A or B
                        salt_c_name = salt_c.get(self.salt_col, 'Unknown')
                        
                        if (salt_c_name == salt_a_name or salt_c_name == salt_b_name):
                            continue
                        
                        # Theoretical generation count: all attempted combinations (including subsequently filtered)
                        total_generated += 1
                        processed += 1
                        
                        # Compatibility hard-filter: after adding salt C, only check AC/BC (no ABC overall filter)
                        mC = _resolve_meta_with_data_priority(meta_map or {}, salt_c_name, salt_c)

                        okAC, _ = evaluate_compatibility_hard_filter(mA, mC, cache=self._hard_filter_cache)
                        if not okAC:
                            continue

                        okBC, _ = evaluate_compatibility_hard_filter(mB, mC, cache=self._hard_filter_cache)
                        if not okBC:
                            continue

                        # Build features and predict
                        feat = build_feature_row([salt_a, salt_b, salt_c], 
                                                [new_ratio_a, new_ratio_b, c_ratio])
                        pred = predict_with_models(self.models, feat)
                        
                        # Check single peak
                        if pred['is_single_peak'] != 1 or pred['single_peak_proba'] < 0.5:
                            continue
                        
                        # Check performance
                        if pd.isna(pred['tm_pred']) or pd.isna(pred['dh_pred']):
                            continue
                        
                        tm_diff = abs(pred['tm_pred'] - tm_target)
                        dh_diff = abs(pred['dh_pred'] - dh_target)
                        
                        # Use judgment criteria consistent with 80% compliance standard
                        if tm_diff <= 6.0 and dh_diff <= 30.0:
                            score = tm_diff + dh_diff * 0.02
                            
                            results.append({
                                'salt_a': salt_a_name,
                                'salt_b': salt_b_name,
                                'salt_c': salt_c_name,
                                'ratio_a': round(new_ratio_a, 3),
                                'ratio_b': round(new_ratio_b, 3),
                                'ratio_c': round(c_ratio, 3),
                                'tm_pred': pred['tm_pred'],
                                'dh_pred': pred['dh_pred'],
                                'single_peak_proba': pred['single_peak_proba'],
                                'is_single_peak': pred['is_single_peak'],
                                'score': score,
                                'type': 'ternary',
                                'target_tm': tm_target,
                                'target_dh': dh_target,
                                'from_failed_binary': True  # Mark as originating from failed binary combination
                            })
                    
                    except Exception as e:
                        logger.warning(f"Ternary combination prediction failed: {e}")
                        continue
        
        return results, total_generated
    
    def _predict_ternary(self, salt_a_list: List[pd.Series], salt_b_list: List[pd.Series],
                        salt_c_list: List[pd.Series], ratios: List[Tuple[float, float, float]],
                        tm_target: float, dh_target: float) -> List[Dict[str, Any]]:
        """Predict ternary combinations"""
        from salt_combination_generator import build_feature_row, predict_with_models, load_structured_meta
        
        results = []
        processed = 0
        
        # Load metadata for compatibility detection
        try:
            meta_map = load_structured_meta('data')
        except Exception as e:
            logger.warning(f"Failed to load metadata, skipping compatibility detection: {e}")
            meta_map = None
        
        for salt_a in salt_a_list:
            for salt_b in salt_b_list:
                # Get metadata for salt A and B (only need once)
                salt_a_name = salt_a.get(self.salt_col, 'Unknown')
                salt_b_name = salt_b.get(self.salt_col, 'Unknown')
                mA = _resolve_meta_with_data_priority(meta_map or {}, salt_a_name, salt_a)
                mB = _resolve_meta_with_data_priority(meta_map or {}, salt_b_name, salt_b)
                
                for salt_c in salt_c_list:
                    for ratio_tuple in ratios:
                        
                        try:
                            # Ensure salt A, B, C are not same type (ignoring hydration)
                            salt_c_name = salt_c.get(self.salt_col, 'Unknown')
                            if (are_same_salt_type(salt_a_name, salt_b_name) or 
                                are_same_salt_type(salt_a_name, salt_c_name) or 
                                are_same_salt_type(salt_b_name, salt_c_name)):
                                processed += 1
                                continue
                            
                            # Compatibility hard-filter: after adding salt C, only check AC/BC (no ABC overall filter)
                            mC = _resolve_meta_with_data_priority(meta_map or {}, salt_c_name, salt_c)

                            okAC, _ = evaluate_compatibility_hard_filter(mA, mC, cache=self._hard_filter_cache)
                            if not okAC:
                                processed += 1
                                continue

                            okBC, _ = evaluate_compatibility_hard_filter(mB, mC, cache=self._hard_filter_cache)
                            if not okBC:
                                processed += 1
                                continue

                            ratio_a, ratio_b, ratio_c = ratio_tuple
                            
                            # Build features and predict
                            feat = build_feature_row([salt_a, salt_b, salt_c], [ratio_a, ratio_b, ratio_c])
                            pred = predict_with_models(self.models, feat)
                            
                            # Check single peak
                            if pred['is_single_peak'] != 1 or pred['single_peak_proba'] < 0.5:
                                processed += 1
                                continue
                            
                            # Check performance
                            if pd.isna(pred['tm_pred']) or pd.isna(pred['dh_pred']):
                                processed += 1
                                continue
                            
                            tm_diff = abs(pred['tm_pred'] - tm_target)
                            dh_diff = abs(pred['dh_pred'] - dh_target)
                            
                            # Use judgment criteria consistent with 80% compliance standard
                            if tm_diff <= 6.0 and dh_diff <= 30.0:
                                score = tm_diff + dh_diff * 0.02
                                
                                results.append({
                                    'salt_a': salt_a.get(self.salt_col, 'Unknown'),
                                    'salt_b': salt_b.get(self.salt_col, 'Unknown'),
                                    'salt_c': salt_c.get(self.salt_col, 'Unknown'),
                                    'ratio_a': ratio_a,
                                    'ratio_b': ratio_b,
                                    'ratio_c': ratio_c,
                                    'tm_pred': pred['tm_pred'],
                                    'dh_pred': pred['dh_pred'],
                                    'single_peak_proba': pred['single_peak_proba'],
                                    'is_single_peak': pred['is_single_peak'],
                                    'score': score,
                                    'type': 'ternary',
                                    'target_tm': tm_target,
                                    'target_dh': dh_target
                                })
                            
                            processed += 1
                            
                            if processed % 100 == 0:
                                logger.info(f"  Processed {processed} ternary combinations, found {len(results)} valid")
                        
                        except Exception as e:
                            logger.warning(f"Ternary combination prediction failed: {e}")
                            processed += 1
                            continue
        
        return results


class DecisionCoordinator:
    """Decision Coordinator - Coordinate agent execution flow and retry logic"""
    
    def __init__(self):
        logger.info("✓ Decision Coordinator initialized")
    
    def decide(self, predictions: List[Dict[str, Any]], 
               strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate prediction result quality and make decisions
        
        Return format:
        {
            'success': bool,
            'candidates': List[Dict],
            'categorized_results': Dict,
            'failure_reason': str (if failed)
        }
        """
        # Count valid binary and ternary results
        binary_count = len([p for p in predictions if p.get('type') == 'binary'])
        ternary_count = len([p for p in predictions if p.get('type') == 'ternary'])
        total_count = len(predictions)
        
        if binary_count > 0 and ternary_count > 0:
            logger.info(f"Decision Coordinator evaluating {total_count} predictions (binary valid {binary_count} + ternary valid {ternary_count})...")
        elif binary_count > 0:
            logger.info(f"Decision Coordinator evaluating {binary_count} predictions (binary valid {binary_count})...")
        elif ternary_count > 0:
            logger.info(f"Decision Coordinator evaluating {ternary_count} predictions (ternary valid {ternary_count})...")
        else:
            logger.info(f"Decision Coordinator evaluating {total_count} predictions...")
        
        if len(predictions) == 0:
            return {
                'success': False,
                'candidates': [],
                'categorized_results': {},
                'failure_reason': 'No valid combinations found'
            }
        
        # Remove duplicate results first
        unique_predictions = self._remove_duplicates(predictions)
        logger.info(f"After deduplication: {len(unique_predictions)} results remaining (from {len(predictions)})")
        
        # Categorize by compliance level
        categorized = self._categorize_by_compliance(unique_predictions)
        
        # Sort within each group by score (lower is better)
        for level in ['perfect', 'good', 'acceptable']:
            if level in categorized and len(categorized[level]) > 0:
                categorized[level] = sorted(categorized[level], key=lambda x: x.get('score', float('inf')))
        
        perfect_count = len(categorized.get('perfect', []))
        good_count = len(categorized.get('good', []))
        acceptable_count = len(categorized.get('acceptable', []))
        
        logger.info(f"Result categories: perfect={perfect_count}, 90% compliance={good_count}, 80% compliance={acceptable_count}")
        
        # Keep only results with ≥80% compliance (already sorted by score)
        valid_candidates = (
            categorized.get('perfect', []) + 
            categorized.get('good', []) + 
            categorized.get('acceptable', [])
        )
        
        if len(valid_candidates) > 0:
            return {
                'success': True,
                'candidates': valid_candidates,
                'categorized_results': categorized,
                'failure_reason': None
            }
        else:
            # Analyze failure reason
            failure_reason = self._analyze_failure_reason(unique_predictions, strategy)
            return {
                'success': False,
                'candidates': unique_predictions,  # Return all prediction results for analysis
                'categorized_results': categorized,
                'failure_reason': failure_reason
            }
    
    def _remove_duplicates(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate combination results
        
        Duplicate definition: same salt combination (salt_a, salt_b, salt_c) and ratios (ratio_a, ratio_b, ratio_c)
        Keep the one with lowest score
        """
        seen = {}
        
        for pred in predictions:
            # Build unique key: salt names + ratios
            salt_a = pred.get('salt_a', '')
            salt_b = pred.get('salt_b', '')
            salt_c = pred.get('salt_c', '')
            ratio_a = round(pred.get('ratio_a', 0), 3)
            ratio_b = round(pred.get('ratio_b', 0), 3)
            ratio_c = round(pred.get('ratio_c', 0), 3)
            
            key = (salt_a, salt_b, salt_c, ratio_a, ratio_b, ratio_c)
            
            current_score = pred.get('score', float('inf'))
            
            # If this combination not seen before, or new score is lower, keep/update
            if key not in seen or current_score < seen[key].get('score', float('inf')):
                seen[key] = pred
        
        return list(seen.values())
    
    def adjust_strategy(self, strategy: Dict[str, Any], 
                       failure_reason: str, attempt: int = 0) -> Dict[str, Any]:
        """Adjust strategy for retry"""
        logger.info(f"Adjusting strategy for retry, failure reason: {failure_reason}, attempt={attempt}")
        
        # Use strategy selector fallback mechanism, update attempt parameter
        new_strategy = strategy.copy()
        new_strategy['attempt'] = attempt
        
        # Keep original tolerance adjustment logic (for compatibility)
        new_strategy['tm_tolerance'] = min(strategy.get('tm_tolerance', 5.0) * 1.5, 15.0)
        new_strategy['dh_tolerance'] = min(strategy.get('dh_tolerance', 15.0) * 1.5, 30.0)
        new_strategy['strategy_level'] = 'relaxed'
        
        logger.info(f"Adjusted strategy: attempt={attempt}, tm_tolerance={new_strategy['tm_tolerance']}, dh_tolerance={new_strategy['dh_tolerance']}")
        
        return new_strategy
    
    def _analyze_failure_reason(self, predictions: List[Dict[str, Any]], strategy: Dict[str, Any]) -> str:
        """
        Analyze specific reasons for binary prediction failure
        
        Return format:
        - "Single peak failure: cannot form single-peak eutectic structure" (if single peak probability < 0.5)
        - "Performance insufficient: need to lower melting point" or "Performance insufficient: need to increase enthalpy"
        """
        if len(predictions) == 0:
            return "No valid combinations found"
        
        tm_target = strategy.get('tm_target', 0)
        dh_target = strategy.get('dh_target', 0)
        
        # Count various failure categories
        no_single_peak_count = 0
        perf_fail_count = 0
        perf_fail_lower_tm = 0
        perf_fail_higher_dh = 0
        
        for pred in predictions:
            # Check single peak issue: single peak probability < 0.5
            single_peak_proba = pred.get('single_peak_proba', 0)
            is_single_peak = pred.get('is_single_peak', 0)
            
            if is_single_peak != 1 or single_peak_proba < 0.5:
                no_single_peak_count += 1
                continue
            
            # Check performance issue (only analyze combinations that passed single peak check)
            tm_pred = pred.get('tm_pred')
            dh_pred = pred.get('dh_pred')
            
            if pd.notna(tm_pred) and pd.notna(dh_pred):
                tm_error = abs(tm_pred - tm_target)
                dh_error = abs(dh_pred - dh_target)
                
                # If not meeting ≥80% compliance (>6°C or >30 J/g), considered performance issue
                if tm_error > 6.0 or dh_error > 30.0:
                    perf_fail_count += 1
                    
                    # Determine if need lower Tm or higher ΔH
                    if tm_pred > tm_target:
                        perf_fail_lower_tm += 1
                    if dh_pred < dh_target:
                        perf_fail_higher_dh += 1
        
        total = len(predictions)
        
        # Determine primary failure reason
        # If all combinations are non-single-peak (probability < 0.5), determine as single peak failure
        if no_single_peak_count == total:
            return "Single peak failure: cannot form single-peak eutectic structure"
        elif perf_fail_count > 0:
            # Performance insufficient: determine need lower Tm or higher ΔH
            if perf_fail_lower_tm > perf_fail_higher_dh:
                return "Performance insufficient: need to lower melting point"
            elif perf_fail_higher_dh > perf_fail_lower_tm:
                return "Performance insufficient: need to increase enthalpy"
            else:
                # If both, prioritize the more severe one
                return "Performance insufficient: need to lower melting point"  # Default: prioritize lowering Tm
        else:
            # If all passed single peak and performance, but none ≥80%, default to performance insufficient
            return "Performance insufficient: need to lower melting point"
    
    def _categorize_by_compliance(self, candidates: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize results by compliance level"""
        categorized = {
            'perfect': [],
            'good': [],
            'acceptable': []
        }
        
        for candidate in candidates:
            # Skip non-single-peak or prediction-failed combinations (score is inf)
            if candidate.get('score') == float('inf'):
                candidate['compliance_level'] = 0.0
                candidate['compliance_desc'] = 'Invalid combination (non-single-peak or prediction failed)'
                continue
            
            tm_pred = candidate.get('tm_pred')
            dh_pred = candidate.get('dh_pred')
            target_tm = candidate.get('target_tm')
            target_dh = candidate.get('target_dh')
            
            # Check data validity
            if pd.isna(tm_pred) or pd.isna(dh_pred) or target_tm is None or target_dh is None:
                candidate['compliance_level'] = 0.0
                candidate['compliance_desc'] = 'Missing data'
                continue
            
            tm_error = abs(tm_pred - target_tm)
            dh_error = abs(dh_pred - target_dh)
            
            if tm_error <= 2.0 and dh_error <= 10.0:
                candidate['compliance_level'] = 1.0
                candidate['compliance_desc'] = 'Fully compliant'
                categorized['perfect'].append(candidate)
            elif tm_error <= 4.0 and dh_error <= 20.0:
                candidate['compliance_level'] = 0.9
                candidate['compliance_desc'] = '90% compliant'
                categorized['good'].append(candidate)
            elif tm_error <= 6.0 and dh_error <= 30.0:
                candidate['compliance_level'] = 0.8
                candidate['compliance_desc'] = '80% compliant'
                categorized['acceptable'].append(candidate)
            else:
                candidate['compliance_level'] = 0.7
                candidate['compliance_desc'] = 'Low compliance'
        
        return categorized


class MultiAgentController:
    """Multi-Agent Main Controller"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.analysis_agent = AnalysisAgent()
        self.retrieval_agent = RetrievalAgent(data_file)
        self.compatibility_agent = CompatibilityAgent(data_file)  # Pass data file
        self.ratio_agent = RatioFunnelAgent()
        self.ternary_agent = TernaryAgent(data_file)
        self.coordinator = DecisionCoordinator()
        
        logger.info("🚀 Multi-Agent Controller initialized")
    
    def _determine_tri_reason(self, failure_reason: str) -> Dict[str, bool]:
        """
        Determine ternary trigger reason from failure reason
        
        Args:
            failure_reason: Failure reason string
        
        Returns:
            tri_reason: Ternary trigger reason dict
        """
        tri_reason = {
            'no_monopeak': False,
            'perf_fail': False,
            'struct_fail': False
        }
        
        if not failure_reason:
            return tri_reason
        
        failure_reason_lower = failure_reason.lower()
        
        # Check single peak issue
        if 'single_peak' in failure_reason_lower or 'monopeak' in failure_reason_lower:
            tri_reason['no_monopeak'] = True
            logger.info("Single peak issue detected, will use low-Tm salt strategy")
        
        # Check performance issue
        elif 'performance' in failure_reason_lower:
            tri_reason['perf_fail'] = True
            logger.info("Performance issue detected, will use high-enthalpy salt strategy")
        
        # Check structural compatibility issue
        elif 'structure' in failure_reason_lower or 'compatible' in failure_reason_lower:
            tri_reason['struct_fail'] = True
            logger.info("Structural compatibility issue detected, will use compatible salt strategy")
        
        # Default: if failure reason is unclear, use compatibility strategy
        else:
            logger.info("Failure reason not clearly identified, using default compatibility strategy")
        
        return tri_reason
    
    def run(self, tm_input: str, dh_input: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Run the complete multi-agent screening process
        
        Return format:
        {
            'success': bool,
            'strategy': Dict,
            'candidates': List[Dict],
            'categorized_results': Dict,
            'message': str
        }
        """
        logger.info(f"\n{'='*60}")
        logger.info("Starting multi-agent salt screening process")
        logger.info(f"{'='*60}\n")
        
        # Step 1: Analysis Agent
        logger.info("[Step 1/7] Analysis Agent: parsing user input")
        strategy = self.analysis_agent.analyze(tm_input, dh_input)
        
        # Step 2: Retrieval Agent
        logger.info("[Step 2/7] Retrieval Agent: screening candidate salts")
        attempt = strategy.get('attempt', 0)
        candidates = self.retrieval_agent.retrieve_salts(strategy, attempt)
        
        # Step 3: Compatibility Agent (first run: no salt C selection, compatibility check only)
        logger.info("[Step 3/7] Compatibility Agent: evaluating eutectic structure (binary only)")
        # Perform compatibility check, return compatible pairs set (retain all candidates)
        candidates, compatible_pairs = self.compatibility_agent.filter_compatible_pairs(candidates, relax_round=attempt)
        # First run: no salt C, return empty compatibility result
        compatibility_result = {
            'compatible': True,
            'need_salt_c': False,
            'salt_c_candidates': [],  # First run: no salt C selection
            'reason': 'First run, binary prediction only'
        }
        
        # Step 4: Ratio Funnel Agent
        logger.info("[Step 4/7] Ratio Funnel Agent: generating ratio schemes")
        ratios = self.ratio_agent.generate_ratios(compatibility_result)
        
        # Step 5: Ternary Agent - try binary combinations first
        logger.info("[Step 5/7] Ternary Agent: XGBoost prediction and validation (binary first)")
        predictions = self.ternary_agent.predict_and_validate(
            candidates, ratios, strategy, compatibility_result, try_ternary=False, compatible_pairs=compatible_pairs
        )
        
        logger.info("[Step 6/7] Decision analysis: evaluating result quality")
        final_result = self.coordinator.decide(predictions, strategy)
        
        # Regardless of binary success, extract unqualified binary combinations for ternary generation
        # Use same criteria as predict_and_validate
        invalid_binary_from_first_run = []
        tm_target = strategy['tm_target']
        dh_target = strategy['dh_target']
        for pred in predictions:
            # Only keep binary combinations (type='binary')
            if pred.get('type') == 'binary':
                # Use same criteria as predict_and_validate: performance qualification
                # Performance criteria: score != inf AND Tm error ≤6°C AND ΔH error ≤30 J/g (consistent with 80% compliance)
                score = pred.get('score', float('inf'))
                tm_pred = pred.get('tm_pred', 0)
                dh_pred = pred.get('dh_pred', 0)
                
                # Judge if performance-qualified (consistent with predict_and_validate criteria)
                is_valid = (
                    score != float('inf') and
                    abs(tm_pred - tm_target) <= 6.0 and
                    abs(dh_pred - dh_target) <= 30.0
                )
                
                # Unqualified binary: not meeting performance criteria
                if not is_valid:
                    invalid_binary_from_first_run.append(pred)
        
        # If binary successful (has ≥80% compliance), can output directly, but also generate ternary for unqualified
        if final_result['success']:
            logger.info(f"✓ Binary success: found {len(final_result['candidates'])} combinations with ≥80% compliance")
            logger.info("Successful binary combinations can be output directly, also generating ternary for unqualified")
            
            # Generate ternary for unqualified binary combinations
            if len(invalid_binary_from_first_run) > 0:
                logger.info(f"\nGenerating ternary for {len(invalid_binary_from_first_run)} unqualified binary combinations...")
                
                # Determine trigger reason from failure reason (analyze unqualified combination reasons)
                # Count failure reasons for unqualified combinations
                no_monopeak_count = 0
                perf_fail_count = 0
                for invalid_pred in invalid_binary_from_first_run:
                    single_peak_proba = invalid_pred.get('single_peak_proba', 0)
                    is_single_peak = invalid_pred.get('is_single_peak', 0)
                    if is_single_peak != 1 or single_peak_proba < 0.5:
                        no_monopeak_count += 1
                    else:
                        perf_fail_count += 1
                
                # Determine primary failure reason
                if no_monopeak_count > perf_fail_count:
                    tri_reason = {'no_monopeak': True, 'perf_fail': False, 'struct_fail': False}
                else:
                    tri_reason = {'no_monopeak': False, 'perf_fail': True, 'struct_fail': False}
                
                # Generate ternary for unqualified binary combinations
                ternary_results = self.ternary_agent._augment_invalid_binary_with_salt_c(
                    invalid_binary_from_first_run, 
                    candidates.get('salt_a_candidates', []), 
                    candidates.get('salt_b_candidates', []),
                    tm_target, 
                    dh_target, 
                    strategy
                )
                
                # Merge ternary results with previous predictions
                all_predictions = predictions + ternary_results
                
                # Re-evaluate (including ternary results)
                final_result = self.coordinator.decide(all_predictions, strategy)
                logger.info(f"✓ After merging, found {len(final_result['candidates'])} combinations with ≥80% compliance (including ternary)")
        else:
            # If binary failed, try ternary combinations
            logger.info("\n" + "="*60)
            logger.info("Binary combinations found no ≥80% compliance results, starting ternary strategy")
            logger.info("="*60)
            
            # Determine trigger reason from failure reason
            tri_reason = self._determine_tri_reason(final_result['failure_reason'])
            
            # Use unqualified combinations from previous predictions to avoid redundant computation
            if len(invalid_binary_from_first_run) > 0:
                logger.info(f"Using {len(invalid_binary_from_first_run)} unqualified binary combinations from previous predictions for ternary...")
                ternary_results = self.ternary_agent._augment_invalid_binary_with_salt_c(
                    invalid_binary_from_first_run, 
                    candidates.get('salt_a_candidates', []), 
                    candidates.get('salt_b_candidates', []),
                    strategy['tm_target'], 
                    strategy['dh_target'], 
                    strategy
                )
                
                # Merge ternary results with previous predictions
                ternary_predictions = predictions + ternary_results
            else:
                # If no saved unqualified combinations, fallback to re-prediction (should not happen)
                logger.warning("No unqualified binary combinations found, falling back to re-prediction...")
                compatibility_result = self.compatibility_agent.judge(candidates, strategy, tri_reason)
                ratios = self.ratio_agent.generate_ratios(compatibility_result)
                ternary_predictions = self.ternary_agent.predict_and_validate(
                    candidates, ratios, strategy, compatibility_result, try_ternary=True, compatible_pairs=compatible_pairs
                )
            
            # Re-evaluate (including ternary results)
            final_result = self.coordinator.decide(ternary_predictions, strategy)
        
        # Retry loop (using fallback mechanism)
        retry_count = 0
        while not final_result['success'] and retry_count < max_retries:
            retry_count += 1
            attempt = retry_count  # attempt starts from 1 (0 is the initial attempt)
            logger.info(f"\n{'='*60}")
            logger.info(f"Retry attempt #{retry_count} (attempt={attempt})...")
            logger.info(f"{'='*60}\n")
            
            # Adjust strategy (pass attempt parameter)
            strategy = self.coordinator.adjust_strategy(strategy, final_result['failure_reason'], attempt)
            
            # Determine trigger reason from failure reason
            tri_reason = self._determine_tri_reason(final_result['failure_reason'])
            
            # Re-retrieve and predict (pass attempt parameter)
            candidates = self.retrieval_agent.retrieve_salts(strategy, attempt)
            
            # Re-run compatibility check
            candidates, compatible_pairs_retry = self.compatibility_agent.filter_compatible_pairs(candidates, relax_round=attempt)
            
            # Try binary combinations first (no salt C)
            compatibility_result = {
                'compatible': True,
                'need_salt_c': False,
                'salt_c_candidates': [],
                'reason': 'Retry round, trying binary first'
            }
            ratios = self.ratio_agent.generate_ratios(compatibility_result)
            predictions = self.ternary_agent.predict_and_validate(
                candidates, ratios, strategy, compatibility_result, try_ternary=False, compatible_pairs=compatible_pairs_retry
            )
            final_result = self.coordinator.decide(predictions, strategy)
            
            # Regardless of binary success, extract unqualified binary combinations for ternary generation
            # Use same criteria as predict_and_validate
            invalid_binary_from_retry = []
            tm_target = strategy['tm_target']
            dh_target = strategy['dh_target']
            for pred in predictions:
                # Only keep binary combinations (type='binary')
                if pred.get('type') == 'binary':
                    # Use same criteria as predict_and_validate: performance qualification
                    # Performance criteria: score != inf AND Tm error ≤6°C AND ΔH error ≤30 J/g (consistent with 80% compliance)
                    score = pred.get('score', float('inf'))
                    tm_pred = pred.get('tm_pred', 0)
                    dh_pred = pred.get('dh_pred', 0)
                    
                    # Judge if performance-qualified (consistent with predict_and_validate criteria)
                    is_valid = (
                        score != float('inf') and
                        abs(tm_pred - tm_target) <= 6.0 and
                        abs(dh_pred - dh_target) <= 30.0
                    )
                    
                    # Unqualified binary: not meeting performance criteria
                    if not is_valid:
                        invalid_binary_from_retry.append(pred)
            
            # If binary successful, also generate ternary for unqualified
            if final_result['success']:
                logger.info(f"✓ Binary success: found {len(final_result['candidates'])} combinations with ≥80% compliance")
                logger.info("Successful binary combinations can be output directly, also generating ternary for unqualified")
                
                # Generate ternary for unqualified binary combinations
                if len(invalid_binary_from_retry) > 0:
                    logger.info(f"Generating ternary for {len(invalid_binary_from_retry)} unqualified binary combinations...")
                    
                    # Count failure reasons for unqualified combinations
                    no_monopeak_count = 0
                    perf_fail_count = 0
                    for invalid_pred in invalid_binary_from_retry:
                        single_peak_proba = invalid_pred.get('single_peak_proba', 0)
                        is_single_peak = invalid_pred.get('is_single_peak', 0)
                        if is_single_peak != 1 or single_peak_proba < 0.5:
                            no_monopeak_count += 1
                        else:
                            perf_fail_count += 1
                    
                    # Determine primary failure reason
                    if no_monopeak_count > perf_fail_count:
                        tri_reason = {'no_monopeak': True, 'perf_fail': False, 'struct_fail': False}
                    else:
                        tri_reason = {'no_monopeak': False, 'perf_fail': True, 'struct_fail': False}
                    
                    # Generate ternary for unqualified binary combinations
                    ternary_results = self.ternary_agent._augment_invalid_binary_with_salt_c(
                        invalid_binary_from_retry, 
                        candidates.get('salt_a_candidates', []), 
                        candidates.get('salt_b_candidates', []),
                        tm_target, 
                        dh_target, 
                        strategy
                    )
                    
                    # Merge ternary results with previous predictions
                    all_predictions = predictions + ternary_results
                    
                    # Re-evaluate (including ternary results)
                    final_result = self.coordinator.decide(all_predictions, strategy)
                    
                    # Count binary and ternary combinations
                    binary_count = len([c for c in final_result['candidates'] if c.get('type') == 'binary'])
                    ternary_count = len([c for c in final_result['candidates'] if c.get('type') == 'ternary'])
                    logger.info(f"✓ After merging, found {len(final_result['candidates'])} combinations with ≥80% compliance")
                    logger.info(f"  Including: {binary_count} binary, {ternary_count} ternary")
            else:
                # If binary failed, try ternary combinations
                logger.info("Binary combinations failed, trying ternary...")
                # Determine trigger reason from failure reason
                tri_reason = self._determine_tri_reason(final_result['failure_reason'])
                
                # Use unqualified combinations from previous predictions to avoid redundant computation
                if len(invalid_binary_from_retry) > 0:
                    logger.info(f"Using {len(invalid_binary_from_retry)} unqualified binary combinations from previous predictions for ternary...")
                    ternary_results = self.ternary_agent._augment_invalid_binary_with_salt_c(
                        invalid_binary_from_retry, 
                        candidates.get('salt_a_candidates', []), 
                        candidates.get('salt_b_candidates', []),
                        strategy['tm_target'], 
                        strategy['dh_target'], 
                        strategy
                    )
                    
                    # Merge ternary results with previous predictions
                    ternary_predictions = predictions + ternary_results
                else:
                    # If no saved unqualified combinations, fallback to re-prediction
                    logger.warning("No unqualified binary combinations found, falling back to re-prediction...")
                    compatibility_result = self.compatibility_agent.judge(candidates, strategy, tri_reason)
                    ratios = self.ratio_agent.generate_ratios(compatibility_result)
                    ternary_predictions = self.ternary_agent.predict_and_validate(
                        candidates, ratios, strategy, compatibility_result, try_ternary=True, compatible_pairs=compatible_pairs_retry
                    )
                
                final_result = self.coordinator.decide(ternary_predictions, strategy)
        
        # Step 7: Output results (no model update)
        logger.info("[Step 7/7] Output results")
        
        if final_result['success']:
            logger.info(f"✅ Screening successful! Found {len(final_result['candidates'])} combinations with ≥80% compliance")
            return {
                'success': True,
                'strategy': strategy,
                'candidates': final_result['candidates'],
                'categorized_results': final_result['categorized_results'],
                'message': f"Successfully found {len(final_result['candidates'])} valid salt combinations",
                'retries': retry_count
            }
        else:
            logger.warning(f"❌ Screening failed after {retry_count} retries")
            return {
                'success': False,
                'strategy': strategy,
                'candidates': final_result.get('candidates', []),
                'categorized_results': final_result.get('categorized_results', {}),
                'message': f"No qualifying salt combinations found: {final_result.get('failure_reason', 'unknown reason')}",
                'retries': retry_count
            }

