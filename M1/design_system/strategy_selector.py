"""
Strategy selector module
Implements target-decomposition-based dual salt strategy system
Salt A: Temperature-tuning → controls overall eutectic melting point
Salt B: Enthalpy-tuning → enhances eutectic enthalpy
"""

import pandas as pd
import logging
from typing import Dict, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StrategyConfig:
    """Strategy configuration class"""
    tm_max: float  # Target melting point upper limit
    dh_min: float  # Target enthalpy lower limit
    attempt: int = 0  # Fallback attempt count (0, 1, 2, 3)
    
    def get_salt_a_range(self) -> Tuple[float, float]:
        """Get salt A melting point range"""
        if self.attempt == 0:
            # Initial setting: salt A strategy: Tm slightly above target
            return (max(0, self.tm_max + 1), self.tm_max + 50)
        elif self.attempt == 1:
            # Fallback attempt 1: salt A unchanged
            return (max(0, self.tm_max + 1), self.tm_max + 50)
        elif self.attempt == 2:
            # Fallback attempt 2: salt A Tm lower bound relaxed
            return (max(0, self.tm_max - 5), self.tm_max + 50)
        else:  # attempt >= 3
            # Fallback attempt 3: salt A Tm lower bound relaxed
            return (max(0, self.tm_max - 5), self.tm_max + 50)
    
    def get_salt_b_dh_min(self) -> float:
        """Get salt B minimum enthalpy"""
        if self.attempt == 0:
            # Initial setting: salt B strategy: enthalpy meets target
            return self.dh_min * 1.0
        elif self.attempt == 1:
            # Fallback attempt 1: salt B enthalpy tolerance lowered
            return self.dh_min * 0.9
        elif self.attempt == 2:
            # Fallback attempt 2: salt B maintains 90% tolerance
            return self.dh_min * 0.9
        else:  # attempt >= 3
            # Fallback attempt 3: salt B enthalpy further relaxed
            return self.dh_min * 0.8
    
    def get_compliance_level(self) -> float:
        """Get current attempt compliance level"""
        if self.attempt == 0:
            return 0.8  # 80% compliance
        elif self.attempt == 1:
            return 0.7  # 70% compliance
        elif self.attempt == 2:
            return 0.6  # 60% compliance
        else:  # attempt >= 3
            return 0.5  # 50% compliance

class StrategySelector:
    """Strategy selector main class"""
    
    def __init__(self, df: pd.DataFrame, salt_col: str):
        self.df = df
        self.salt_col = salt_col
        logger.info(f"Strategy selector initialized, data size: {len(df)}")
    
    def select_salt_a_candidates(self, config: StrategyConfig) -> pd.DataFrame:
        """Select salt A candidates - temperature tuning"""
        tm_min, tm_max = config.get_salt_a_range()
        
        # Filter valid melting point data
        df_clean = self.df[pd.notna(self.df['Tm (°C)'])].copy()
        
        # Apply melting point range filter
        condition = (df_clean['Tm (°C)'] >= tm_min) & (df_clean['Tm (°C)'] <= tm_max)
        candidates = df_clean[condition].copy()
        
        # Sort by Tm, prioritize those closest to target
        target_tm = config.tm_max
        candidates['tm_score'] = abs(candidates['Tm (°C)'] - target_tm)
        candidates = candidates.sort_values('tm_score')
        
        logger.info(f"Salt A candidate screening (attempt={config.attempt}): "
                    f"Tm range [{tm_min:.1f}, {tm_max:.1f}] °C, found {len(candidates)} candidates")
        
        return candidates
    
    def select_salt_b_candidates(self, config: StrategyConfig) -> pd.DataFrame:
        """Select salt B candidates - enthalpy tuning"""
        dh_min = config.get_salt_b_dh_min()
        
        # Filter valid enthalpy data
        df_clean = self.df[pd.notna(self.df['ΔHm (J/g)'])].copy()
        
        # Apply enthalpy filter
        condition = df_clean['ΔHm (J/g)'] >= dh_min
        candidates = df_clean[condition].copy()
        
        # Sort by enthalpy descending, prioritize high enthalpy
        candidates = candidates.sort_values('ΔHm (J/g)', ascending=False)
        
        logger.info(f"Salt B candidate screening (attempt={config.attempt}): "
                    f"ΔHm >= {dh_min:.1f} J/g, found {len(candidates)} candidates")
        
        return candidates
    
    def evaluate_strategy_effectiveness(self, salt_a_count: int, salt_b_count: int, 
                                      config: StrategyConfig) -> Dict[str, Any]:
        """Evaluate strategy effectiveness"""
        effectiveness = {
            'salt_a_count': salt_a_count,
            'salt_b_count': salt_b_count,
            'total_combinations': salt_a_count * salt_b_count,
            'attempt': config.attempt,
            'compliance_level': config.get_compliance_level(),
            'strategy_viable': salt_a_count > 0 and salt_b_count > 0,
            'combination_density': 'high' if salt_a_count * salt_b_count > 50 else 
                                 'medium' if salt_a_count * salt_b_count > 10 else 'low'
        }
        
        # Strategy recommendation
        if not effectiveness['strategy_viable']:
            effectiveness['recommendation'] = 'Insufficient candidate salts, need further fallback or target adjustment'
        elif effectiveness['total_combinations'] > 100:
            effectiveness['recommendation'] = 'Too many candidate combinations, suggest further filtering'
        else:
            effectiveness['recommendation'] = 'Strategy viable, can proceed with combination evaluation'
        
        logger.info(f"Strategy evaluation (attempt={config.attempt}): "
                    f"salt A {salt_a_count}, salt B {salt_b_count}, "
                    f"compliance {effectiveness['compliance_level']}, {effectiveness['recommendation']}")
        
        return effectiveness
    
    def get_strategy_summary(self, config: StrategyConfig) -> Dict[str, str]:
        """Get strategy summary"""
        tm_min, tm_max = config.get_salt_a_range()
        dh_min = config.get_salt_b_dh_min()
        compliance = config.get_compliance_level()
        
        summary = {
            'attempt_level': f"Fallback attempt {config.attempt}",
            'compliance_level': f"{compliance*100:.0f}% compliance",
            'salt_a_strategy': f"Temperature-tuning: select salts with Tm in {tm_min:.1f}~{tm_max:.1f} °C",
            'salt_b_strategy': f"Enthalpy-tuning: select salts with ΔH >= {dh_min:.1f} J/g",
            'strategy_logic': self._get_strategy_logic_description(config.attempt)
        }
        
        return summary
    
    def _get_strategy_logic_description(self, attempt: int) -> str:
        """Get strategy logic description"""
        descriptions = {
            0: "Initial setting: salt A Tm slightly above target [Tm_target+1, Tm_target+50], salt B enthalpy meets target (100%)",
            1: "Fallback attempt 1: salt A unchanged [Tm_max+1, Tm_max+50], salt B enthalpy tolerance lowered to 90%",
            2: "Fallback attempt 2: salt A Tm lower bound relaxed by 5 °C [Tm_max-5, Tm_max+50], salt B maintains 90% tolerance", 
            3: "Fallback attempt 3: salt A Tm lower bound relaxed by 5 °C [Tm_max-5, Tm_max+50], salt B enthalpy tolerance lowered to 80%"
        }
        return descriptions.get(attempt, "Unknown strategy")

def create_strategy_config(tm_target: float, dh_target: float, attempt: int = 0) -> StrategyConfig:
    """Create strategy configuration"""
    return StrategyConfig(
        tm_max=tm_target,
        dh_min=dh_target,
        attempt=attempt
    )


