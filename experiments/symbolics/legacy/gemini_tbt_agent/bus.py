"""
bus.py
Thalamocortical I/O Bus with Lateral Impulse Routing & Epiplexity Reporting
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from arc3_engine import GameAction


@dataclass(frozen=True)
class ColumnInputBus:
    """
    STRICT INPUT BUS:
    The ONLY information permitted to enter a Cortical Column.
    """
    entity_color: int
    efference_action: Optional[GameAction] = None
    impulse_vector: Optional[Tuple[int, int]] = None
    contact_probe: int = 0
    actual_delta: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class ColumnOutputBus:
    """
    STRICT OUTPUT BUS:
    The ONLY information emitted by a Cortical Column.
    """
    entity_color: int
    predicted_delta: Tuple[int, int]
    attn_routing: Dict[str, float]
    
    # Lateral impulse to be transferred to the contacted entity (if any)
    transferred_impulse: Optional[Tuple[Tuple[int, int], int]] = None
    
    # Epiplexic value: learnable structural uncertainty for the probed feature regime
    epiplexity: float = 0.0
    
    prediction_error: float = 0.0


class ThalamicProbe:
    """Samples local leading edges and constructs standard thalamocortical buses."""

    @staticmethod
    def extract_probe_color(
        grid: List[List[int]],
        current_pos: Tuple[int, int],
        action_delta: Tuple[int, int]
    ) -> int:
        target_x = current_pos[0] + action_delta[0]
        target_y = current_pos[1] + action_delta[1]
        
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        if 0 <= target_x < width and 0 <= target_y < height:
            return grid[target_y][target_x]
        return -1  # Boundary collision

    @classmethod
    def create_query_bus(
        cls,
        entity_color: int,
        grid: List[List[int]],
        current_pos: Tuple[int, int],
        action: Optional[GameAction] = None,
        impulse: Optional[Tuple[int, int]] = None,
        motor_prior: Tuple[int, int] = (0, 0)
    ) -> ColumnInputBus:
        delta = motor_prior if action is not None else (impulse or (0, 0))
        probe_color = cls.extract_probe_color(grid, current_pos, delta)
        return ColumnInputBus(
            entity_color=entity_color,
            efference_action=action,
            impulse_vector=impulse,
            contact_probe=probe_color,
            actual_delta=None
        )

    @classmethod
    def create_feedback_bus(
        cls,
        entity_color: int,
        action: Optional[GameAction],
        probe_color: int,
        actual_delta: Tuple[int, int],
        impulse: Optional[Tuple[int, int]] = None
    ) -> ColumnInputBus:
        return ColumnInputBus(
            entity_color=entity_color,
            efference_action=action,
            impulse_vector=impulse,
            contact_probe=probe_color,
            actual_delta=actual_delta
        )