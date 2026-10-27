"""
column.py
Symbolic Cortical Column with AttnRes and Epiplexity Drive (No Coordinate Lists)
"""

import math
from typing import Dict, Tuple, Optional
from arc3_engine import GameAction
from bus import ColumnInputBus, ColumnOutputBus


class SymbolicCorticalColumn:
    def __init__(self, column_id: int, tuned_color: int, temperature: float = 0.5):
        self.id = column_id
        self.tuned_color = tuned_color
        self.temperature = temperature

        # Layer 1: Motor Impulses
        self.layer1_motor: Dict[GameAction, Tuple[int, int]] = {
            GameAction.ACTION1: (0, -1),
            GameAction.ACTION2: (0, 1),
            GameAction.ACTION3: (-1, 0),
            GameAction.ACTION4: (1, 0),
        }

        # Layer 2: Residual Corrections
        self.layer2_residuals: Dict[Tuple[GameAction, int], Tuple[int, int]] = {}

        # AttnRes Routing Logits: color -> {'free': score, 'blocked': score, 'residual': score}
        self.routing_logits: Dict[int, Dict[str, float]] = {}
        
        # Observation evidence count per feature regime (NOT coordinates)
        self.observation_counts: Dict[int, int] = {}

    def _init_color_logits(self, color: int):
        if color not in self.routing_logits:
            # Default prior: uniform uncommitted uncertainty
            self.routing_logits[color] = {"free": 1.0, "blocked": 0.0, "residual": 0.0}

    def get_attention_weights(self, color: int) -> Dict[str, float]:
        self._init_color_logits(color)
        logits = self.routing_logits[color]
        max_l = max(logits.values())
        exp_vals = {k: math.exp((v - max_l) / self.temperature) for k, v in logits.items()}
        sum_exp = sum(exp_vals.values())
        return {k: v / sum_exp for k, v in exp_vals.items()}

    def compute_epiplexity(self, color: int) -> float:
        """
        Epiplexity = Learnable Structural Uncertainty.
        High when routing distribution is uncertain / uncommitted.
        Collapses to 0.0 as observations accumulate and entropy falls.
        """
        attn = self.get_attention_weights(color)
        # Shannon entropy across the 3 functional regimes
        entropy = -sum(p * math.log(p + 1e-9) for p in attn.values())
        max_entropy = math.log(3.0)
        norm_entropy = entropy / max_entropy

        # Bounded by observation count for this feature
        obs_count = self.observation_counts.get(color, 0)
        epiplexity = norm_entropy / (1.0 + 0.5 * obs_count)
        return max(0.0, min(1.0, epiplexity))

    def forward(self, bus_in: ColumnInputBus) -> ColumnOutputBus:
        color = bus_in.contact_probe
        attn = self.get_attention_weights(color)
        epiplexity = self.compute_epiplexity(color)

        if bus_in.efference_action is not None:
            base_dx, base_dy = self.layer1_motor.get(bus_in.efference_action, (0, 0))
            res_dx, res_dy = self.layer2_residuals.get((bus_in.efference_action, color), (0, 0))
        elif bus_in.impulse_vector is not None:
            base_dx, base_dy = bus_in.impulse_vector
            res_dx, res_dy = (0, 0)
        else:
            base_dx, base_dy = (0, 0)
            res_dx, res_dy = (0, 0)

        net_dx = attn["free"] * base_dx + attn["residual"] * (base_dx + res_dx)
        net_dy = attn["free"] * base_dy + attn["residual"] * (base_dy + res_dy)
        final_delta = (int(round(net_dx)), int(round(net_dy)))

        # Lateral impulse generation:
        # Activated either when residual regime is established OR when epiplexity is high
        # (the column tests the affordance of an unmapped entity).
        transferred = None
        if color > 0 and (attn["residual"] > 0.40 or epiplexity > 0.40):
            transferred = ((base_dx, base_dy), color)

        pred_error = 0.0
        if bus_in.actual_delta is not None:
            pred_error = math.sqrt(
                (final_delta[0] - bus_in.actual_delta[0]) ** 2 +
                (final_delta[1] - bus_in.actual_delta[1]) ** 2
            )

        return ColumnOutputBus(
            entity_color=self.tuned_color,
            predicted_delta=final_delta,
            attn_routing=attn,
            transferred_impulse=transferred,
            epiplexity=epiplexity,
            prediction_error=pred_error
        )

    def update(self, bus_in: ColumnInputBus) -> ColumnOutputBus:
        if bus_in.actual_delta is None:
            return self.forward(bus_in)

        color = bus_in.contact_probe
        actual = bus_in.actual_delta
        self._init_color_logits(color)
        self.observation_counts[color] = self.observation_counts.get(color, 0) + 1

        if bus_in.efference_action is not None:
            base_delta = self.layer1_motor.get(bus_in.efference_action, (0, 0))
        else:
            base_delta = bus_in.impulse_vector or (0, 0)

        if actual == base_delta and actual != (0, 0):
            self.routing_logits[color]["free"] += 3.0
            self.routing_logits[color]["blocked"] = max(0.0, self.routing_logits[color]["blocked"] - 1.0)
        elif actual == (0, 0):
            self.routing_logits[color]["blocked"] += 4.0
            self.routing_logits[color]["free"] = max(0.0, self.routing_logits[color]["free"] - 2.0)
        else:
            if bus_in.efference_action is not None:
                residual = (actual[0] - base_delta[0], actual[1] - base_delta[1])
                self.layer2_residuals[(bus_in.efference_action, color)] = residual
            self.routing_logits[color]["residual"] += 4.0

        return self.forward(bus_in)

    def render_routing_table(self) -> str:
        lines = [
            f"\n--- [AttnRes Routing Map for Column (Color {self.tuned_color})] ---",
            f"{'Color':<7} | {'Free Move':<12} | {'Blocked':<12} | {'Residual':<12} | {'Epiplexity':<10}",
            "-" * 65
        ]
        for col_id in sorted(self.routing_logits.keys()):
            attn = self.get_attention_weights(col_id)
            epi = self.compute_epiplexity(col_id)
            lines.append(
                f"{col_id:<7d} | {attn['free']*100:9.2f}%  | {attn['blocked']*100:9.2f}%  | {attn['residual']*100:9.2f}%  | {epi:9.3f}"
            )
        return "\n".join(lines)