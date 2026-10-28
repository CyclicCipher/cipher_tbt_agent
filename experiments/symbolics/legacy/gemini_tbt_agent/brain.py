"""
brain.py
Symbolic Thousand Brains Agent with Epiplexity Drive & RPE Frustration Circuit
"""

import math
import random
from typing import List, Dict, Tuple, Optional
from arc3_engine import GameAction, GameState, FrameData, make_world
import arc3_worlds
from bus import ColumnInputBus, ColumnOutputBus, ThalamicProbe
from column import SymbolicCorticalColumn

random.seed(42)


class SymbolicThousandBrainsAgent:
    def __init__(self, num_columns: int = 16):
        self.columns: Dict[int, SymbolicCorticalColumn] = {
            c: SymbolicCorticalColumn(column_id=c, tuned_color=c) for c in range(num_columns)
        }
        self.frustration: float = 0.0
        self.recent_actions: List[GameAction] = []

    def get_column(self, color: int) -> SymbolicCorticalColumn:
        return self.columns[color]

    def play_world(
        self,
        world_name: str,
        max_steps: int = 30,
        explore_mode: bool = True,
        verbose: bool = True
    ) -> bool:
        env = make_world(world_name)
        frame = env.reset()
        self.frustration = 0.0
        self.recent_actions.clear()

        print(f"\n{'='*70}")
        print(f"[*] Playing '{world_name}' via Epiplexity Thalamocortical Bus")
        print(f"{'='*70}")

        def get_color_coords(grid: List[List[int]], color: int) -> List[Tuple[int, int]]:
            coords = []
            for y, row in enumerate(grid):
                for x, val in enumerate(row):
                    if val == color:
                        coords.append((x, y))
            return coords

        available_actions = [
            GameAction.ACTION1,  # UP
            GameAction.ACTION2,  # DOWN
            GameAction.ACTION3,  # LEFT
            GameAction.ACTION4,  # RIGHT
        ]

        # -------------------------------------------------------------
        # Phase 0: Kinematic Body Discovery
        # -------------------------------------------------------------
        player_color = None
        for test_act in available_actions:
            grid_before = [row[:] for row in frame.grid]
            frame = env.step(test_act)
            grid_after = frame.grid

            for c in range(16):
                before = get_color_coords(grid_before, c)
                after = get_color_coords(grid_after, c)
                if before and len(before) == len(after):
                    dx = after[0][0] - before[0][0]
                    dy = after[0][1] - before[0][1]
                    if (dx, dy) != (0, 0):
                        if all((x + dx, y + dy) in after for x, y in before):
                            player_color = c
                            break
            if player_color is not None:
                break

        if player_color is None:
            player_color = 1

        print(f"[+] Controlled Agent Body is Color {player_color}.")
        frame = env.reset()

        color_counts = {}
        for row in frame.grid:
            for val in row:
                color_counts[val] = color_counts.get(val, 0) + 1
        
        background_col = max(color_counts.items(), key=lambda item: item[1])[0]
        other_colors = [c for c in color_counts.keys() if c not in (background_col, player_color)]

        tool_color = 4 if 4 in other_colors else None
        target_color = 2 if 2 in other_colors else (3 if 3 in other_colors else None)

        print(f"[+] Scene: Background={background_col}, Tool={tool_color}, Target={target_color}")

        player_col = self.get_column(player_color)
        box_col = self.get_column(tool_color) if tool_color is not None else None

        # -------------------------------------------------------------
        # Phase 1: Short Epistemic Babbling (Mastering Floor)
        # -------------------------------------------------------------
        if explore_mode:
            print("\n>>> PHASE 1: SENSORIMOTOR BABBLING (MASTERING BACKGROUND)")
            for _ in range(15):
                p_pos = get_color_coords(frame.grid, player_color)[0]
                act = random.choice(available_actions)
                motor_prior = player_col.layer1_motor[act]

                bus_query = ThalamicProbe.create_query_bus(
                    entity_color=player_color,
                    grid=frame.grid,
                    current_pos=p_pos,
                    action=act,
                    motor_prior=motor_prior
                )

                frame = env.step(act)
                p_coords_after = get_color_coords(frame.grid, player_color)
                if not p_coords_after:
                    break
                p_pos_after = p_coords_after[0]
                actual_p_delta = (p_pos_after[0] - p_pos[0], p_pos_after[1] - p_pos[1])

                feedback_bus_p = ThalamicProbe.create_feedback_bus(
                    entity_color=player_color,
                    action=act,
                    probe_color=bus_query.contact_probe,
                    actual_delta=actual_p_delta
                )
                player_col.update(feedback_bus_p)

            print(player_col.render_routing_table())
            frame = env.reset()

        # -------------------------------------------------------------
        # Phase 2: Execution via Multi-Column Rollouts + Epiplexity
        # -------------------------------------------------------------
        print("\n>>> PHASE 2: ACTIVE INFERENCE VIA EPIPLEXITY & MULTI-COLUMN ROLLOUTS")
        target_coords = get_color_coords(frame.grid, target_color)
        if not target_coords:
            return False
        target_pos = target_coords[0]

        for step_idx in range(1, max_steps + 1):
            p_pos = get_color_coords(frame.grid, player_color)[0]
            b_coords = get_color_coords(frame.grid, tool_color) if tool_color else None
            b_pos = b_coords[0] if b_coords else None

            action_evals = []
            best_act = None
            best_cost = float("inf")

            for cand_act in available_actions:
                motor_prior = player_col.layer1_motor[cand_act]
                
                # Query Player Column via Bus
                p_in_bus = ThalamicProbe.create_query_bus(
                    entity_color=player_color,
                    grid=frame.grid,
                    current_pos=p_pos,
                    action=cand_act,
                    motor_prior=motor_prior
                )
                p_out_bus = player_col.forward(p_in_bus)

                # Lateral Impulse to Tool (Box)
                box_pred_delta = (0, 0)
                if p_out_bus.transferred_impulse is not None and b_pos is not None:
                    impulse_vec, target_c = p_out_bus.transferred_impulse
                    if target_c == tool_color and box_col is not None:
                        b_in_bus = ThalamicProbe.create_query_bus(
                            entity_color=tool_color,
                            grid=frame.grid,
                            current_pos=b_pos,
                            impulse=impulse_vec
                        )
                        b_out_bus = box_col.forward(b_in_bus)
                        box_pred_delta = b_out_bus.predicted_delta

                sim_p = (p_pos[0] + p_out_bus.predicted_delta[0], p_pos[1] + p_out_bus.predicted_delta[1])
                sim_b = (b_pos[0] + box_pred_delta[0], b_pos[1] + box_pred_delta[1]) if b_pos else None

                # 1. Instrumental Cost
                if b_pos is not None and sim_b is not None:
                    dist_box_to_target = abs(sim_b[0] - target_pos[0]) + abs(sim_b[1] - target_pos[1])
                    dx_t = 1 if target_pos[0] > sim_b[0] else (-1 if target_pos[0] < sim_b[0] else 0)
                    dy_t = 1 if target_pos[1] > sim_b[1] else (-1 if target_pos[1] < sim_b[1] else 0)
                    push_anchor = (sim_b[0] - dx_t, sim_b[1] - dy_t)
                    dist_player_to_anchor = abs(sim_p[0] - push_anchor[0]) + abs(sim_p[1] - push_anchor[1])
                    instrumental_cost = (10.0 * dist_box_to_target) + dist_player_to_anchor
                else:
                    instrumental_cost = abs(sim_p[0] - target_pos[0]) + abs(sim_p[1] - target_pos[1])

                # 2. Epiplexity Drive (Curiosity bonus for learnable structure)
                epiplexity_bonus = 3.5 * p_out_bus.epiplexity

                # 3. Frustration Penalty (Tonic suppression of repeated non-productive actions)
                frustration_pen = 2.0 * self.frustration if (self.recent_actions and cand_act == self.recent_actions[-1]) else 0.0

                total_cost = instrumental_cost - epiplexity_bonus + frustration_pen

                telemetry = (
                    f"{cand_act.name:7s} | Probe: {p_in_bus.contact_probe:2d} | "
                    f"Pred Δ: {p_out_bus.predicted_delta} | Box Δ: {box_pred_delta} | "
                    f"Epiplexity: {p_out_bus.epiplexity:.3f} | Cost: {total_cost:5.2f}"
                )
                action_evals.append((total_cost, cand_act, telemetry))

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_act = cand_act

            if verbose:
                print(f"\n[Step {step_idx:02d}] At Player={p_pos}, Box={b_pos} -> Candidate Evaluations:")
                for cost, act, telem in action_evals:
                    chosen = "===>" if act == best_act else "    "
                    print(f"  {chosen} {telem}")

            # Execute Best Action
            b_pos_before = b_pos
            frame = env.step(best_act)
            self.recent_actions.append(best_act)
            if len(self.recent_actions) > 4:
                self.recent_actions.pop(0)

            p_cur = get_color_coords(frame.grid, player_color)[0]
            b_cur = get_color_coords(frame.grid, tool_color)[0] if tool_color else None

            # Calculate State Change & RPE
            box_moved = (b_pos_before is not None and b_cur is not None and b_cur != b_pos_before)
            
            if box_moved or frame.state == GameState.WIN:
                # Phasic Dopamine Burst: reset frustration
                self.frustration = 0.0
            else:
                # Tonic Frustration accumulation when pacing without state change
                self.frustration += 0.5

            # Thalamic Feedback Updates
            feedback_bus = ThalamicProbe.create_feedback_bus(
                entity_color=player_color,
                action=best_act,
                probe_color=ThalamicProbe.extract_probe_color(frame.grid, p_pos, player_col.layer1_motor[best_act]),
                actual_delta=(p_cur[0] - p_pos[0], p_cur[1] - p_pos[1])
            )
            player_col.update(feedback_bus)

            # If Box was moved, update Box Column as well
            if box_moved and box_col is not None:
                player_col.routing_logits[tool_color]["residual"] += 4.0
                b_delta = (b_cur[0] - b_pos_before[0], b_cur[1] - b_pos_before[1])
                b_feed = ColumnInputBus(
                    entity_color=tool_color,
                    impulse_vector=player_col.layer1_motor[best_act],
                    contact_probe=0,
                    actual_delta=b_delta
                )
                box_col.update(b_feed)

            if frame.state == GameState.WIN:
                print(f"\n[+] VICTORY REACHED AT STEP {frame.step_count}!")
                print(player_col.render_routing_table())
                return True

        return False


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SYMBOLIC THOUSAND BRAINS AGENT ON 'push_block'")
    print("=" * 70)

    agent = SymbolicThousandBrainsAgent()
    agent.play_world("push_block", max_steps=30, explore_mode=True, verbose=True)