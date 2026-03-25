"""
Tests for the ScienceLabEnv — verifies world mechanics, win conditions,
interconnected systems, and the needle-in-a-haystack detail.

Run with:
    ./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/tests/test_science_lab.py -v
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.environments.science_lab import (
    ScienceLabEnv, LOGBOOK_PAGES, NEEDLE_PAGE_INDEX,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tok_names(obs: list) -> list[str]:
    """Extract just the token strings from observation pairs."""
    return [t[0] for t in obs]


def _go(env: ScienceLabEnv, *directions: str) -> None:
    """Navigate through a sequence of rooms."""
    for d in directions:
        env.act(f'go_{d}')


# ── Basic world mechanics ─────────────────────────────────────────────────────

class TestBasicMechanics:

    def test_reset_and_observe(self):
        """Fresh state produces a valid token sequence."""
        env = ScienceLabEnv()
        obs = env.observe()
        assert len(obs) > 0
        # First token has None edge type (sequence start)
        assert obs[0][1] is None
        toks = _tok_names(obs)
        assert 'AT_main_corridor' in toks
        assert any(t.startswith('ENERGY_') for t in toks)
        assert any(t.startswith('HEALTH_') for t in toks)

    def test_movement(self):
        """Can navigate between accessible rooms."""
        env = ScienceLabEnv()
        env.act('go_west')  # main_corridor → supply_closet
        toks = _tok_names(env.observe())
        assert 'AT_supply_closet' in toks

        env.act('go_east')  # back to corridor
        toks = _tok_names(env.observe())
        assert 'AT_main_corridor' in toks

    def test_locked_door_rejected(self):
        """Can't enter locked rooms without keycard."""
        env = ScienceLabEnv()
        env.act('go_north')  # server_room needs keycard_blue
        toks = _tok_names(env.observe())
        assert 'AT_main_corridor' in toks  # didn't move

    def test_keycard_unlocks_door(self):
        """Picking up keycard allows passage through locked door."""
        env = ScienceLabEnv()
        env.act('take_keycard_blue')
        env.act('go_north')  # should work now
        toks = _tok_names(env.observe())
        assert 'AT_server_room' in toks

    def test_take_and_drop(self):
        """Inventory management works."""
        env = ScienceLabEnv()
        env.act('take_keycard_blue')
        toks = _tok_names(env.observe())
        assert 'HOLD_keycard_blue' in toks
        assert not any(t == 'SEE_keycard_blue' for t in toks)

        env.act('drop_keycard_blue')
        toks = _tok_names(env.observe())
        assert 'HOLD_nothing' in toks
        assert 'SEE_keycard_blue' in toks


# ── Documents and the needle ──────────────────────────────────────────────────

class TestDocuments:

    def test_examine_shows_hint(self):
        """Examining an item shows its hint as a TEXT token."""
        env = ScienceLabEnv()
        env.act('go_west')  # supply_closet
        env.act('examine_dusty_logbook')
        toks = _tok_names(env.observe())
        assert any('dusty_logbook_hint_' in t for t in toks)

    def test_read_document_pages(self):
        """Reading a document shows all pages as TEXT tokens."""
        env = ScienceLabEnv()
        env.act('go_west')  # supply_closet
        env.act('read_dusty_logbook')
        toks = _tok_names(env.observe())
        # Should have multiple page tokens
        page_toks = [t for t in toks if t.startswith('TEXT_dusty_logbook_p')]
        assert len(page_toks) == len(LOGBOOK_PAGES)

    def test_needle_in_haystack_page_present(self):
        """The critical margin note is among the logbook pages."""
        env = ScienceLabEnv()
        env.act('go_west')
        env.act('read_dusty_logbook')
        toks = _tok_names(env.observe())
        needle_page = LOGBOOK_PAGES[NEEDLE_PAGE_INDEX]
        found = any(needle_page in t for t in toks)
        assert found, "The needle (sample label swap note) must be in the logbook output"


# ── Chemistry and instruments ─────────────────────────────────────────────────

class TestChemistry:

    def test_spectrometer_swapped_samples(self):
        """Spectrometer reveals B1 has catalyst (labels are swapped)."""
        env = ScienceLabEnv()
        env.act('go_south')  # chem_lab
        env.act('take_sample_b1')
        env.act('use_sample_b1_on_spectrometer')
        toks = _tok_names(env.observe())
        assert any('silver_catalyst_traces_detected' in t for t in toks)

    def test_spectrometer_b2_clean(self):
        """B2 (the real control) has no catalyst."""
        env = ScienceLabEnv()
        env.act('go_south')  # chem_lab
        env.act('take_sample_b2')
        env.act('use_sample_b2_on_spectrometer')
        toks = _tok_names(env.observe())
        assert any('no_catalyst_detected' in t for t in toks)

    def test_combine_safe_in_fume_hood(self):
        """Combining reagents in fume_hood is safe."""
        env = ScienceLabEnv()
        env.act('go_south')  # chem_lab
        env.act('take_reagent_a')
        env.act('take_reagent_b')
        env.act('combine_reagent_a_with_reagent_b')
        toks = _tok_names(env.observe())
        assert any('neutralization_safe' in t for t in toks)

    def test_combine_toxic_outside_fume_hood(self):
        """Combining without fume_hood causes health damage."""
        env = ScienceLabEnv()
        env.act('go_south')  # chem_lab
        env.act('take_reagent_a')
        env.act('take_reagent_b')
        # Move away from fume_hood
        env.act('go_north')  # corridor (no fume hood)
        health_before = env._health
        env.act('combine_reagent_a_with_reagent_b')
        assert env._health < health_before


# ── Contamination system ──────────────────────────────────────────────────────

class TestContamination:

    def test_reactor_lab_contaminates(self):
        """Can't enter reactor_lab without hazmat_suit equipped."""
        env = ScienceLabEnv()
        env.act('go_south')  # chem_lab
        actions = env.available_actions()
        # Can't go south without hazmat_suit equipped
        assert 'go_south' not in actions

    def test_hazmat_reduces_contamination(self):
        """Equipping hazmat_suit allows entry and reduces contamination rate."""
        env = ScienceLabEnv()
        # Hazmat suit is now in supply_closet (accessible)
        env.act('go_west')  # supply_closet
        env.act('take_hazmat_suit')
        env.act('equip_hazmat_suit')
        assert env._equipped == 'hazmat_suit'
        env.act('go_east')   # corridor
        env.act('go_south')  # chem_lab
        env.act('go_south')  # reactor_lab
        assert env._location == 'reactor_lab'
        contam_before = env._contamination
        env.act('examine_reactor_console')  # one step
        assert env._contamination > contam_before
        assert env._contamination <= contam_before + 1.5  # hazmat limits to ~1/step


# ── Power system ──────────────────────────────────────────────────────────────

class TestPowerSystem:

    def test_bus2_starts_off(self):
        """Bus 2 (lobby power) starts OFF."""
        env = ScienceLabEnv()
        assert env._bus_power['bus_2'] is False

    def test_spare_fuse_restores_bus2(self):
        """Installing spare fuse in generator panel restores bus 2."""
        env = ScienceLabEnv()
        # Get to generator room with required items
        env._inventory.extend(['wrench', 'spare_fuse'])
        env._location = 'generator_room'
        env.act('use_wrench_on_generator_panel')
        assert env._panel_open
        env.act('use_spare_fuse_on_generator_panel')
        assert env._bus_power['bus_2'] is True
        assert 'spare_fuse' not in env._inventory

    def test_bus_reroute(self):
        """Wire cutters + tape on panel separates bus 3."""
        env = ScienceLabEnv()
        env._inventory.extend(['wrench', 'wire_cutters', 'electrical_tape'])
        env._location = 'generator_room'
        env.act('use_wrench_on_generator_panel')
        env.act('use_wire_cutters_on_generator_panel')
        assert env._bus_rerouted
        assert 'electrical_tape' not in env._inventory


# ── Win conditions ────────────────────────────────────────────────────────────

class TestWinConditions:

    def test_escape_win_condition(self):
        """Full escape path: hazmat→reactor→maintenance→generator→ventilation→lobby."""
        env = ScienceLabEnv()

        # 1. Get hazmat suit from supply closet
        env.act('go_west')   # supply_closet
        env.act('take_hazmat_suit')
        env.act('take_wrench')
        env.act('equip_hazmat_suit')

        # 2. Navigate to maintenance via reactor_lab
        env.act('go_east')   # corridor
        env.act('go_south')  # chem_lab
        env.act('go_south')  # reactor_lab (hazmat equipped)
        assert env._location == 'reactor_lab'
        env.act('go_south')  # maintenance

        # 3. Get spare fuse + tools from maintenance
        env.act('take_spare_fuse')
        env.act('take_wire_cutters')
        env.act('take_electrical_tape')

        # 4. Go to generator room, install fuse to power bus 2
        env.act('go_east')   # generator_room
        env.act('use_wrench_on_generator_panel')
        assert env._panel_open
        env.act('use_spare_fuse_on_generator_panel')
        assert env._bus_power['bus_2'] is True

        # 5. Reroute bus 3 (so ventilation survives)
        env.act('use_wire_cutters_on_generator_panel')
        assert env._bus_rerouted

        # 6. Navigate to lobby to get keycard_red
        env.act('go_west')   # maintenance
        env.act('go_south')  # ventilation_hub
        env.act('go_south')  # lobby
        assert env._location == 'lobby'
        env.act('take_keycard_red')
        assert 'keycard_red' in env._inventory

        # 7. Go back to director_office for override code
        env.act('go_north')  # ventilation_hub
        env.act('go_north')  # maintenance
        env.act('go_north')  # reactor_lab
        env.act('go_north')  # chem_lab
        env.act('go_north')  # corridor
        env.act('go_east')   # director_office
        assert env._location == 'director_office'

        # 8. Open safe, get override code
        env.act('enter_code_on_wall_safe')
        assert env._safe_opened
        env.act('take_override_code_document')
        assert 'override_code_document' in env._inventory

        # 9. Go back to lobby and enter code
        env.act('go_west')   # corridor
        env.act('go_south')  # chem_lab
        env.act('go_south')  # reactor_lab
        env.act('go_south')  # maintenance
        env.act('go_south')  # ventilation_hub
        env.act('go_south')  # lobby
        assert env._location == 'lobby'
        env.act('enter_code_on_lobby_terminal')
        assert env._override_entered
        assert env.won

    def test_radio_win_condition(self):
        """Alternative win: get items, reach roof, repair antenna, call rescue."""
        env = ScienceLabEnv()

        # 1. Get keycard_blue from corridor
        env.act('take_keycard_blue')

        # 2. Get hazmat suit from supply closet
        env.act('go_west')
        env.act('take_hazmat_suit')
        env.act('equip_hazmat_suit')
        env.act('go_east')   # corridor

        # 3. Get electrical_tape from maintenance
        env.act('go_south')  # chem_lab
        env.act('go_south')  # reactor_lab
        env.act('go_south')  # maintenance
        env.act('take_electrical_tape')

        # 4. Get power_cell from reactor_lab
        env.act('go_north')  # reactor_lab
        env.act('take_power_cell')

        # 5. Navigate to roof
        env.act('go_north')  # chem_lab
        env.act('go_north')  # corridor
        env.act('go_north')  # server_room (keycard_blue unlocks)
        assert env._location == 'server_room'

        # Need override_code for roof_access — set for test focus on radio path
        env._override_entered = True
        env.act('go_up')
        assert env._location == 'roof_access'

        # 6. Repair antenna
        env.act('use_electrical_tape_on_antenna')
        assert env._antenna_repaired

        # 7. Power radio
        env.act('use_power_cell_on_radio_transmitter')
        assert env._radio_powered

        # 8. Call for rescue
        env.act('use_radio_transmitter')
        assert env.won


# ── Energy and survival ───────────────────────────────────────────────────────

class TestSurvival:

    def test_energy_drains_per_step(self):
        """Energy decreases by 1 each step."""
        env = ScienceLabEnv()
        e0 = env._energy
        env.act('examine_notice_board')  # benign action
        assert env._energy == e0 - 1

    def test_starvation_damages_health(self):
        """When energy hits 0, health drains."""
        env = ScienceLabEnv()
        env._energy = 1
        health_before = env._health
        env.act('examine_notice_board')  # energy → 0
        env.act('examine_notice_board')  # now starving
        assert env._health < health_before

    def test_food_restores_energy(self):
        """Eating a protein bar restores energy."""
        env = ScienceLabEnv()
        env._energy = 40
        env._location = 'lobby'
        env.act('take_protein_bar')
        env.act('eat_protein_bar')
        assert env._energy > 40  # protein_bar gives +20 minus 2 steps of drain
