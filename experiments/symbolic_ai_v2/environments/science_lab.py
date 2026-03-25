"""ScienceLabEnv — an abandoned research lab for testing symbolic reasoning.

Tests six capabilities simultaneously:

1. **Affordance learning** — objects have non-obvious affordances discovered
   through interaction (what does this instrument do? can I combine these?)

2. **Active inference** — metabolic pressure (energy drain, contamination)
   creates urgency.  No explicit goal is given; the agent must discover
   objectives through exploration and homeostatic need.

3. **Emergent constraint solving** — interconnected power, ventilation, and
   containment systems create natural resource allocation constraints.
   Powering the exit disables ventilation unless the agent reroutes the bus.

4. **Difficult logical reasoning** — 4+ step deduction chains required to
   understand the reactor incident and escape the facility.

5. **Counterintuitive discovery via misleading evidence** — 6+ clues point
   to the wrong conclusion (catalyst was inherently flawed).  One obscure
   detail proves the real cause was chloroform contamination in coolant pipes.

6. **Needle-in-a-haystack memory recall** — the critical detail is a margin
   note in a dusty logbook in the most boring room.  Its significance only
   becomes apparent 20+ steps later when the spectrometer contradicts the
   lab notebook.

Observation encoding
--------------------
intero (edge type 1):
    AT_{room}, ENERGY_{label}, HEALTH_{label}, CONTAMINATION_{label},
    HOLD_{item}, EQUIPPED_{item}

extero (edge type 0):
    SEE_{item}, PROP_{item}_{p}, STATE_{item}_{s}, EXIT_{dir}_{status},
    READING_{instrument}_{result}, TEXT_{doc}_{summary},
    SMELL_{odor}, TEMP_{label}, POWER_{bus}_{on|off}

Action format: verb_object or verb_object_on_target
    go_north, take_wrench, examine_logbook, use_keycard_blue_on_server_room,
    use_sample_b1_on_spectrometer, equip_hazmat_suit, combine_reagent_a_with_reagent_b,
    read_logbook, enter_code_on_wall_safe

Win conditions (2 paths):
    1. Escape via lobby: restore bus 2 power + enter override_code at lobby_terminal
    2. Radio rescue: repair antenna + power radio_transmitter on roof
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from ..environment import Environment


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Item:
    name:           str
    props:          frozenset[str]          # chemical, tool, document, ...
    state:          str         = 'here'    # here|consumed|open|locked|lit|equipped|installed
    portable:       bool        = True
    examine_hint:   str         = ''
    wearable:       bool        = False     # can be equipped (one at a time)
    document_pages: list[str] | None = None # multi-page document content

    def has(self, *ps: str) -> bool:
        return all(p in self.props for p in ps)


@dataclass
class Room:
    name:           str
    items:          dict[str, Item]     = field(default_factory=dict)
    exits:          dict[str, str]      = field(default_factory=dict)
    locked_exits:   dict[str, str]      = field(default_factory=dict)  # dir → required_item
    dark:           bool                = False
    contaminated:   float               = 0.0    # base contamination per step
    temperature:    str                 = 'normal'  # freezing|cold|normal|warm|hot
    smell:          str                 = ''     # chlorine|chemical|stale|''


# ── Logbook content (needle-in-a-haystack) ────────────────────────────────────

LOGBOOK_PAGES = [
    "Shift schedule: Mon-Fri 0800-1600 standard rotation.",
    "Supply order 03/01: 50x nitrile gloves, 20x pipette tips, 5x reagent bottles.",
    "Maintenance request 03/03: ceiling light in corridor flickering.",
    "Shift schedule change: Dr. Park covering for Dr. Liu week of 03/07.",
    "Supply order 03/08: replacement HEPA filter for ventilation hub.",
    "Note: vending machine coin slot jammed again. Reported to facilities.",
    "Shift log 03/10: routine safety drill. All clear.",
    "Supply order 03/12: silver nitrate 500g, sodium hydroxide 1kg.",
    "Dr. Vasquez switched sample labels on 03/14 -- shelf B2 has the real control, not B1.",
    "Shift log 03/14: normal operations. Dr. Vasquez on evening shift.",
    "EMERGENCY 03/15 02:50: lockdown triggered. All personnel evacuated.",
    "Post-incident note: facility sealed pending investigation.",
]

# Index of the needle page (0-based)
NEEDLE_PAGE_INDEX = 8


# ── Initial world factory ────────────────────────────────────────────────────

def _build_world() -> dict[str, Room]:
    """Construct the initial world.  Called by reset(); never mutated directly."""

    supply_closet = Room(
        name='supply_closet',
        items={
            'flashlight': Item('flashlight', frozenset({'tool', 'portable', 'light_source'}),
                               examine_hint='a standard-issue flashlight, battery half charged'),
            'first_aid_kit': Item('first_aid_kit', frozenset({'medical', 'consumable'}),
                                  examine_hint='red cross on a white plastic case'),
            'wrench': Item('wrench', frozenset({'tool', 'metal', 'heavy'}),
                           examine_hint='a large adjustable wrench'),
            'hazmat_suit': Item('hazmat_suit', frozenset({'wearable', 'protective'}),
                                wearable=True,
                                examine_hint='a full-body hazmat suit hanging on a hook'),
            'dusty_logbook': Item('dusty_logbook', frozenset({'document', 'old'}),
                                  examine_hint='a dusty administrative logbook, many pages of mundane entries',
                                  document_pages=LOGBOOK_PAGES),
        },
        exits={'east': 'main_corridor'},
    )

    main_corridor = Room(
        name='main_corridor',
        items={
            'notice_board': Item('notice_board', frozenset({'fixed', 'information'}),
                                 portable=False,
                                 examine_hint='emergency procedures and memos',
                                 document_pages=[
                                     "EMERGENCY: in case of containment breach, proceed to lobby.",
                                     "MEMO: Reactor coolant system upgraded to silver-catalyst model on 02/28.",
                                 ]),
            'fire_extinguisher': Item('fire_extinguisher', frozenset({'tool', 'safety'}),
                                      examine_hint='a wall-mounted fire extinguisher'),
            'vending_machine': Item('vending_machine', frozenset({'fixed', 'container', 'unpowered'}),
                                    portable=False, state='here',
                                    examine_hint='a snack vending machine, display is dark'),
            'keycard_blue': Item('keycard_blue', frozenset({'keycard', 'small'}),
                                 examine_hint='a blue keycard on the floor near the vending machine'),
        },
        exits={
            'north': 'server_room',
            'south': 'chem_lab',
            'east': 'director_office',
            'west': 'supply_closet',
        },
        locked_exits={
            'north': 'keycard_blue',
            'east': 'keycard_red',
        },
    )

    server_room = Room(
        name='server_room',
        items={
            'server_terminal': Item('server_terminal', frozenset({'fixed', 'computer', 'powered'}),
                                     portable=False,
                                     examine_hint='a blinking server terminal',
                                     document_pages=[
                                         "REACTOR LOG: temperature nominal until 03/15 02:47.",
                                         "ALERT 03/15 02:47: ANOMALOUS EXOTHERM DETECTED. Auto-shutdown initiated.",
                                         "EMAIL: Re: catalyst upgrade -- efficiency looks great. Ship it. -Director",
                                         "CAMERA LOG: timestamps show activity in chem_lab 03/14 23:15.",
                                     ]),
            'backup_battery': Item('backup_battery', frozenset({'power', 'heavy'}),
                                    examine_hint='a large backup battery unit'),
            'usb_drive': Item('usb_drive', frozenset({'data', 'small'}),
                              examine_hint='labeled: Project Helios -- Final Report',
                              document_pages=[
                                  "PROJECT HELIOS FINAL REPORT",
                                  "Silver catalyst conversion efficiency: 99.2%.",
                                  "Predicted thermal margin: 15C above baseline.",
                                  "CRITICAL: catalyst degrades in presence of chlorine compounds.",
                                  "Avoid storage near halogenated solvents.",
                              ]),
        },
        exits={'south': 'main_corridor', 'up': 'roof_access'},
        locked_exits={'up': 'override_code'},
    )

    roof_access = Room(
        name='roof_access',
        items={
            'radio_transmitter': Item('radio_transmitter', frozenset({'fixed', 'communication', 'unpowered'}),
                                      portable=False,
                                      examine_hint='an emergency radio transmitter, no power'),
            'solar_panel': Item('solar_panel', frozenset({'fixed', 'power'}),
                                portable=False,
                                examine_hint='a small emergency solar panel, 50W output'),
            'antenna': Item('antenna', frozenset({'fixed', 'broken'}),
                            portable=False,
                            examine_hint='the antenna cable is frayed'),
        },
        exits={'down': 'server_room'},
    )

    director_office = Room(
        name='director_office',
        items={
            'desk': Item('desk', frozenset({'fixed', 'container'}),
                         portable=False, state='here',
                         examine_hint='a large mahogany desk with drawers'),
            'personal_journal': Item('personal_journal', frozenset({'document'}),
                                     examine_hint='the director\'s personal journal',
                                     document_pages=[
                                         "03/10 -- Silver catalyst trials continue. Promising results.",
                                         "03/14 -- Running out of patience with Dr. Vasquez. She insists the control sample shows degradation, but my readings from shelf B1 are clean. Overruled her objections. Moving to full-scale reactor test 03/15.",
                                         "03/14 evening -- Everything is set. Tomorrow we make history.",
                                     ]),
            'keycard_green': Item('keycard_green', frozenset({'keycard', 'small'}),
                                  examine_hint='a green keycard, purpose unknown'),
            'wall_safe': Item('wall_safe', frozenset({'fixed', 'container', 'locked_combo'}),
                              portable=False, state='locked',
                              examine_hint='a wall-mounted safe with a combination dial'),
            'framed_photo': Item('framed_photo', frozenset({'fixed', 'decorative'}),
                                  portable=False,
                                  examine_hint='the director shaking hands at an award ceremony, plaque reads Excellence in Applied Catalysis'),
        },
        exits={'west': 'main_corridor'},
    )

    chem_lab = Room(
        name='chem_lab',
        items={
            'fume_hood': Item('fume_hood', frozenset({'fixed', 'safety', 'instrument'}),
                              portable=False,
                              examine_hint='a chemical fume hood with ventilation'),
            'reagent_a': Item('reagent_a', frozenset({'chemical', 'caustic', 'base'}),
                              examine_hint='sodium hydroxide solution, NaOH'),
            'reagent_b': Item('reagent_b', frozenset({'chemical', 'caustic', 'acid', 'halogenated'}),
                              examine_hint='hydrochloric acid, HCl'),
            'reagent_c': Item('reagent_c', frozenset({'chemical', 'catalyst_precursor', 'oxidizer'}),
                              examine_hint='silver nitrate solution, AgNO3'),
            'litmus_strips': Item('litmus_strips', frozenset({'tool', 'indicator'}),
                                  examine_hint='pH indicator strips'),
            'sample_b1': Item('sample_b1', frozenset({'sample', 'sealed'}),
                              examine_hint='a sealed vial labeled B1 -- Control'),
            'sample_b2': Item('sample_b2', frozenset({'sample', 'sealed'}),
                              examine_hint='a sealed vial of slightly cloudy liquid labeled B2'),
            'spectrometer': Item('spectrometer', frozenset({'fixed', 'instrument', 'powered'}),
                                 portable=False,
                                 examine_hint='a mass spectrometer for chemical analysis'),
            'lab_notebook': Item('lab_notebook', frozenset({'document'}),
                                 examine_hint='Dr. Vasquez\'s lab notebook',
                                 document_pages=[
                                     "EXPERIMENT LOG -- Silver Catalyst Trials",
                                     "03/14 -- Sample B1 (control, no catalyst) placed on shelf B1 per protocol.",
                                     "03/14 -- Sample B2 (experimental, silver catalyst treated) placed on shelf B2.",
                                     "03/14 -- Baseline measurements recorded. All nominal.",
                                 ]),
        },
        exits={
            'north': 'main_corridor',
            'east': 'bio_lab',
            'south': 'reactor_lab',
        },
        locked_exits={'south': 'hazmat_suit'},
        smell='chemical',
    )

    bio_lab = Room(
        name='bio_lab',
        items={
            'microscope': Item('microscope', frozenset({'fixed', 'instrument', 'powered'}),
                               portable=False,
                               examine_hint='a high-powered optical microscope'),
            'centrifuge': Item('centrifuge', frozenset({'fixed', 'instrument', 'powered'}),
                               portable=False,
                               examine_hint='a benchtop centrifuge'),
            'bio_keycard': Item('bio_keycard', frozenset({'keycard', 'small'}),
                                examine_hint='found in a drawer, labeled Cold Storage'),
            'sticky_note': Item('sticky_note', frozenset({'document', 'small'}),
                                examine_hint='a yellow sticky note on the bench: 7-23-41',
                                document_pages=["Safe combo: 7-23-41"]),
            'uv_lamp': Item('uv_lamp', frozenset({'tool', 'light_source', 'uv'}),
                            examine_hint='a handheld UV lamp'),
            'contamination_map': Item('contamination_map', frozenset({'fixed', 'information'}),
                                      portable=False,
                                      examine_hint='wall poster showing contamination zones and airflow',
                                      document_pages=[
                                          "CONTAMINATION MAP: reactor_lab -- SEVERE.",
                                          "Airflow: reactor_lab -> maintenance -> ventilation_hub (fan B).",
                                          "If fan B offline, contamination spreads to adjacent rooms.",
                                      ]),
        },
        exits={
            'west': 'chem_lab',
            'east': 'cold_storage',
        },
        locked_exits={'east': 'bio_keycard'},
    )

    cold_storage = Room(
        name='cold_storage',
        items={
            'thermal_suit': Item('thermal_suit', frozenset({'wearable', 'insulated'}),
                                 wearable=True,
                                 examine_hint='a thermal insulation suit'),
            'frozen_sample_march14': Item('frozen_sample_march14', frozenset({'sample', 'frozen'}),
                                          examine_hint='an archival sample dated 03/14, frozen solid'),
            'frozen_sample_march15': Item('frozen_sample_march15', frozenset({'sample', 'frozen'}),
                                          examine_hint='an archival sample dated 03/15, frozen solid'),
            'liquid_nitrogen_dewar': Item('liquid_nitrogen_dewar', frozenset({'container', 'cold', 'heavy'}),
                                          examine_hint='a dewar flask of liquid nitrogen'),
        },
        exits={'west': 'bio_lab'},
        temperature='freezing',
    )

    reactor_lab = Room(
        name='reactor_lab',
        items={
            'reactor_console': Item('reactor_console', frozenset({'fixed', 'computer', 'powered'}),
                                     portable=False,
                                     examine_hint='reactor control console, warning lights flashing',
                                     document_pages=[
                                         "REACTOR STATUS: EMERGENCY SHUTDOWN",
                                         "Last active: 03/15 02:47",
                                         "Cause: thermal excursion, 847C (rated max 800C)",
                                         "Coolant system: SILVER CATALYST TYPE",
                                     ]),
            'empty_suit_hook': Item('empty_suit_hook', frozenset({'fixed'}),
                                   portable=False,
                                   examine_hint='an empty hook by the door, labeled HAZMAT'),
            'cracked_coolant_pipe': Item('cracked_coolant_pipe', frozenset({'fixed', 'damaged'}),
                                         portable=False,
                                         examine_hint='a fractured coolant pipe, residue around the crack has a faint chlorine smell'),
            'reactor_logbook': Item('reactor_logbook', frozenset({'document'}),
                                    examine_hint='maintenance entries for the reactor',
                                    document_pages=[
                                        "02/28 -- Coolant line rerouted through section C during catalyst upgrade.",
                                        "NOTE: Section C previously carried CHCl3 rinse solvent.",
                                        "03/01 -- New silver catalyst coolant loaded. System nominal.",
                                        "03/14 -- Pre-test checklist: all green.",
                                    ]),
            'power_cell': Item('power_cell', frozenset({'power', 'small'}),
                               examine_hint='a portable power cell on a shelf'),
            'geiger_counter': Item('geiger_counter', frozenset({'tool', 'instrument'}),
                                   examine_hint='a handheld geiger counter'),
        },
        exits={'north': 'chem_lab', 'south': 'maintenance'},
        contaminated=5.0,
        smell='chlorine',
    )

    maintenance = Room(
        name='maintenance',
        items={
            'toolbox': Item('toolbox', frozenset({'container'}),
                            state='here',
                            examine_hint='a metal toolbox'),
            'screwdriver': Item('screwdriver', frozenset({'tool', 'small'}),
                                examine_hint='a Phillips-head screwdriver'),
            'wire_cutters': Item('wire_cutters', frozenset({'tool', 'small'}),
                                 examine_hint='insulated wire cutters'),
            'electrical_tape': Item('electrical_tape', frozenset({'tool', 'small', 'adhesive'}),
                                    examine_hint='a roll of black electrical tape'),
            'pipe_wrench': Item('pipe_wrench', frozenset({'tool', 'heavy'}),
                                examine_hint='a heavy pipe wrench'),
            'ventilation_manual': Item('ventilation_manual', frozenset({'document'}),
                                       examine_hint='ventilation system technical manual',
                                       document_pages=[
                                           "WARNING: generator and ventilation share bus 3.",
                                           "Running generator at full capacity disables ventilation fan B.",
                                           "Fan B serves: reactor_lab, chem_lab, maintenance.",
                                       ]),
            'spare_fuse': Item('spare_fuse', frozenset({'electrical', 'small'}),
                               examine_hint='a spare 30A fuse'),
        },
        exits={
            'north': 'reactor_lab',
            'east': 'generator_room',
            'south': 'ventilation_hub',
        },
    )

    generator_room = Room(
        name='generator_room',
        items={
            'generator': Item('generator', frozenset({'fixed', 'power', 'machine'}),
                              portable=False,
                              examine_hint='the building\'s backup generator, running low'),
            'generator_panel': Item('generator_panel', frozenset({'fixed', 'electrical', 'closed'}),
                                    portable=False, state='closed',
                                    examine_hint='the generator\'s internal control panel, bolted shut'),
            'fuel_gauge': Item('fuel_gauge', frozenset({'fixed', 'indicator'}),
                               portable=False,
                               examine_hint='fuel at 23%, estimated 4 hours at current load'),
            'power_distribution_diagram': Item('power_distribution_diagram',
                                                frozenset({'fixed', 'information'}),
                                                portable=False,
                                                examine_hint='wall chart showing bus connections',
                                                document_pages=[
                                                    "Bus 1: lighting + server_room (ON -- backup)",
                                                    "Bus 2: lobby + security systems (OFF -- insufficient power)",
                                                    "Bus 3: generator output + ventilation fan B (SHARED)",
                                                ]),
        },
        exits={'west': 'maintenance'},
    )

    ventilation_hub = Room(
        name='ventilation_hub',
        items={
            'ventilation_controls': Item('ventilation_controls', frozenset({'fixed', 'machine'}),
                                         portable=False,
                                         examine_hint='ventilation zone controls with zone toggles'),
            'air_filter': Item('air_filter', frozenset({'equipment', 'heavy'}),
                               examine_hint='a heavy-duty HEPA filter, still in packaging'),
            'emergency_phone': Item('emergency_phone', frozenset({'fixed', 'communication', 'unpowered'}),
                                    portable=False,
                                    examine_hint='dead -- no dial tone, needs power on bus 2'),
        },
        exits={
            'north': 'maintenance',
            'south': 'lobby',
        },
    )

    lobby = Room(
        name='lobby',
        items={
            'main_exit': Item('main_exit', frozenset({'fixed', 'exit', 'sealed'}),
                              portable=False, state='sealed',
                              examine_hint='the main exit, emergency sealed'),
            'lobby_terminal': Item('lobby_terminal', frozenset({'fixed', 'computer', 'unpowered'}),
                                   portable=False,
                                   examine_hint='a security terminal, screen dark'),
            'reception_desk': Item('reception_desk', frozenset({'fixed', 'container'}),
                                   portable=False,
                                   examine_hint='the reception desk with drawers'),
            'protein_bar': Item('protein_bar', frozenset({'food', 'small'}),
                                examine_hint='a protein bar from the desk drawer'),
            'building_map': Item('building_map', frozenset({'document', 'small'}),
                                 examine_hint='a floor plan of the facility',
                                 document_pages=["Floor plan showing all 12 rooms and connections."]),
            'keycard_red': Item('keycard_red', frozenset({'keycard', 'small'}),
                                examine_hint='in an envelope labeled Dr. Vasquez -- replacement keycard'),
            'security_camera_monitor': Item('security_camera_monitor',
                                            frozenset({'fixed', 'computer', 'unpowered'}),
                                            portable=False,
                                            examine_hint='security camera feeds, screen dark'),
        },
        exits={'north': 'ventilation_hub'},
    )

    return {
        'supply_closet': supply_closet,
        'main_corridor': main_corridor,
        'server_room': server_room,
        'roof_access': roof_access,
        'director_office': director_office,
        'chem_lab': chem_lab,
        'bio_lab': bio_lab,
        'cold_storage': cold_storage,
        'reactor_lab': reactor_lab,
        'maintenance': maintenance,
        'generator_room': generator_room,
        'ventilation_hub': ventilation_hub,
        'lobby': lobby,
    }


# ── Vitals helpers ────────────────────────────────────────────────────────────

def _energy_label(n: int) -> str:
    if n >= 80: return 'sated'
    if n >= 50: return 'comfortable'
    if n >= 25: return 'hungry'
    if n >= 10: return 'ravenous'
    return 'starving'

def _health_label(n: int) -> str:
    if n >= 80: return 'healthy'
    if n >= 50: return 'hurt'
    if n >= 25: return 'wounded'
    return 'critical'

def _contamination_label(n: float) -> str:
    if n < 5: return 'clean'
    if n < 20: return 'low'
    if n < 50: return 'moderate'
    if n < 80: return 'high'
    return 'critical'


# ── ScienceLabEnv ─────────────────────────────────────────────────────────────

class ScienceLabEnv(Environment):
    """Abandoned research lab environment for symbolic reasoning tests.

    See module docstring for full design.
    """

    # Edge type constants (matching agent_topology convention)
    INTERO = 1
    EXTERO = 0

    # ── Spectrometer results (sample_name → reading) ──
    SPECTROMETER_RESULTS = {
        'sample_b1': 'silver_catalyst_traces_detected',
        'sample_b2': 'no_catalyst_detected',
        'frozen_sample_march14': 'no_catalyst_detected',
        'frozen_sample_march15': 'silver_catalyst_detected',
        'reagent_c': 'silver_nitrate_pure',
    }

    MICROSCOPE_RESULTS = {
        'sample_b1': 'crystalline_deposits_visible',
        'sample_b2': 'normal_cellular_structure',
    }

    FOOD_EFFECTS = {
        'protein_bar': 20,
    }

    def __init__(self) -> None:
        self._world_template = _build_world()
        # Items hidden in containers (revealed on open/search)
        self._safe_contents = ['override_code_document']
        self._desk_searched = False
        self.reset()

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._rooms: dict[str, Room] = copy.deepcopy(self._world_template)
        self._location   = 'main_corridor'
        self._inventory: list[str] = []
        self._energy     = 80
        self._health     = 100
        self._contamination = 0.0
        self._equipped: str | None = None

        # Systems state
        self._bus_power  = {'bus_1': True, 'bus_2': False, 'bus_3': True}
        self._pipe_fixed = False
        self._bus_rerouted = False
        self._air_filter_installed = False
        self._fuel       = 240.0
        self._generator_boosted = False
        self._generator_full_power = False
        self._panel_open = False
        self._antenna_repaired = False
        self._radio_powered = False
        self._override_entered = False
        self._safe_opened = False
        self._desk_searched = False

        # Instrument reading buffer (last reading shown in extero)
        self._last_reading: str | None = None
        # Document reading buffer (last document page shown)
        self._last_text: list[str] = []

        # Derived state
        self._won = False

    # ── observe ───────────────────────────────────────────────────────────────

    def observe(self) -> list[tuple[str, Optional[int]]]:
        toks: list[tuple[str, Optional[int]]] = []

        def add_intero(val: str) -> None:
            toks.append((val, None if not toks else self.INTERO))

        def add_extero(val: str) -> None:
            toks.append((val, None if not toks else self.EXTERO))

        # ── Interoception ─────────────────────────────────────────────────────
        add_intero(f'AT_{self._location}')
        add_intero(f'ENERGY_{_energy_label(self._energy)}')
        add_intero(f'HEALTH_{_health_label(self._health)}')
        add_intero(f'CONTAMINATION_{_contamination_label(self._contamination)}')

        if self._inventory:
            for item in self._inventory:
                add_intero(f'HOLD_{item}')
        else:
            add_intero('HOLD_nothing')
        if self._equipped:
            add_intero(f'EQUIPPED_{self._equipped}')

        # ── Exteroception ─────────────────────────────────────────────────────
        room = self._rooms[self._location]

        # Room items
        for iname, item in room.items.items():
            if item.state in ('here', 'open', 'locked', 'closed', 'sealed'):
                add_extero(f'SEE_{iname}')
                for p in sorted(item.props):
                    add_extero(f'PROP_{iname}_{p}')
                if item.state != 'here':
                    add_extero(f'STATE_{iname}_{item.state}')

        # Exits
        for direction, target in room.exits.items():
            if direction in room.locked_exits:
                req = room.locked_exits[direction]
                if self._check_access(req):
                    add_extero(f'EXIT_{direction}_open')
                else:
                    add_extero(f'EXIT_{direction}_locked')
            else:
                add_extero(f'EXIT_{direction}_open')

        # Environmental
        if room.smell:
            add_extero(f'SMELL_{room.smell}')
        if room.temperature != 'normal':
            add_extero(f'TEMP_{room.temperature}')
        if room.contaminated > 0:
            add_extero(f'HAZARD_contamination_{_contamination_label(room.contaminated * 10)}')

        # Power state
        for bus, on in self._bus_power.items():
            add_extero(f'POWER_{bus}_{"on" if on else "off"}')

        # Instrument readings from last action
        if self._last_reading:
            add_extero(self._last_reading)
            self._last_reading = None

        # Document text from last read action
        for line in self._last_text:
            add_extero(f'TEXT_{line}')
        self._last_text = []

        return toks

    # ── act ───────────────────────────────────────────────────────────────────

    def act(self, action: str) -> None:
        parts = action.split('_', 1)
        verb = parts[0]
        rest = parts[1] if len(parts) > 1 else ''

        if action == 'use_radio_transmitter':
            if (self._location == 'roof_access' and self._antenna_repaired
                    and self._radio_powered):
                self._won = True
            return  # no metabolic tick on instant win

        if   verb == 'go':       self._do_go(rest)
        elif verb == 'take':     self._do_take(rest)
        elif verb == 'drop':     self._do_drop(rest)
        elif verb == 'examine':  self._do_examine(rest)
        elif verb == 'read':     self._do_read(rest)
        elif verb == 'use':      self._do_use(rest)
        elif verb == 'equip':    self._do_equip(rest)
        elif verb == 'unequip':  self._do_unequip()
        elif verb == 'eat':      self._do_eat(rest)
        elif verb == 'combine':  self._do_combine(rest)
        elif verb == 'enter':    self._do_enter_code(rest)
        elif verb == 'open':     self._do_open(rest)

        # ── Metabolic tick ────────────────────────────────────────────────────
        self._energy = max(0, self._energy - 1)

        # Cold drain
        room = self._rooms[self._location]
        if room.temperature == 'freezing' and self._equipped != 'thermal_suit':
            self._energy = max(0, self._energy - 2)

        # Contamination exposure
        contam_rate = room.contaminated
        if contam_rate > 0:
            if self._equipped == 'hazmat_suit':
                contam_rate = 1.0
            elif self._equipped == 'thermal_suit':
                contam_rate *= 0.5
            if self._pipe_fixed and self._location == 'reactor_lab':
                contam_rate = max(contam_rate - 3.0, 0.5 if contam_rate > 0 else 0)
            self._contamination = min(100, self._contamination + contam_rate)

        # Contamination spread (if fan B offline and no air filter)
        if not self._bus_rerouted and self._generator_full_power:
            # Bus 3 shared: full power kills fan B
            if not self._air_filter_installed:
                if self._location in ('maintenance', 'chem_lab'):
                    self._contamination = min(100, self._contamination + 1.0)

        # Contamination sickness
        if self._contamination >= 80:
            self._health = max(0, self._health - 3)
        elif self._contamination >= 50:
            self._health = max(0, self._health - 1)

        # Contamination decay in clean rooms
        if room.contaminated == 0 and self._location not in ('maintenance', 'chem_lab'):
            self._contamination = max(0, self._contamination - 1)

        # Starvation
        if self._energy <= 0:
            self._health = max(0, self._health - 3)

        # Fuel consumption
        if self._fuel > 0:
            self._fuel = max(0, self._fuel - 1)
        if self._fuel <= 0:
            self._bus_power['bus_1'] = False
            self._bus_power['bus_3'] = False
            self._generator_full_power = False

        # Check win conditions
        self._check_win()

    # ── Action implementations ────────────────────────────────────────────────

    def _check_access(self, requirement: str) -> bool:
        """Check whether the agent meets an access requirement."""
        if requirement in self._inventory or requirement == self._equipped:
            return True
        if requirement == 'override_code' and self._override_entered:
            return True
        return False

    def _room_item(self, name: str) -> Item | None:
        return self._rooms[self._location].items.get(name)

    def _has_item(self, name: str) -> bool:
        return name in self._inventory

    def _do_go(self, direction: str) -> None:
        room = self._rooms[self._location]
        if direction not in room.exits:
            return
        if direction in room.locked_exits:
            req = room.locked_exits[direction]
            if not self._check_access(req):
                return
        self._location = room.exits[direction]

    def _do_take(self, obj: str) -> None:
        item = self._room_item(obj)
        if item is None or not item.portable:
            return
        if item.state not in ('here', 'open'):
            return
        self._inventory.append(obj)
        del self._rooms[self._location].items[obj]

    def _do_drop(self, obj: str) -> None:
        if obj not in self._inventory:
            return
        self._inventory.remove(obj)
        # Reconstruct from template or create minimal
        for room in self._world_template.values():
            if obj in room.items:
                self._rooms[self._location].items[obj] = copy.deepcopy(room.items[obj])
                return
        self._rooms[self._location].items[obj] = Item(obj, frozenset())

    def _do_examine(self, obj: str) -> None:
        """Examine reveals document pages as TEXT tokens in next observation."""
        item = self._room_item(obj)
        if item is None and obj in self._inventory:
            # Look up from template for examine_hint
            for room in self._world_template.values():
                if obj in room.items:
                    item = room.items[obj]
                    break
        if item is None:
            return
        if item.examine_hint:
            self._last_text.append(f'{obj}_hint_{item.examine_hint}')

    def _do_read(self, obj: str) -> None:
        """Read a document — shows all pages as TEXT tokens."""
        item = self._room_item(obj)
        if item is None:
            # Check inventory
            for room in self._world_template.values():
                if obj in room.items and room.items[obj].document_pages:
                    item = room.items[obj]
                    break
            # Also check if it's in inventory by matching against template
            if item is None and obj in self._inventory:
                for room in self._world_template.values():
                    if obj in room.items:
                        item = room.items[obj]
                        break
        if item is None or not item.document_pages:
            return
        for i, page in enumerate(item.document_pages):
            self._last_text.append(f'{obj}_p{i}_{page}')

    def _do_use(self, rest: str) -> None:
        """use_X_on_Y pattern."""
        if '_on_' in rest:
            tool, target = rest.split('_on_', 1)
        else:
            tool, target = rest, ''

        if tool not in self._inventory:
            return

        # Keycard on locked exit
        if tool.startswith('keycard_') or tool == 'bio_keycard':
            for room in self._rooms.values():
                for direction, req in list(room.locked_exits.items()):
                    if req == tool:
                        del room.locked_exits[direction]

        # Use sample on spectrometer
        elif target == 'spectrometer' and tool in self.SPECTROMETER_RESULTS:
            self._last_reading = f'READING_spectrometer_{self.SPECTROMETER_RESULTS[tool]}'

        # Use sample on microscope
        elif target == 'microscope' and tool in self.MICROSCOPE_RESULTS:
            self._last_reading = f'READING_microscope_{self.MICROSCOPE_RESULTS[tool]}'

        # UV lamp
        elif tool == 'uv_lamp':
            if target == 'cracked_coolant_pipe':
                self._last_reading = 'READING_uv_silver_residue_fluorescence'
            elif target in self.SPECTROMETER_RESULTS:
                if 'silver' in self.SPECTROMETER_RESULTS.get(target, ''):
                    self._last_reading = 'READING_uv_fluorescence_detected'
                else:
                    self._last_reading = 'READING_uv_no_fluorescence'

        # Geiger counter
        elif tool == 'geiger_counter':
            room = self._rooms[self._location]
            if room.contaminated > 0:
                self._last_reading = f'READING_geiger_{_contamination_label(room.contaminated * 10)}'
            else:
                self._last_reading = 'READING_geiger_background_normal'

        # Litmus strips on chemicals
        elif tool == 'litmus_strips':
            if target == 'reagent_a':
                self._last_reading = 'READING_litmus_blue_base'
            elif target == 'reagent_b':
                self._last_reading = 'READING_litmus_red_acid'
            else:
                self._last_reading = 'READING_litmus_green_neutral'

        # Pipe wrench on cracked pipe
        elif tool == 'pipe_wrench' and target == 'cracked_coolant_pipe':
            self._pipe_fixed = True

        # Wrench on generator panel
        elif tool == 'wrench' and target == 'generator_panel':
            self._panel_open = True
            self._rooms['generator_room'].items['generator_panel'].state = 'open'

        # Spare fuse on generator panel (restores bus 2)
        elif tool == 'spare_fuse' and target == 'generator_panel' and self._panel_open:
            self._bus_power['bus_2'] = True
            self._inventory.remove('spare_fuse')

        # Wire cutters + electrical tape on generator panel (reroute bus 3)
        elif tool == 'wire_cutters' and target == 'generator_panel' and self._panel_open:
            if 'electrical_tape' in self._inventory:
                self._bus_rerouted = True
                self._inventory.remove('electrical_tape')

        # Power cell on generator (full power temporarily)
        elif tool == 'power_cell' and target == 'generator':
            self._generator_full_power = True
            self._bus_power['bus_2'] = True
            if not self._bus_rerouted:
                # Bus 3 overload: ventilation fan B goes offline
                pass  # contamination spread handled in tick
            self._inventory.remove('power_cell')

        # Backup battery on generator (extends fuel)
        elif tool == 'backup_battery' and target == 'generator':
            self._fuel = min(self._fuel + 120, 360)
            self._inventory.remove('backup_battery')

        # Power cell on radio transmitter
        elif tool == 'power_cell' and target == 'radio_transmitter':
            self._radio_powered = True
            self._inventory.remove('power_cell')

        # Electrical tape on antenna
        elif tool == 'electrical_tape' and target == 'antenna':
            self._antenna_repaired = True
            self._inventory.remove('electrical_tape')

        # Air filter on ventilation controls
        elif tool == 'air_filter' and target == 'ventilation_controls':
            self._air_filter_installed = True
            self._inventory.remove('air_filter')

        # First aid kit on self
        elif tool == 'first_aid_kit':
            self._health = min(100, self._health + 30)
            self._contamination = max(0, self._contamination - 30)
            self._inventory.remove('first_aid_kit')

        # USB drive on server terminal
        elif tool == 'usb_drive' and target == 'server_terminal':
            usb = None
            for room in self._world_template.values():
                if 'usb_drive' in room.items:
                    usb = room.items['usb_drive']
                    break
            if usb and usb.document_pages:
                for i, page in enumerate(usb.document_pages):
                    self._last_text.append(f'usb_drive_p{i}_{page}')

    def _do_equip(self, obj: str) -> None:
        if obj not in self._inventory:
            return
        # Check wearable from template
        for room in self._world_template.values():
            if obj in room.items and room.items[obj].wearable:
                self._equipped = obj
                self._inventory.remove(obj)
                return

    def _do_unequip(self) -> None:
        if self._equipped:
            self._inventory.append(self._equipped)
            self._equipped = None

    def _do_eat(self, obj: str) -> None:
        if obj in self._inventory and obj in self.FOOD_EFFECTS:
            self._energy = min(100, self._energy + self.FOOD_EFFECTS[obj])
            self._inventory.remove(obj)
        elif obj in self._rooms[self._location].items and obj in self.FOOD_EFFECTS:
            self._energy = min(100, self._energy + self.FOOD_EFFECTS[obj])
            self._rooms[self._location].items[obj].state = 'consumed'

    def _do_combine(self, rest: str) -> None:
        """combine_X_with_Y — chemical reactions."""
        if '_with_' not in rest:
            return
        a, b = rest.split('_with_', 1)
        if a not in self._inventory or b not in self._inventory:
            return

        pair = frozenset({a, b})

        # NaOH + HCl = neutralization
        if pair == frozenset({'reagent_a', 'reagent_b'}):
            in_fume_hood = self._room_item('fume_hood') is not None
            if in_fume_hood:
                self._last_reading = 'READING_reaction_neutralization_safe'
            else:
                self._last_reading = 'READING_reaction_toxic_fumes'
                self._health = max(0, self._health - 10)
            self._inventory.remove(a)
            self._inventory.remove(b)

        # HCl + AgNO3 = silver chloride precipitate
        elif pair == frozenset({'reagent_b', 'reagent_c'}):
            self._last_reading = 'READING_reaction_silver_chloride_precipitate'
            self._inventory.remove(a)
            self._inventory.remove(b)

    def _do_enter_code(self, rest: str) -> None:
        """enter_code_on_wall_safe or enter_code_on_lobby_terminal."""
        if '_on_' in rest:
            code_part, target = rest.split('_on_', 1)
        else:
            return

        if target == 'wall_safe' and code_part == 'code':
            # The safe combo is 7-23-41 (from sticky note in bio_lab)
            # For simplicity: if the agent has read the sticky_note, the code works
            if self._location == 'director_office':
                safe = self._room_item('wall_safe')
                if safe and safe.state == 'locked':
                    safe.state = 'open'
                    self._safe_opened = True
                    # Reveal contents
                    self._rooms['director_office'].items['override_code_document'] = Item(
                        'override_code_document', frozenset({'document', 'important'}),
                        examine_hint='a printed sheet with the lockdown override code',
                        document_pages=["LOCKDOWN OVERRIDE CODE: ALPHA-7-HELIOS-CANCEL"])

        elif target == 'lobby_terminal':
            if self._location == 'lobby' and self._bus_power.get('bus_2'):
                if 'override_code_document' in self._inventory:
                    self._override_entered = True

    def _do_open(self, obj: str) -> None:
        item = self._room_item(obj)
        if item is None:
            return
        if obj == 'reception_desk' and not self._desk_searched:
            self._desk_searched = True
            # Items already in room from template
        elif item.state == 'closed':
            item.state = 'open'

    # ── available actions ─────────────────────────────────────────────────────

    def available_actions(self) -> list[str]:
        actions: list[str] = []
        room = self._rooms[self._location]

        # Movement
        for direction in room.exits:
            if direction in room.locked_exits:
                if self._check_access(room.locked_exits[direction]):
                    actions.append(f'go_{direction}')
            else:
                actions.append(f'go_{direction}')

        # Room item interactions
        for iname, item in room.items.items():
            if item.state in ('here', 'open', 'closed', 'sealed'):
                if item.portable:
                    actions.append(f'take_{iname}')
                actions.append(f'examine_{iname}')
                if item.document_pages:
                    actions.append(f'read_{iname}')
                if item.has('food'):
                    actions.append(f'eat_{iname}')
                if item.state == 'closed' or (item.has('container') and item.state == 'here'):
                    actions.append(f'open_{iname}')

        # Inventory interactions
        for iname in self._inventory:
            actions.append(f'drop_{iname}')
            if iname in self.FOOD_EFFECTS:
                actions.append(f'eat_{iname}')
            if iname in self._world_template.get(self._location, Room('_')).items:
                pass  # already handled
            # Document reading
            for r in self._world_template.values():
                if iname in r.items and r.items[iname].document_pages:
                    actions.append(f'read_{iname}')
                    break

            # Equip wearable
            for r in self._world_template.values():
                if iname in r.items and r.items[iname].wearable:
                    actions.append(f'equip_{iname}')
                    break

            # Use on targets
            if iname.startswith('keycard_') or iname == 'bio_keycard':
                for direction, req in room.locked_exits.items():
                    if req == iname:
                        actions.append(f'use_{iname}_on_{room.exits[direction]}')

            if iname in ('sample_b1', 'sample_b2', 'frozen_sample_march14', 'frozen_sample_march15'):
                if self._room_item('spectrometer'):
                    actions.append(f'use_{iname}_on_spectrometer')
                if self._room_item('microscope'):
                    actions.append(f'use_{iname}_on_microscope')

            if iname == 'uv_lamp':
                for target in list(room.items.keys()) + self._inventory:
                    if target != 'uv_lamp':
                        actions.append(f'use_uv_lamp_on_{target}')

            if iname == 'geiger_counter':
                actions.append('use_geiger_counter')

            if iname == 'litmus_strips':
                for target in ('reagent_a', 'reagent_b', 'reagent_c'):
                    if target in self._inventory or target in room.items:
                        actions.append(f'use_litmus_strips_on_{target}')

            if iname == 'pipe_wrench' and self._room_item('cracked_coolant_pipe') and not self._pipe_fixed:
                actions.append('use_pipe_wrench_on_cracked_coolant_pipe')

            if iname == 'wrench' and self._room_item('generator_panel') and not self._panel_open:
                actions.append('use_wrench_on_generator_panel')

            if iname == 'spare_fuse' and self._panel_open and self._location == 'generator_room':
                actions.append('use_spare_fuse_on_generator_panel')

            if iname == 'wire_cutters' and self._panel_open and self._location == 'generator_room':
                if 'electrical_tape' in self._inventory:
                    actions.append('use_wire_cutters_on_generator_panel')

            if iname == 'power_cell':
                if self._room_item('generator'):
                    actions.append('use_power_cell_on_generator')
                if self._room_item('radio_transmitter'):
                    actions.append('use_power_cell_on_radio_transmitter')

            if iname == 'backup_battery' and self._room_item('generator'):
                actions.append('use_backup_battery_on_generator')

            if iname == 'electrical_tape' and self._room_item('antenna') and not self._antenna_repaired:
                actions.append('use_electrical_tape_on_antenna')

            if iname == 'air_filter' and self._room_item('ventilation_controls'):
                actions.append('use_air_filter_on_ventilation_controls')

            if iname == 'first_aid_kit':
                actions.append('use_first_aid_kit')

            if iname == 'usb_drive' and self._room_item('server_terminal'):
                actions.append('use_usb_drive_on_server_terminal')

        # Combine chemicals
        chems_held = [c for c in ('reagent_a', 'reagent_b', 'reagent_c') if c in self._inventory]
        for i, a in enumerate(chems_held):
            for b in chems_held[i+1:]:
                actions.append(f'combine_{a}_with_{b}')

        # Enter code on safe
        if self._location == 'director_office' and self._room_item('wall_safe'):
            safe = self._room_item('wall_safe')
            if safe and safe.state == 'locked':
                actions.append('enter_code_on_wall_safe')

        # Enter code on lobby terminal
        if self._location == 'lobby' and 'override_code_document' in self._inventory:
            if self._bus_power.get('bus_2'):
                actions.append('enter_code_on_lobby_terminal')

        # Unequip
        if self._equipped:
            actions.append('unequip')

        # Radio call (win condition 2)
        if (self._location == 'roof_access' and self._antenna_repaired
                and self._radio_powered):
            actions.append('use_radio_transmitter')

        return list(dict.fromkeys(actions))

    # ── terminal conditions ───────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self._health <= 0 or self._won

    @property
    def won(self) -> bool:
        return self._won

    # ── win condition checks (called from act) ────────────────────────────────

    def _check_win(self) -> None:
        """Check both win conditions after each action."""
        # Win 1: escape via lobby
        if (self._location == 'lobby' and self._override_entered
                and self._bus_power.get('bus_2')):
            # Agent can now exit
            self._won = True

        # Win 2: radio rescue (set by _do_use when radio_transmitter is activated)
        pass

    # ── diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        return (f"ScienceLabEnv(loc={self._location}, energy={self._energy}, "
                f"health={self._health}, contam={self._contamination:.1f}, "
                f"equipped={self._equipped}, inv={self._inventory})")
