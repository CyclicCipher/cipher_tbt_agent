"""TextWorldEnv — semantic learning through affordances.

The central thesis: an agent learns what objects ARE by discovering what
it can DO with them and what happens as a result.  No labels, no ontology
given up front.  Semantic categories emerge from repeated interaction.

Self-awareness via agent_topology()
-------------------------------------
Every observation is split into two structural streams using edge types:

  intero (edge type 1) — interoceptive tokens: self-state
      AT_{room}         current location
      HUNGER_{label}    hunger level (sated / comfortable / hungry / ravenous / starving)
      HEALTH_{label}    health level (healthy / hurt / wounded / critical)
      HOLD_{item}       items in inventory (HOLD_nothing if empty)
      TORCH_lit         only present when carrying a lit torch

  extero (edge type 0) — exteroceptive tokens: world-state
      SEE_{item}        item present in current room
      PROP_{item}_{p}   property of item (edible, sharp, hot, container, ...)
      STATE_{item}_{s}  non-default state (open, locked, lit, full, ...)
      EXIT_{dir}_open   accessible exit
      EXIT_{dir}_locked locked exit

The AgentLoop feeds the chosen action BETWEEN observations with edge type 2
(action).  This makes agency structural: the MorphismGraph learns sequences

    intero/extero tokens -[action]-> action_token
                                      |
    next intero/extero tokens  (fresh sequence start)

Without this structural distinction the model cannot tell whether
"hunger decreased" because the agent ate something or because of some
spontaneous world event — the schizophrenic case.  With agent_topology(),
causal agency is visible in the graph structure without any per-experiment
boilerplate.

Semantic categories taught
--------------------------
FOOD         apple, bread, berries, mushroom   — eat → hunger improves
POISON       poison_berry, poison_mush         — eat → health worsens
TOOL         knife, key, torch, rope           — use → specific effect
CONTAINER    chest, bucket                     — open/fill → holds items
HAZARD       fire, spikes                      — touch → health worsens
LIQUID       water                             — drink/fill bucket

Multi-step affordances (requires planning):
    find key → use key on chest → chest opens → take treasure
    take bucket → fill at well → carry water → douse fire
    take torch → light at fire → torch lit → enter dark room

World layout
------------
    garden <-> kitchen <-> cellar   (cellar locked until key found)
    garden has path to forest (always open)

Rooms and objects are reset between episodes via reset().
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Optional

from ..environment import Environment
from ..core.topology import agent_topology, Topology


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Item:
    name:         str
    props:        frozenset[str]          # edible, poisonous, sharp, hot, …
    state:        str         = 'here'    # here|consumed|broken|lit|open|locked|full
    portable:     bool        = True      # can the agent pick this up?
    examine_hint: str         = ''        # shown on EXAMINE (not a property tag)

    def has(self, *ps: str) -> bool:
        return all(p in self.props for p in ps)


@dataclass
class Room:
    name:         str
    items:        dict[str, Item]         = field(default_factory=dict)
    exits:        dict[str, str]          = field(default_factory=dict)
    locked_exits: set[str]               = field(default_factory=set)
    dark:         bool                   = False   # needs torch to see objects


# ── Initial world factory ─────────────────────────────────────────────────────

def _build_world() -> dict[str, Room]:
    """Construct the initial world.  Called by reset(); never mutated directly."""

    kitchen = Room(
        name  = 'kitchen',
        items = {
            'apple':  Item('apple',  frozenset({'edible', 'small', 'fragile'}),
                           examine_hint='round and red'),
            'bread':  Item('bread',  frozenset({'edible', 'small'}),
                           examine_hint='a crusty loaf'),
            'knife':  Item('knife',  frozenset({'sharp', 'small', 'tool'}),
                           examine_hint='a short kitchen blade'),
            'bucket': Item('bucket', frozenset({'container', 'small'}),
                           examine_hint='an empty wooden bucket'),
        },
        exits = {'south': 'garden', 'down': 'cellar'},
        locked_exits = {'down'},        # need key from garden
    )

    garden = Room(
        name  = 'garden',
        items = {
            'berries':      Item('berries',      frozenset({'edible', 'small', 'fragile'}),
                                 examine_hint='small red berries'),
            'mushroom':     Item('mushroom',     frozenset({'edible', 'small', 'fungus'}),
                                 examine_hint='brown, smells earthy'),
            'poison_berry': Item('poison_berry', frozenset({'poisonous', 'small', 'fragile'}),
                                 examine_hint='black berries with a faint sheen'),
            'fire':         Item('fire',         frozenset({'hot', 'dangerous', 'light_source'}),
                                 state='here', portable=False,
                                 examine_hint='a campfire, crackling'),
            'rope':         Item('rope',         frozenset({'flexible', 'long'}),
                                 examine_hint='a coil of rope'),
            'key':          Item('key',           frozenset({'tool', 'metal', 'small'}),
                                 examine_hint='an iron key'),  # unlocks cellar
            'well':         Item('well',          frozenset({'container', 'water_source'}),
                                 state='here', portable=False,
                                 examine_hint='a stone well, full of water'),
        },
        exits = {'north': 'kitchen', 'east': 'forest'},
    )

    cellar = Room(
        name  = 'cellar',
        dark  = True,           # torch required to see items
        items = {
            'chest':   Item('chest',   frozenset({'container', 'heavy', 'locked'}),
                            state='locked', portable=False,
                            examine_hint='a heavy iron-bound chest'),
            'torch':   Item('torch',   frozenset({'tool', 'flammable', 'small'}),
                            examine_hint='an unlit wooden torch'),
            'gem':     Item('gem',     frozenset({'valuable', 'small'}),
                            examine_hint='a sparkling gem'),  # inside chest
            'old_key': Item('old_key', frozenset({'tool', 'metal', 'small'}),
                            examine_hint='a tarnished key'),  # inside chest
        },
        exits = {'up': 'kitchen'},
    )

    forest = Room(
        name  = 'forest',
        items = {
            'deer':        Item('deer',        frozenset({'animal', 'heavy'}),
                                portable=False,
                                examine_hint='a deer grazing peacefully'),
            'mushroom2':   Item('mushroom2',   frozenset({'edible', 'small', 'fungus'}),
                                examine_hint='similar to the garden mushroom'),
            'poison_mush': Item('poison_mush', frozenset({'poisonous', 'small', 'fungus'}),
                                examine_hint='looks almost identical to an edible mushroom'),
            'stream':      Item('stream',      frozenset({'liquid', 'cold', 'water_source'}),
                                portable=False,
                                examine_hint='a clear shallow stream'),
        },
        exits = {'west': 'garden'},
    )

    return {
        'kitchen': kitchen,
        'garden':  garden,
        'cellar':  cellar,
        'forest':  forest,
    }


# ── Vitals helpers ────────────────────────────────────────────────────────────

def _hunger_label(n: int) -> str:
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


# ── TextWorldEnv ──────────────────────────────────────────────────────────────

class TextWorldEnv(Environment):
    """Minimal semantically-rich text world for affordance learning.

    Uses agent_topology() to split the observation stream into two structural
    streams (intero and extero).  The AgentLoop feeds the chosen action between
    observations using the 'action' edge type — no per-environment boilerplate.

    Semantic categories the model should discover purely from interaction:
      FOOD:      objects where eat → hunger label improves next step
      POISON:    objects where eat → health label worsens next step
      TOOL:      objects where use(target) → specific non-trivial world change
      CONTAINER: objects that reveal new items when opened
      HAZARD:    objects/states where touch → health label worsens next step

    The free-energy objective F = -EIG(extero) + PE(intero) is tracked by
    the AgentLoop automatically once it receives intero_etypes from this env.
    """

    # Hunger/health eat amounts
    EAT_EFFECTS = {
        'apple':        ('hunger', +15),
        'bread':        ('hunger', +30),
        'berries':      ('hunger', +10),
        'mushroom':     ('hunger', +8),
        'mushroom2':    ('hunger', +8),
        'poison_berry': ('health', -25),
        'poison_mush':  ('health', -20),
    }

    def __init__(self, seed: int = 0) -> None:
        self._topo           = agent_topology()
        self._rng            = random.Random(seed)
        self._world_template = _build_world()
        # chest starts with gem+old_key hidden (not in room items until opened)
        self._chest_contents = ['gem', 'old_key']
        self.reset()

    @property
    def topology(self) -> Topology:
        return self._topo

    @property
    def intero_etypes(self) -> frozenset[int]:
        """Interoceptive tokens are tagged with the 'intero' edge type."""
        return frozenset({self._topo.registry.code('intero')})

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._rooms      = copy.deepcopy(self._world_template)
        self._location   = 'kitchen'
        self._inventory: list[str] = []
        self._hunger     = 70
        self._health     = 100
        self._torch_lit  = False

    # ── observe ────────────────────────────────────────────────────────────────

    def observe(self) -> list[tuple[str, Optional[int]]]:
        """Emit the full observation sequence for this timestep.

        Token structure (edge types from agent_topology):
          intero tokens  — agent body state (AT_, HUNGER_, HEALTH_, HOLD_)
          extero tokens  — world state (SEE_, PROP_, STATE_, EXIT_)

        The action the agent just took is NOT included here — the AgentLoop
        feeds it separately with edge type 'action', which is what makes
        agency structural rather than a naming convention.

        First token in the sequence uses None etype (sequence start).
        """
        reg    = self._topo.registry
        intero = reg.code('intero')
        extero = reg.code('extero')

        toks: list[tuple[str, Optional[int]]] = []

        def add_intero(val: str) -> None:
            toks.append((val, None if not toks else intero))

        def add_extero(val: str) -> None:
            toks.append((val, None if not toks else extero))

        # ── Interoception (self-model) ─────────────────────────────────────────
        add_intero(f'AT_{self._location}')
        add_intero(f'HUNGER_{_hunger_label(self._hunger)}')
        add_intero(f'HEALTH_{_health_label(self._health)}')
        if self._inventory:
            for item in self._inventory:
                add_intero(f'HOLD_{item}')
                if item == 'torch' and self._torch_lit:
                    add_intero('TORCH_lit')
        else:
            add_intero('HOLD_nothing')

        # ── Exteroception (world model) ────────────────────────────────────────
        room = self._rooms[self._location]

        # Dark room: items invisible without lit torch
        visible = not room.dark or self._torch_lit

        if room.dark and not visible:
            add_extero('dark')
        else:
            for iname, item in room.items.items():
                if item.state in ('here', 'open', 'locked', 'lit', 'full'):
                    add_extero(f'SEE_{iname}')
                    for p in sorted(item.props):
                        add_extero(f'PROP_{iname}_{p}')
                    if item.state != 'here':
                        add_extero(f'STATE_{iname}_{item.state}')

        # Exits
        for direction, target in room.exits.items():
            locked = direction in room.locked_exits
            add_extero(f'EXIT_{direction}_{"locked" if locked else "open"}')

        return toks

    # ── act ────────────────────────────────────────────────────────────────────

    def act(self, action: str) -> None:
        """Execute action, advancing the world state.

        Consequences are visible in the NEXT call to observe() as changed
        intero/extero tokens — the structural affordance record.
        """
        parts = action.split('_')          # e.g. 'eat_apple' -> ['eat', 'apple']
        verb  = parts[0]
        obj   = '_'.join(parts[1:]) if len(parts) > 1 else ''

        if   verb == 'eat':    self._do_eat(obj)
        elif verb == 'take':   self._do_take(obj)
        elif verb == 'drop':   self._do_drop(obj)
        elif verb == 'use':    self._do_use(parts)
        elif verb == 'open':   self._do_open(obj)
        elif verb == 'fill':   self._do_fill(obj)
        elif verb == 'drink':  self._do_drink(obj)
        elif verb == 'go':     self._do_go(obj)
        elif verb == 'examine':self._do_examine(obj)
        elif verb == 'touch':  self._do_touch(obj)

        # Hunger ticks down every action
        self._hunger = max(0, self._hunger - 1)
        if self._hunger == 0:
            self._health = max(0, self._health - 3)

    # ── action implementations ─────────────────────────────────────────────────

    def _room_item(self, name: str) -> Optional[Item]:
        return self._rooms[self._location].items.get(name)

    def _do_eat(self, obj: str) -> None:
        if obj in self._inventory:
            src = 'inventory'
        elif obj in self._rooms[self._location].items:
            src = 'room'
        else:
            return

        item = (self._rooms[self._location].items[obj] if src == 'room' else None)
        if item and item.state != 'here':
            return

        if obj not in self.EAT_EFFECTS:
            return

        stat, delta = self.EAT_EFFECTS[obj]
        if stat == 'hunger':
            self._hunger = min(100, self._hunger + delta)
        else:
            self._health = max(0, self._health + delta)

        if src == 'inventory':
            self._inventory.remove(obj)
        else:
            self._rooms[self._location].items[obj].state = 'consumed'

    def _do_take(self, obj: str) -> None:
        item = self._room_item(obj)
        if item is None or item.state not in ('here', 'open', 'full'):
            return
        if not item.portable:
            return
        self._inventory.append(obj)
        del self._rooms[self._location].items[obj]

    def _do_drop(self, obj: str) -> None:
        if obj not in self._inventory:
            return
        self._inventory.remove(obj)
        item_def = None
        for room in self._world_template.values():
            if obj in room.items:
                item_def = copy.deepcopy(room.items[obj])
                break
        if item_def is None:
            item_def = Item(obj, frozenset())
        self._rooms[self._location].items[obj] = item_def

    def _do_use(self, parts: list[str]) -> None:
        if len(parts) >= 4 and parts[2] == 'on':
            tool, target = parts[1], '_'.join(parts[3:])
        elif len(parts) >= 2:
            tool, target = parts[1], ''
        else:
            return

        if tool not in self._inventory:
            return

        if tool == 'key' and target == 'cellar':
            room = self._rooms['kitchen']
            room.locked_exits.discard('down')

        elif tool in ('key', 'old_key') and target == 'chest':
            chest = self._rooms['cellar'].items.get('chest')
            if chest and chest.state == 'locked':
                chest.state = 'open'
                for citem in self._chest_contents:
                    orig = None
                    for room in self._world_template.values():
                        if citem in room.items:
                            orig = copy.deepcopy(room.items[citem]); break
                    if orig is None:
                        orig = Item(citem, frozenset({'valuable', 'small'}))
                    self._rooms['cellar'].items[citem] = orig

        elif tool == 'torch' and target == 'fire':
            fire = self._room_item('fire')
            if fire and fire.has('hot'):
                self._torch_lit = True

        elif tool == 'bucket':
            water_src = None
            for wname in ('well', 'stream', 'water'):
                wi = self._room_item(wname)
                if wi and (wi.has('water_source') or wi.has('liquid')):
                    water_src = wname; break
            if water_src:
                for room in self._rooms.values():
                    if 'bucket' in room.items:
                        room.items['bucket'] = Item(
                            'bucket', frozenset({'container', 'small', 'full', 'liquid'}),
                            state='full')
                if 'bucket' in self._inventory:
                    self._inventory.remove('bucket')
                    self._inventory.append('bucket_full')

        elif tool == 'bucket_full':
            fire = self._room_item('fire')
            if fire:
                fire.state = 'consumed'
            if 'bucket_full' in self._inventory:
                self._inventory.remove('bucket_full')
                self._inventory.append('bucket')

        elif tool == 'knife' and target == 'rope':
            if 'rope' in self._rooms[self._location].items:
                self._rooms[self._location].items['rope'].state = 'consumed'
            elif 'rope' in self._inventory:
                self._inventory.remove('rope')

    def _do_open(self, obj: str) -> None:
        item = self._room_item(obj)
        if item is None or item.state in ('open', 'consumed'):
            return
        if item.state == 'locked':
            return   # needs a key; use_key_on_X is the right action
        item.state = 'open'

    def _do_fill(self, obj: str) -> None:
        """Alias: fill bucket -> use bucket on water source."""
        self._do_use(['use', obj, 'on', 'well'])

    def _do_drink(self, obj: str) -> None:
        found = False
        for wsrc in ('well', 'stream', 'water'):
            if self._room_item(wsrc) is not None:
                found = True; break
        if 'bucket_full' in self._inventory:
            self._inventory.remove('bucket_full')
            self._inventory.append('bucket')
            found = True
        if found:
            self._hunger = min(100, self._hunger + 5)
            self._health = min(100, self._health + 5)

    def _do_go(self, direction: str) -> None:
        room = self._rooms[self._location]
        if direction not in room.exits:
            return
        if direction in room.locked_exits:
            return
        self._location = room.exits[direction]

    def _do_examine(self, obj: str) -> None:
        pass   # examine is a no-op in terms of state change; it provides info
               # through the existing PROP tokens visible in exteroception

    def _do_touch(self, obj: str) -> None:
        item = self._room_item(obj)
        if item is None:
            return
        if item.has('hot'):
            self._health = max(0, self._health - 10)
        elif item.has('sharp'):
            self._health = max(0, self._health - 5)

    # ── available actions ──────────────────────────────────────────────────────

    def available_actions(self) -> list[str]:
        actions: list[str] = []
        room = self._rooms[self._location]

        # Movement
        for direction in room.exits:
            if direction not in room.locked_exits:
                actions.append(f'go_{direction}')

        # Item interactions (room)
        for iname, item in room.items.items():
            if item.state in ('here', 'open', 'full', 'lit'):
                if item.portable:
                    actions.append(f'take_{iname}')
                if item.has('edible') or item.has('poisonous'):
                    actions.append(f'eat_{iname}')
                if item.has('hot') or item.has('sharp'):
                    actions.append(f'touch_{iname}')
                if item.has('container') and item.state != 'locked':
                    actions.append(f'open_{iname}')
                actions.append(f'examine_{iname}')

        # Inventory interactions
        for iname in self._inventory:
            actions.append(f'drop_{iname}')
            if iname in self.EAT_EFFECTS:
                actions.append(f'eat_{iname}')
            if iname == 'key':
                if 'down' in room.locked_exits:
                    actions.append('use_key_on_cellar')
                chest = room.items.get('chest')
                if chest and chest.state == 'locked':
                    actions.append('use_key_on_chest')
            if iname == 'torch':
                fire = room.items.get('fire')
                if fire and fire.has('hot'):
                    actions.append('use_torch_on_fire')
            if iname == 'bucket':
                for wsrc in ('well', 'stream', 'water'):
                    if room.items.get(wsrc):
                        actions.append('fill_bucket')
                        break
            if iname == 'bucket_full':
                if room.items.get('fire'):
                    actions.append('use_bucket_full_on_fire')
                actions.append('drink_water')
            if iname == 'rope' and 'forest' in room.exits.values():
                actions.append('use_rope')
            if iname == 'knife':
                if 'rope' in room.items or 'rope' in self._inventory:
                    actions.append('use_knife_on_rope')

        return list(dict.fromkeys(actions))   # deduplicate, preserve order

    # ── terminal conditions ────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self._health <= 0

    @property
    def won(self) -> bool:
        return 'gem' in self._inventory

    # ── diagnostics ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        return (f"TextWorldEnv(location={self._location}, "
                f"hunger={self._hunger}, health={self._health}, "
                f"inv={self._inventory})")
