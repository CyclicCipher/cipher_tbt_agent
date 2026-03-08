"""TextWorld modality for the symbolic AI agent.

Two backends, selected automatically:
  - NanoTextEnv: zero-dependency pure-Python 5-room world (default).
  - TextWorldModality wrapping real TextWorld games (pass --game path.ulx).

The modality exposes the same observe/act interface as MinecraftModality:
  - `get_obs()` → dict with text, score, done, inventory, location, admissible
  - `send_action(cmd)` → (obs_text, reward, done)
  - `get_events()` → list of {'type': ..., ...} dicts (acquired/lost/moved)
  - `current_priority(engine)` → ('MODE', urgency, target) string

EFFECTFUL primitives (blocked during synthesis dry_run):
  tw_go, tw_take, tw_drop, tw_put, tw_open, tw_close, tw_unlock, tw_eat,
  tw_examine
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# TextWorld optional import
# ---------------------------------------------------------------------------

try:
    import textworld          # type: ignore
    import textworld.gym      # type: ignore
    import gym                # type: ignore
    _HAS_TW = True
except ImportError:
    _HAS_TW = False


# ===========================================================================
# NanoTextEnv — zero-dependency pure-Python text environment
# ===========================================================================

class NanoTextEnv:
    """A minimal 5-room text world for testing without TextWorld installed.

    World layout:
        living room  ←  kitchen  →  garden
                          ↓
                        cellar

        (bedroom is north of living room, door locked with brass key)

    Quest (max score 3):
      1. eat the red apple   (+1)
      2. take the brass key  (+1)
      3. unlock and enter the bedroom  (+1)
    """

    _ROOMS: Dict[str, Dict] = {
        'kitchen': {
            'desc': ('You are in the kitchen. There is a wooden counter here. '
                     'A red apple and a sharp knife rest on the counter. '
                     'A wooden box sits in the corner.'),
            'exits': {'west': 'living room', 'north': 'garden', 'down': 'cellar'},
            'items': {'red apple', 'sharp knife', 'wooden box'},
        },
        'living room': {
            'desc': ('You are in the living room. A comfortable sofa sits against '
                     'the wall. A bookshelf holds several dusty volumes. '
                     'An oak door leads north to the bedroom.'),
            'exits': {'east': 'kitchen'},
            'locked_exits': {'north': ('bedroom', 'brass key')},
            'items': set(),
        },
        'garden': {
            'desc': ('You are in the garden. Sunlight filters through the trees. '
                     'A brass key glints in the grass.'),
            'exits': {'south': 'kitchen'},
            'items': {'brass key'},
        },
        'cellar': {
            'desc': ('You are in the cellar. It is dark and musty. '
                     'Shelves of old jars line the walls.'),
            'exits': {'up': 'kitchen'},
            'items': set(),
        },
        'bedroom': {
            'desc': ('You are in the bedroom. A soft bed and a writing desk '
                     'stand here. This was the goal.'),
            'exits': {'south': 'living room'},
            'items': set(),
        },
    }

    # Items that can be eaten (give score).
    _EDIBLE: Set[str] = {'red apple'}
    # Items that unlock doors: {key_name: (from_room, direction, to_room)}
    _KEYS: Dict[str, Tuple[str, str, str]] = {
        'brass key': ('living room', 'north', 'bedroom'),
    }

    def __init__(self) -> None:
        self._room_states: Dict[str, Dict] = {}
        self._location: str = 'kitchen'
        self._inventory: Set[str] = set()
        self._score: float = 0.0
        self._done: bool = False
        self._unlocked: Set[str] = set()  # unlocked locked_exits (key names)
        self._ate: Set[str] = set()       # items eaten (score only once)
        self._scored_takes: Set[str] = set()  # items that already gave take bonus
        self._step_count: int = 0
        self._reset_rooms()

    def _reset_rooms(self) -> None:
        import copy
        self._room_states = {
            name: {**data, 'items': set(data['items'])}
            for name, data in self._ROOMS.items()
        }

    def reset(self) -> Tuple[str, dict]:
        self._location = 'kitchen'
        self._inventory = set()
        self._score = 0.0
        self._done = False
        self._unlocked = set()
        self._ate = set()
        self._scored_takes = set()
        self._step_count = 0
        self._reset_rooms()
        return self._observe(), self._info()

    def step(self, command: str) -> Tuple[str, float, bool, dict]:
        self._step_count += 1
        reward = 0.0
        cmd = command.strip().lower()
        feedback = self._execute(cmd)
        obs = self._observe() + '\n' + feedback if feedback else self._observe()
        info = self._info()
        return obs, reward, self._done, info

    # ------------------------------------------------------------------
    def _execute(self, cmd: str) -> str:
        room = self._room_states[self._location]

        # -- movement --
        m = re.match(r'^go\s+(\w+)$', cmd)
        if not m:
            m = re.match(r'^(north|south|east|west|up|down)$', cmd)
            if m:
                cmd_dir = m.group(1)
            else:
                cmd_dir = None
        else:
            cmd_dir = m.group(1)

        if cmd_dir:
            exits = room.get('exits', {})
            locked = room.get('locked_exits', {})
            if cmd_dir in exits:
                self._location = exits[cmd_dir]
                return ''
            elif cmd_dir in locked:
                dest, key_needed = locked[cmd_dir]
                if key_needed in self._unlocked:
                    self._location = dest
                    if dest == 'bedroom':
                        self._score += 1.0
                        self._done = True
                    return ''
                else:
                    return f'The door to the {cmd_dir} is locked.'
            return f'You cannot go {cmd_dir} from here.'

        # -- take --
        m = re.match(r'^take\s+(?:the\s+)?(.+)$', cmd)
        if m:
            item = m.group(1).strip()
            if item in room['items']:
                room['items'].discard(item)
                self._inventory.add(item)
                if item == 'brass key' and item not in self._scored_takes:
                    self._score += 1.0
                    self._scored_takes.add(item)
                return f'You take the {item}.'
            return f'There is no {item} here.'

        # -- drop --
        m = re.match(r'^drop\s+(?:the\s+)?(.+)$', cmd)
        if m:
            item = m.group(1).strip()
            if item in self._inventory:
                self._inventory.discard(item)
                room['items'].add(item)
                return f'You drop the {item}.'
            return f'You do not have the {item}.'

        # -- eat --
        m = re.match(r'^eat\s+(?:the\s+)?(.+)$', cmd)
        if m:
            item = m.group(1).strip()
            if item in self._inventory:
                if item in self._EDIBLE and item not in self._ate:
                    self._inventory.discard(item)
                    self._ate.add(item)
                    self._score += 1.0
                    return f'You eat the {item}. Delicious!'
                elif item in self._EDIBLE:
                    return f'You already ate the {item}.'
                return f'You cannot eat the {item}.'
            return f'You do not have the {item}.'

        # -- unlock --
        m = re.match(r'^unlock\s+(?:the\s+)?(.+?)\s+with\s+(?:the\s+)?(.+)$', cmd)
        if m:
            target, key = m.group(1).strip(), m.group(2).strip()
            for key_name, (from_room, direction, _) in self._KEYS.items():
                if key in key_name or key_name in key:
                    if key_name in self._inventory:
                        self._unlocked.add(key_name)
                        return f'You unlock the {target} with the {key_name}.'
                    return f'You do not have the {key_name}.'
            return f'You cannot unlock that with the {key}.'

        # -- examine --
        m = re.match(r'^(?:examine|look at|x)\s+(?:the\s+)?(.+)$', cmd)
        if m:
            thing = m.group(1).strip()
            if thing == 'room' or thing == 'around':
                return ''
            for item in list(room['items']) + list(self._inventory):
                if thing in item or item in thing:
                    return f'You examine the {item}. It seems ordinary.'
            return f'You see no {thing} to examine.'

        # -- look --
        if cmd in ('look', 'l'):
            return ''

        # -- inventory --
        if cmd in ('inventory', 'i', 'inv'):
            if self._inventory:
                return 'You are carrying: ' + ', '.join(sorted(self._inventory)) + '.'
            return 'You are not carrying anything.'

        return f'I do not understand "{cmd}".'

    def _observe(self) -> str:
        room = self._room_states[self._location]
        desc = room['desc']
        items = sorted(room['items'])
        exits = list(room.get('exits', {}).keys())
        locked = room.get('locked_exits', {})
        for direction, (dest, key_needed) in locked.items():
            if direction not in exits:
                if key_needed in self._unlocked:
                    exits.append(direction)          # door now open
                else:
                    exits.append(f'{direction} (locked)')
        inv_str = (', '.join(sorted(self._inventory))
                   if self._inventory else 'nothing')
        obs = (f'{desc}\n'
               f'Items here: {", ".join(items) if items else "none"}.\n'
               f'Exits: {", ".join(exits)}.\n'
               f'You are carrying: {inv_str}.')
        return obs

    def _info(self) -> dict:
        room = self._room_states[self._location]
        # Build admissible commands
        adm: List[str] = ['look', 'inventory']
        exits = list(room.get('exits', {}).keys())
        locked = room.get('locked_exits', {})
        adm += [f'go {d}' for d in exits]
        for d, (dest, key_needed) in locked.items():
            # Only include locked exits in admissible once unlocked —
            # prevents the agent from bumping into locked doors and setting
            # spurious affordance blocks that the inference can't clear.
            if key_needed in self._unlocked:
                adm.append(f'go {d}')
        for item in sorted(room['items']):
            adm.append(f'take {item}')
        for item in sorted(self._inventory):
            adm.append(f'drop {item}')
            if item in self._EDIBLE:
                adm.append(f'eat {item}')
            for d, (_, key_needed) in locked.items():
                if key_needed in item or item in key_needed:
                    # Only offer unlock when the exit is not already unlocked.
                    if key_needed not in self._unlocked:
                        adm.append(f'unlock door with {item}')
        return {
            'admissible_commands': adm,
            'inventory': sorted(self._inventory),
            'score':     self._score,
            'won':       self._done,
        }


# ===========================================================================
# MicroTextWorld — procedural pure-Python text game generator
# ===========================================================================

_MTW_ROOM_NAMES = [
    'kitchen', 'garden', 'cellar', 'living room', 'bedroom', 'study',
    'attic', 'pantry', 'shed', 'library', 'hallway', 'workshop',
    'dining room', 'bathroom', 'greenhouse',
]
_MTW_ROOM_DESCS = {
    'kitchen':     'A kitchen with a worn counter. Copper pots hang from hooks.',
    'garden':      'A garden. Sunlight filters through the trees.',
    'cellar':      'A dark cellar. Shelves of old jars line the damp stone walls.',
    'living room': 'A comfortable living room. A sofa and bookshelf stand here.',
    'bedroom':     'A bedroom with a soft bed and a writing desk.',
    'study':       'A wood-panelled study lined with books. A lamp glows on the desk.',
    'attic':       'A dusty attic. Old trunks are stacked beneath the rafters.',
    'pantry':      'A small pantry. Shelves hold preserved foods and supplies.',
    'shed':        'A garden shed smelling of earth. Tools hang on the walls.',
    'library':     'A quiet library. Tall colour-coded shelves line the walls.',
    'hallway':     'A narrow hallway. Coat hooks line one wall.',
    'workshop':    'A cluttered workshop. A workbench runs along the far wall.',
    'dining room': 'A dining room with a long oak table and high-backed chairs.',
    'bathroom':    'A tiled bathroom. A claw-foot tub stands by the window.',
    'greenhouse':  'A warm greenhouse. Glass panes let in soft diffuse light.',
}
_MTW_DIR_PAIRS = [('north', 'south'), ('east', 'west'), ('up', 'down')]
_MTW_OPP: Dict[str, str] = {a: b for a, b in _MTW_DIR_PAIRS}
_MTW_OPP.update({b: a for a, b in _MTW_DIR_PAIRS})
_MTW_FOOD       = ['red apple', 'banana', 'carrot', 'bread roll', 'orange']
_MTW_RAW        = ['raw potato', 'raw egg', 'raw meat']
_MTW_COOKED     = {'raw potato': 'baked potato', 'raw egg': 'fried egg',
                   'raw meat': 'cooked meat'}
_MTW_KEYS       = ['brass key', 'iron key', 'silver key', 'copper key', 'bronze key']
_MTW_CONTAINERS = ['wooden box', 'oak chest', 'wicker basket', 'tin can', 'glass jar']
_MTW_HEAT       = ['stove', 'fireplace', 'oven']
_MTW_MISC       = [
    'sharp knife', 'candle', 'old book', 'glass bottle', 'leather glove',
    'copper coin', 'small mirror', 'quill pen', 'brass compass', 'iron nail',
    'wooden spoon', 'clay pot',
]


class MicroTextWorld:
    """Procedural pure-Python text game generator. Zero dependencies.

    Generates a new random world on every call to reset().

    Quest types:
      'eat'    — find any food item and eat it                         (score 1)
      'fetch'  — carry a named item to a named target room            (score 1)
      'unlock' — find a key, unlock a door, enter the locked room     (score 2)
      'cook'   — find raw food near a heat source, cook it, eat it    (score 2)
      'put'    — find item, open container if needed, put item inside (score 2)

    Interface identical to NanoTextEnv:
      reset()  -> (obs: str, info: dict)
      step(cmd)-> (obs: str, reward: float, done: bool, info: dict)
      info keys: admissible_commands, inventory, score, won,
                 quest_type, quest_item, quest_dest, quest_container,
                 heat_src_here, containers_here
    """

    QUEST_TYPES = ['eat', 'fetch', 'unlock', 'cook', 'put']

    def __init__(
        self,
        n_rooms:    int = 6,
        quest_type: str = '',   # '' = random each episode
        seed:       Optional[int] = None,
        max_steps:  int = 100,
    ) -> None:
        self._n_rooms   = max(4, n_rooms)
        self._qt_fixed  = quest_type
        self._ep_seed   = seed
        self._max_steps = max_steps
        # State populated by _generate_world()
        self._rooms:           List[str]         = []
        self._exits:           Dict[str, Dict]   = {}
        self._locked_exits:    Dict[str, Dict]   = {}
        self._room_items:      Dict[str, Set]    = {}
        self._containers:      Dict[str, str]    = {}   # name -> room
        self._container_open:  Dict[str, bool]   = {}
        self._container_items: Dict[str, Set]    = {}
        self._heat_sources:    Dict[str, str]    = {}   # name -> room
        self._takeable:        Set[str]          = set()
        self._edible:          Set[str]          = set()
        self._cookable:        Dict[str, str]    = {}   # raw -> cooked
        self._inventory:       Set[str]          = set()
        self._location:        str               = ''
        self._score:           float             = 0.0
        self._done:            bool              = False
        self._unlocked:        Set[str]          = set()
        self._ate:             Set[str]          = set()
        self._cooked:          Set[str]          = set()
        self._fetch_done:      bool              = False
        self._put_done:        bool              = False
        self._step_count:      int               = 0
        self._quest:           Dict              = {}

    # ------------------------------------------------------------------
    def reset(self) -> Tuple[str, dict]:
        import random as _rnd
        rng = _rnd.Random(self._ep_seed)
        if self._ep_seed is not None:
            self._ep_seed += 1
        self._inventory  = set();  self._score = 0.0;  self._done = False
        self._unlocked   = set();  self._ate   = set(); self._cooked = set()
        self._fetch_done = False;  self._put_done = False;  self._step_count = 0
        self._generate_world(rng)
        return self._observe(), self._info()

    def step(self, command: str) -> Tuple[str, float, bool, dict]:
        self._step_count += 1
        cmd      = command.strip().lower()
        feedback = self._execute(cmd)
        obs      = self._observe() + (f'\n{feedback}' if feedback else '')
        done     = self._done or self._step_count >= self._max_steps
        return obs, 0.0, done, self._info()

    # ------------------------------------------------------------------
    # World generation
    # ------------------------------------------------------------------
    def _generate_world(self, rng) -> None:
        import random as _rnd
        # 1. Choose rooms
        pool = list(_MTW_ROOM_NAMES); rng.shuffle(pool)
        self._rooms = pool[:self._n_rooms]
        # 2. Spanning tree
        shuffled = list(self._rooms); rng.shuffle(shuffled)
        in_tree  = [shuffled[0]];  out_tree = shuffled[1:]
        raw_edges: List[Tuple[str, str]] = []
        while out_tree:
            a   = rng.choice(in_tree)
            idx = rng.randrange(len(out_tree))
            b   = out_tree.pop(idx)
            raw_edges.append((a, b))
            in_tree.append(b)
        # 3. Extra edges
        for _ in range(20):
            if len(raw_edges) >= self._n_rooms + 1:
                break
            a, b = rng.sample(self._rooms, 2)
            if (a, b) not in raw_edges and (b, a) not in raw_edges:
                raw_edges.append((a, b))
        # 4. Assign directions (greedy)
        self._exits       = {r: {} for r in self._rooms}
        self._locked_exits = {r: {} for r in self._rooms}
        pairs = list(_MTW_DIR_PAIRS)
        for a, b in raw_edges:
            rng.shuffle(pairs)
            for d_ab, d_ba in pairs:
                if d_ab not in self._exits[a] and d_ba not in self._exits[b]:
                    self._exits[a][d_ab] = b
                    self._exits[b][d_ba] = a
                    break
        # 5. Starting room + item storage
        self._location        = self._rooms[0]
        self._room_items      = {r: set() for r in self._rooms}
        self._containers      = {};  self._container_open  = {}
        self._container_items = {};  self._heat_sources    = {}
        self._takeable        = set(); self._edible = set(); self._cookable = {}
        # 6. Quest
        qt = self._qt_fixed or rng.choice(self.QUEST_TYPES)
        {'eat': self._gen_eat, 'fetch': self._gen_fetch,
         'unlock': self._gen_unlock, 'cook': self._gen_cook,
         'put': self._gen_put}[qt](rng)
        self._add_distractors(rng)

    def _place(self, item: str, room: str) -> None:
        self._room_items[room].add(item)
        self._takeable.add(item)

    def _other(self, rng, exclude: str = '') -> str:
        c = [r for r in self._rooms if r != exclude] or self._rooms
        return rng.choice(c)

    def _gen_eat(self, rng) -> None:
        food = rng.choice(_MTW_FOOD)
        room = self._other(rng)
        self._place(food, room);  self._edible.add(food)
        self._quest = {'type': 'eat', 'item': food, 'dest': '', 'key': '',
                       'container': '', 'max_score': 1,
                       'description': f'QUEST: eat the {food}.'}

    def _gen_fetch(self, rng) -> None:
        item   = rng.choice(_MTW_MISC)
        iroom  = self._other(rng)
        troom  = self._other(rng, exclude=iroom)
        self._place(item, iroom)
        self._quest = {'type': 'fetch', 'item': item, 'dest': troom, 'key': '',
                       'container': '', 'max_score': 1,
                       'description': f'QUEST: take the {item} to the {troom}.'}

    def _gen_unlock(self, rng) -> None:
        # Collect candidate edges to lock
        candidates = []
        for room in self._rooms[1:]:
            for r, exits in self._exits.items():
                for d, dest in list(exits.items()):
                    if dest == room and r != room:
                        candidates.append((r, d, room))
        if not candidates:
            self._gen_eat(rng); return
        from_room, direction, dest = rng.choice(candidates)
        rev = _MTW_OPP.get(direction, '')
        key = rng.choice(_MTW_KEYS)
        del self._exits[from_room][direction]
        if rev and rev in self._exits.get(dest, {}):
            del self._exits[dest][rev]
        self._locked_exits[from_room][direction] = (dest, key)
        if rev:  # always allow exit from locked room
            self._exits[dest][rev] = from_room
        key_room = self._other(rng, exclude=dest)
        self._place(key, key_room)
        self._quest = {'type': 'unlock', 'item': key, 'dest': dest, 'key': key,
                       'container': '', 'max_score': 2,
                       'description': f'QUEST: unlock the door and enter the {dest}.'}

    def _gen_cook(self, rng) -> None:
        raw    = rng.choice(_MTW_RAW)
        cooked = _MTW_COOKED[raw]
        heat   = rng.choice(_MTW_HEAT)
        hroom  = self._other(rng)
        self._heat_sources[heat] = hroom
        self._place(raw, hroom)     # raw food in same room as heat source
        self._cookable[raw] = cooked;  self._edible.add(cooked)
        self._quest = {'type': 'cook', 'item': raw, 'cooked': cooked, 'heat': heat,
                       'dest': '', 'key': '', 'container': '', 'max_score': 2,
                       'description': f'QUEST: cook the {raw} and eat it.'}

    def _gen_put(self, rng) -> None:
        item  = rng.choice(_MTW_MISC)
        cname = rng.choice(_MTW_CONTAINERS)
        iroom = self._other(rng)
        croom = self._other(rng)
        self._place(item, iroom)
        self._containers[cname] = croom
        self._container_open[cname] = False
        self._container_items[cname] = set()
        self._quest = {'type': 'put', 'item': item, 'dest': '', 'key': '',
                       'container': cname, 'max_score': 2,
                       'description': f'QUEST: put the {item} in the {cname}.'}

    def _add_distractors(self, rng) -> None:
        pool = _MTW_MISC + _MTW_FOOD
        for item in rng.sample(pool, min(rng.randint(2, 4), len(pool))):
            if item not in self._takeable:
                self._place(item, rng.choice(self._rooms))
                if item in _MTW_FOOD:
                    self._edible.add(item)

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------
    def _execute(self, cmd: str) -> str:  # noqa: C901
        room     = self._location
        room_obj = self._room_items[room]

        # -- movement --
        m = (re.match(r'^go\s+(\w+)$', cmd)
             or re.match(r'^(north|south|east|west|up|down)$', cmd))
        if m:
            d = m.group(1)
            if d in self._exits.get(room, {}):
                self._location = self._exits[room][d]
                self._check_fetch(); return ''
            if d in self._locked_exits.get(room, {}):
                dest, key = self._locked_exits[room][d]
                if key in self._unlocked:
                    self._location = dest
                    self._check_enter_locked(dest); return ''
                return f'The door to the {d} is locked.'
            return f'You cannot go {d} from here.'

        # -- take --
        m = re.match(r'^take\s+(?:the\s+)?(.+)$', cmd)
        if m:
            item = m.group(1).strip()
            if item in room_obj:
                room_obj.discard(item); self._inventory.add(item)
                return f'You take the {item}.'
            for cn, cr in self._containers.items():
                if cr == room and self._container_open.get(cn):
                    if item in self._container_items.get(cn, set()):
                        self._container_items[cn].discard(item)
                        self._inventory.add(item)
                        return f'You take the {item} from the {cn}.'
            return f'There is no {item} here.'

        # -- drop --
        m = re.match(r'^drop\s+(?:the\s+)?(.+)$', cmd)
        if m:
            item = m.group(1).strip()
            if item in self._inventory:
                self._inventory.discard(item); room_obj.add(item)
                return f'You drop the {item}.'
            return f'You do not have the {item}.'

        # -- eat --
        m = re.match(r'^eat\s+(?:the\s+)?(.+)$', cmd)
        if m:
            item = m.group(1).strip()
            if item not in self._inventory:
                return f'You do not have the {item}.'
            if item in self._edible and item not in self._ate:
                self._inventory.discard(item); self._ate.add(item)
                self._check_eat(item)
                return f'You eat the {item}. Delicious!'
            if item in self._edible:
                return f'You already ate the {item}.'
            return f'You cannot eat the {item}.'

        # -- cook --
        m = re.match(r'^cook\s+(?:the\s+)?(.+)$', cmd)
        if m:
            item = m.group(1).strip()
            if item not in self._inventory:
                return f'You do not have the {item}.'
            if item not in self._cookable:
                return f'You cannot cook the {item}.'
            if not any(s for s, sr in self._heat_sources.items() if sr == room):
                return 'There is no heat source here to cook with.'
            cooked = self._cookable[item]
            self._inventory.discard(item); self._inventory.add(cooked)
            self._cooked.add(cooked); self._edible.add(cooked)
            self._check_cook(); return f'You cook the {item}. You now have the {cooked}.'

        # -- open --
        m = re.match(r'^open\s+(?:the\s+)?(.+)$', cmd)
        if m:
            thing = m.group(1).strip()
            for cn, cr in self._containers.items():
                if cr == room and (thing in cn or cn in thing):
                    if self._container_open[cn]:
                        return f'The {cn} is already open.'
                    self._container_open[cn] = True
                    contents = self._container_items.get(cn, set())
                    inner = f' Inside: {", ".join(sorted(contents))}.' if contents else ''
                    return f'You open the {cn}.{inner}'
            return f'You cannot open the {thing}.'

        # -- close --
        m = re.match(r'^close\s+(?:the\s+)?(.+)$', cmd)
        if m:
            thing = m.group(1).strip()
            for cn, cr in self._containers.items():
                if cr == room and (thing in cn or cn in thing):
                    if not self._container_open[cn]:
                        return f'The {cn} is already closed.'
                    self._container_open[cn] = False
                    return f'You close the {cn}.'
            return f'You cannot close the {thing}.'

        # -- put X in Y --
        m = re.match(r'^put\s+(?:the\s+)?(.+?)\s+in(?:to)?\s+(?:the\s+)?(.+)$', cmd)
        if m:
            item, ctarget = m.group(1).strip(), m.group(2).strip()
            if item not in self._inventory:
                return f'You do not have the {item}.'
            for cn, cr in self._containers.items():
                if cr == room and (ctarget in cn or cn in ctarget):
                    if not self._container_open[cn]:
                        return f'The {cn} is closed.'
                    self._inventory.discard(item)
                    self._container_items[cn].add(item)
                    self._check_put(item, cn)
                    return f'You put the {item} in the {cn}.'
            return f'There is no {ctarget} here.'

        # -- unlock --
        m = re.match(r'^unlock\s+(?:the\s+)?(.+?)\s+with\s+(?:the\s+)?(.+)$', cmd)
        if m:
            key_used = m.group(2).strip()
            for d, (dest, key_needed) in self._locked_exits.get(room, {}).items():
                if key_used in key_needed or key_needed in key_used:
                    if key_needed in self._inventory:
                        self._unlocked.add(key_needed)
                        self._check_unlock()
                        return f'You unlock the door with the {key_needed}.'
                    return f'You do not have the {key_needed}.'
            return f'You cannot unlock that with the {key_used}.'

        if cmd in ('look', 'l'):        return ''
        if cmd in ('inventory', 'i', 'inv'):
            return ('You carry: ' + ', '.join(sorted(self._inventory)) + '.'
                    if self._inventory else 'You are not carrying anything.')
        return f'I do not understand "{cmd}".'

    # ------------------------------------------------------------------
    # Quest checks
    # ------------------------------------------------------------------
    def _check_eat(self, item: str) -> None:
        q = self._quest
        if q['type'] == 'eat' and item == q['item']:
            self._score += 1.0; self._done = True
        elif q['type'] == 'cook' and item == q.get('cooked', ''):
            self._score += 1.0; self._done = True

    def _check_cook(self) -> None:
        q = self._quest
        if q['type'] == 'cook':
            self._score += 1.0   # halfway; eating gives the second point

    def _check_fetch(self) -> None:
        q = self._quest
        if (q['type'] == 'fetch' and not self._fetch_done
                and self._location == q['dest'] and q['item'] in self._inventory):
            self._score += 1.0; self._done = True; self._fetch_done = True

    def _check_enter_locked(self, dest: str) -> None:
        q = self._quest
        if q['type'] == 'unlock' and dest == q['dest']:
            self._score += 1.0; self._done = True

    def _check_unlock(self) -> None:
        if self._quest['type'] == 'unlock':
            self._score += 1.0   # halfway

    def _check_put(self, item: str, container: str) -> None:
        q = self._quest
        if (q['type'] == 'put' and not self._put_done
                and item == q['item'] and container == q['container']):
            self._score += 1.0; self._done = True; self._put_done = True

    # ------------------------------------------------------------------
    # Observation + info
    # ------------------------------------------------------------------
    def _observe(self) -> str:
        room      = self._location
        desc      = _MTW_ROOM_DESCS.get(room, f'You are in the {room}.')
        items_here = sorted(self._room_items[room])
        exits_here = sorted(self._exits.get(room, {}).keys())
        for d, (dest, key) in self._locked_exits.get(room, {}).items():
            exits_here.append(d if key in self._unlocked else f'{d} (locked)')
        heat_here = [h for h, hr in self._heat_sources.items() if hr == room]
        inv_str   = ', '.join(sorted(self._inventory)) if self._inventory else 'nothing'
        parts     = [
            f'You are in the {room}. {desc}',
            f'Items here: {", ".join(items_here) if items_here else "none"}.',
        ]
        if heat_here:
            parts.append(f'Heat sources here: {", ".join(heat_here)}.')
        for cn, cr in self._containers.items():
            if cr == room:
                st  = 'open' if self._container_open[cn] else 'closed'
                inn = self._container_items.get(cn, set())
                inner = (f', containing {", ".join(sorted(inn))}' if self._container_open[cn] and inn else '')
                parts.append(f'There is a {cn} here ({st}{inner}).')
        parts.append(f'Exits: {", ".join(exits_here) if exits_here else "none"}.')
        parts.append(f'You are carrying: {inv_str}.')
        parts.append(self._quest.get('description', ''))
        return '\n'.join(parts)

    def _info(self) -> dict:
        room = self._location
        adm: List[str] = ['look', 'inventory']
        for d in sorted(self._exits.get(room, {}).keys()):
            adm.append(f'go {d}')
        for d, (dest, key) in self._locked_exits.get(room, {}).items():
            if key in self._unlocked:
                adm.append(f'go {d}')
        for item in sorted(self._room_items[room]):
            adm.append(f'take {item}')
        for cn, cr in self._containers.items():
            if cr == room and self._container_open.get(cn):
                for item in sorted(self._container_items.get(cn, set())):
                    adm.append(f'take {item}')
        for item in sorted(self._inventory):
            adm.append(f'drop {item}')
            if item in self._edible and item not in self._ate:
                adm.append(f'eat {item}')
            if item in self._cookable:
                if any(s for s, sr in self._heat_sources.items() if sr == room):
                    adm.append(f'cook {item}')
        for cn, cr in self._containers.items():
            if cr != room: continue
            if not self._container_open[cn]:
                adm.append(f'open {cn}')
            else:
                adm.append(f'close {cn}')
                for item in sorted(self._inventory):
                    adm.append(f'put {item} in {cn}')
        for d, (dest, key_needed) in self._locked_exits.get(room, {}).items():
            if key_needed not in self._unlocked and key_needed in self._inventory:
                adm.append(f'unlock door with {key_needed}')
        return {
            'admissible_commands': adm,
            'inventory':           sorted(self._inventory),
            'score':               self._score,
            'won':                 self._done,
            'quest_type':          self._quest.get('type', ''),
            'quest_item':          self._quest.get('item', ''),
            'quest_dest':          self._quest.get('dest', ''),
            'quest_container':     self._quest.get('container', ''),
            'heat_src_here':       [h for h, r in self._heat_sources.items() if r == room],
            'containers_here':     [(cn, self._container_open[cn])
                                    for cn, cr in self._containers.items() if cr == room],
        }


# ===========================================================================
# TextWorldModality — wraps NanoTextEnv, MicroTextWorld, or real TextWorld
# ===========================================================================

class TextWorldModality:
    """Modality for text-adventure environments.

    Automatically uses NanoTextEnv (zero deps) unless a TextWorld game file
    is provided via game_path AND TextWorld is installed.

    Primitives exposed to the process interpreter:
      Observation (safe, no side-effects):
        tw_obs        → current observation text (str)
        tw_score      → cumulative score (float)
        tw_done       → episode finished flag (bool)
        tw_inventory  → sorted list of held items (list)
        tw_location   → current room name (str)
        tw_admissible → list of valid commands (list)

      Action (effectful — blocked during synthesis dry_run):
        tw_go(direction)            → move in direction
        tw_take(item)               → pick up item
        tw_drop(item)               → drop item
        tw_eat(item)                → eat item
        tw_examine(thing)           → examine object
        tw_put(item, container)     → put item in container
        tw_unlock(target, key)      → unlock door/container with key
    """

    EFFECTFUL: frozenset = frozenset({
        'tw_go', 'tw_take', 'tw_drop', 'tw_put',
        'tw_open', 'tw_close', 'tw_unlock', 'tw_eat', 'tw_examine',
    })

    def __init__(
        self,
        game_path:  str           = '',
        world_type: str           = 'nano',  # 'nano' | 'micro'
        world_seed: Optional[int] = None,
        quest_type: str           = '',      # '' = random (micro only)
        n_rooms:    int           = 6,       # micro only
    ) -> None:
        """
        Args:
            game_path:  Path to a .ulx or .z8 TextWorld game file.
            world_type: 'nano' (default) or 'micro' (procedural generator).
            world_seed: Seed for MicroTextWorld (None = unseeded).
            quest_type: Fixed quest type for MicroTextWorld ('' = random).
            n_rooms:    Number of rooms for MicroTextWorld.
        """
        self._game_path        = game_path
        self._world_type       = world_type
        self._world_seed       = world_seed
        self._quest_type_micro = quest_type
        self._n_rooms          = n_rooms
        self._use_nano         = (not game_path) or (not _HAS_TW)
        self._env: Any    = None      # NanoTextEnv or gym.Env
        self._obs: str    = ''
        self._score: float = 0.0
        self._done: bool  = False
        self._info: dict  = {}
        self._prev_inventory: Set[str] = set()
        self._current_room: str = ''
        self._prev_room: str    = ''

        # Primitives dict (read by ProcessInterpreter via modality injection)
        self._primitives: Dict[str, Any] = {
            # Observation (safe)
            'tw_obs':        lambda: self._obs,
            'tw_score':      lambda: self._score,
            'tw_done':       lambda: self._done,
            'tw_inventory':  lambda: sorted(self._prev_inventory),
            'tw_location':   lambda: self._current_room,
            'tw_admissible': lambda: self._info.get('admissible_commands', []),
            # Action (effectful)
            'tw_go':         self._go,
            'tw_take':       self._take,
            'tw_drop':       self._drop,
            'tw_eat':        self._eat,
            'tw_examine':    self._examine,
            'tw_put':        self._put,
            'tw_unlock':     self._unlock,
            'tw_open':       lambda t: self._cmd(f'open {t}'),
            'tw_close':      lambda t: self._cmd(f'close {t}'),
        }

    # ------------------------------------------------------------------
    # Modality protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return 'textworld'

    @property
    def primitives(self) -> Dict[str, Any]:
        return self._primitives

    def connect(self) -> str:
        """Initialise the environment and return the first observation."""
        if self._world_type == 'micro' and not self._game_path:
            self._env      = MicroTextWorld(
                n_rooms    = self._n_rooms,
                quest_type = self._quest_type_micro,
                seed       = self._world_seed,
            )
            obs, info = self._env.reset()
        elif self._use_nano:
            self._env = NanoTextEnv()
            obs, info = self._env.reset()
        else:
            env_id = textworld.gym.register_game(
                self._game_path,
                max_episode_steps=200,
                request_infos=textworld.EnvInfos(
                    inventory=True,
                    description=True,
                    admissible_commands=True,
                    won=True,
                    lost=True,
                ),
            )
            self._env = gym.make(env_id)
            obs, info = self._env.reset()

        self._obs   = obs if isinstance(obs, str) else str(obs)
        self._score = 0.0
        self._done  = False
        self._info  = info if isinstance(info, dict) else {}
        self._prev_inventory = set(self._info.get('inventory', []))
        self._current_room = self._parse_room(self._obs)
        self._prev_room    = self._current_room
        return self._obs

    def get_obs(self) -> dict:
        return {
            'text':           self._obs,
            'score':          self._score,
            'done':           self._done,
            'inventory':      sorted(self._prev_inventory),
            'location':       self._current_room,
            'admissible':     self._info.get('admissible_commands', []),
            # Quest-specific fields (MicroTextWorld; empty strings/lists for NanoTextEnv)
            'quest_type':     self._info.get('quest_type', ''),
            'quest_item':     self._info.get('quest_item', ''),
            'quest_dest':     self._info.get('quest_dest', ''),
            'quest_container': self._info.get('quest_container', ''),
            'heat_src_here':  self._info.get('heat_src_here', []),
            'containers_here': self._info.get('containers_here', []),
        }

    def send_action(self, cmd: str) -> Tuple[str, float, bool]:
        """Execute one command. Returns (new_obs, reward, done)."""
        obs, reward, done, info = self._env.step(cmd)
        self._obs   = obs if isinstance(obs, str) else str(obs)
        info        = info if isinstance(info, dict) else {}
        # Prefer explicit score from info (NanoTextEnv + TextWorld both set it).
        if 'score' in info:
            self._score = float(info['score'])
        else:
            self._score += float(reward)
        self._done   = bool(done)
        self._info   = info
        new_room = self._parse_room(self._obs)
        self._prev_room    = self._current_room
        self._current_room = new_room
        return self._obs, float(reward), self._done

    def get_events(self) -> List[dict]:
        """Diff inventory and room to produce semantic events.

        Returns list of dicts with 'type' in:
          'acquired'  — item picked up or appeared in inventory
          'lost'      — item dropped or disappeared from inventory
          'moved'     — player changed rooms
          'scored'    — score increased this step
        """
        events: List[dict] = []

        new_inventory = set(self._info.get(
            'inventory',
            self._parse_inventory(self._obs),
        ))
        for item in new_inventory - self._prev_inventory:
            events.append({'type': 'acquired', 'item': item, 'room': self._current_room})
        for item in self._prev_inventory - new_inventory:
            events.append({'type': 'lost', 'item': item, 'room': self._current_room})
        self._prev_inventory = new_inventory

        if self._current_room and self._current_room != self._prev_room:
            events.append({
                'type': 'moved',
                'from': self._prev_room,
                'to':   self._current_room,
            })

        return events

    def current_priority(self, engine) -> Tuple[str, float, str]:
        if self._done:
            return ('RESET', 1.0, 'episode_over')
        if self._score > 0:
            return ('EXPLOIT', min(1.0, self._score / 3.0), 'score_positive')
        return ('EXPLORE', 0.4, 'novelty')

    def disconnect(self) -> None:
        if self._env is not None and not self._use_nano:
            try:
                self._env.close()
            except Exception:
                pass
        self._env = None

    # ------------------------------------------------------------------
    # Action helpers (effectful — blocked in dry_run)
    # ------------------------------------------------------------------

    def _cmd(self, command: str) -> str:
        obs, _, _ = self.send_action(command)
        return obs

    def _go(self, direction: str) -> str:
        return self._cmd(f'go {direction}')

    def _take(self, item: str) -> str:
        return self._cmd(f'take {item}')

    def _drop(self, item: str) -> str:
        return self._cmd(f'drop {item}')

    def _eat(self, item: str) -> str:
        return self._cmd(f'eat {item}')

    def _examine(self, thing: str) -> str:
        return self._cmd(f'examine {thing}')

    def _put(self, item: str, container: str) -> str:
        return self._cmd(f'put {item} in {container}')

    def _unlock(self, target: str, key: str) -> str:
        return self._cmd(f'unlock {target} with {key}')

    # ------------------------------------------------------------------
    # Text parsing helpers
    # ------------------------------------------------------------------

    def _parse_room(self, obs: str) -> str:
        """Extract room name from observation text (best-effort)."""
        # "You are in the kitchen." or "= Kitchen ="
        m = re.search(r'you are in (?:the )?([a-z ]+?)[\.\n]', obs, re.I)
        if m:
            return m.group(1).strip().lower()
        m = re.search(r'=+\s*([A-Za-z ]+?)\s*=+', obs)
        if m:
            return m.group(1).strip().lower()
        return self._current_room or 'unknown'

    def _parse_inventory(self, obs: str) -> Set[str]:
        """Fallback inventory parser when info dict lacks it."""
        m = re.search(r'you are carrying:\s*([^\n\.]+)', obs, re.I)
        if m:
            raw = m.group(1).strip()
            if raw.lower() in ('nothing', 'nothing.'):
                return set()
            items = re.split(r',\s*|\band\b', raw)
            return {i.strip().strip('.').lower() for i in items if i.strip()}
        return set()
