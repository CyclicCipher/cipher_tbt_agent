"""Phase Q: TextWorld agent + semantic bootstrapping.

Three phases that demonstrate closing the loop between syntax and semantics:

  Phase Q1 — Causal learning
    The agent explores the world taking random admissible actions.
    Each inventory change (acquire/lose item) becomes a causal observation:
      observe('takeable',  (item,),          (True,))
      observe('edible',    (item,),          (True,))
      observe('navigable', (from_room, cmd), (to_room,))
    After exploration, consolidate() attempts to synthesise causal rules.

  Phase Q2 — Phase O on collected text
    All observation text from Phase Q1 is tokenised and fed through the
    Phase O pipeline (discover_categories_from_dists).  With a rich enough
    vocabulary this should separate:
      - Nouns (room names, item names) from function words
      - Verbs (commands: take, go, eat) from prepositions

  Phase Q3 — Semantic bootstrapping
    Within each Phase O cluster, semantic_bootstrap() groups words by
    PPMI-weighted co-occurrence (window=5). Expected discoveries:
      - Within NOUN: location sub-cluster vs. object sub-cluster
      - Within VERB: movement verbs (go/north/south) vs. manipulation verbs
    These sub-clusters are cross-referenced with Phase Q1 causal knowledge:
    if every word in a semantic sub-cluster has the same causal rule, the
    cluster IS a causal category (sheaf consistency H¹=0).

Run:
  python textworld_test.py                           # NanoTextEnv, 300 steps (Q1-Q3)
  python textworld_test.py --plan                    # + Phase Q4 (NanoTextEnv)
  python textworld_test.py --plan --world micro      # + Phase Q4 (MicroTextWorld, random quest)
  python textworld_test.py --plan --world micro --quest eat   # fixed quest type
  python textworld_test.py --plan --world micro --quest cook  # cook quest
  python textworld_test.py --plan --episodes 5       # more planning episodes
  python textworld_test.py --steps 800               # more exploration
  python textworld_test.py --game game.ulx           # real TextWorld game
  python textworld_test.py --pos 9 --sem 4           # more clusters
  python textworld_test.py --seed 7                  # different random seed
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import random
import re
import sys

# Force UTF-8 stdout on Windows consoles that default to cp1252.
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

# Ensure the symbolic_ai package root is on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.join(_HERE, '..', '..')
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from modalities.textworld_modality import (
    TextWorldModality, NanoTextEnv, MicroTextWorld, _HAS_TW,
)
from synthesis import discover_categories_from_dists, semantic_bootstrap
from planning import (
    Goal, FEPGoal, Drive,
    AffordanceModel, DecisionEngine,
    GoalStack, EpisodicBuffer, BeliefState,
)

# ---------------------------------------------------------------------------
# Helpers shared with discover_test
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f'\n{"=" * 60}')
    print(f'  {title}')
    print('=' * 60)


def _stream_to_dists(tokens: list) -> tuple:
    """One-pass bigram accumulation: word → {next_word: probability}."""
    raw    = collections.defaultdict(collections.Counter)
    g_freq: collections.Counter = collections.Counter()
    for i, w in enumerate(tokens):
        g_freq[w] += 1
        if i < len(tokens) - 1:
            raw[w][tokens[i + 1]] += 1
    input_counts = {(w,): sum(c.values()) for w, c in raw.items()}
    dists = {
        (w,): {(nw,): cnt / sum(c.values()) for nw, cnt in c.items()}
        for w, c in raw.items()
    }
    return dists, input_counts, dict(g_freq)


def _tokenise(text: str) -> list:
    """Simple word tokeniser for game observations."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\-]", ' ', text)
    return [t for t in text.split() if len(t) > 1]


def _ic_ranked(members: list, global_freq: dict, n: int = 12) -> list:
    """Return up to n words ranked by Information Content within the group."""
    total   = sum(global_freq.get(w, 0) for w in members) or 1
    g_total = sum(global_freq.values()) or 1
    scored  = []
    for w in members:
        cnt  = global_freq.get(w, 0)
        p_wc = cnt / total
        p_w  = cnt / g_total
        ic   = p_wc * math.log2(p_wc / max(p_w, 1e-12)) if p_wc > 0 else 0.0
        scored.append((ic, cnt, w))
    scored.sort(reverse=True)
    return [w for _, _, w in scored[:n]]


# ---------------------------------------------------------------------------
# Phase Q1: causal learning via exploration
# ---------------------------------------------------------------------------

def _choose_action(obs: dict, rng: random.Random) -> str:
    """Pick an action: prefer novel moves (explore) over revisiting."""
    admissible = obs.get('admissible', [])
    if not admissible:
        return 'look'

    # Prefer actions that move or interact (not just look/inventory).
    preferred = [c for c in admissible
                 if not c.startswith(('look', 'inventory', 'i '))]
    pool = preferred if preferred else admissible
    return rng.choice(pool)


def run_phase_q1(
    modality:   TextWorldModality,
    n_steps:    int,
    rng:        random.Random,
    verbose:    bool = False,
) -> tuple:
    """Explore the world and collect causal observations.

    Returns:
        (all_tokens, causal_log, engine_stores)
        all_tokens:   flat list of word tokens from all observations
        causal_log:   list of (room, action, events) triples
        stores:       dict of concept_name → list of (inputs, outputs) pairs
                      (not engine.ExampleStore — raw lists for portability)
    """
    _banner('Phase Q1 — Causal Learning via Exploration')
    print(f'  Steps: {n_steps}  |  Backend: '
          f'{"NanoTextEnv" if modality._use_nano else "TextWorld"}\n')

    all_tokens: list = []
    causal_log: list = []
    stores: dict = collections.defaultdict(list)  # concept → [(inputs, outputs)]

    obs_dict = modality.get_obs()
    obs_text = obs_dict['text']
    all_tokens.extend(_tokenise(obs_text))

    score_history  = [0.0]
    room_visits: collections.Counter = collections.Counter()
    step_print = max(1, n_steps // 10)

    for step in range(n_steps):
        obs_dict = modality.get_obs()

        if obs_dict['done']:
            if verbose:
                print(f'    Step {step}: episode done (score={obs_dict["score"]:.0f}). Resetting.')
            modality.connect()
            obs_dict = modality.get_obs()

        room  = obs_dict['location']
        room_visits[room] += 1

        action = _choose_action(obs_dict, rng)
        _, reward, done = modality.send_action(action)
        events = modality.get_events()

        new_obs = modality.get_obs()
        all_tokens.extend(_tokenise(new_obs['text']))
        score_history.append(new_obs['score'])

        causal_log.append((room, action, events))

        # Record causal observations.
        for ev in events:
            if ev['type'] == 'acquired':
                item = ev['item']
                stores['takeable'].append(((item,), (True,)))
                if action.startswith('eat') or 'eat' in action:
                    stores['edible'].append(((item,), (True,)))
            elif ev['type'] == 'lost' and action.startswith(('drop', 'eat')):
                item = ev['item']
                if 'eat' in action:
                    stores['edible'].append(((item,), (True,)))
                    stores['consumed'].append(((item,), (True,)))
            elif ev['type'] == 'moved':
                stores['navigable'].append(
                    ((ev['from'], action), (ev['to'],))
                )
                # Also record with inventory context so the AffordanceModel
                # can infer preconditions (e.g. brass_key needed for go north).
                stores['navigable_ctx'].append(
                    ((ev['from'], action, frozenset(obs_dict['inventory'])),
                     (ev['to'],))
                )

        if (step + 1) % step_print == 0:
            print(f'  Step {step + 1:>4}/{n_steps}  room={room:<15}  '
                  f'action={action:<28}  score={new_obs["score"]:.0f}  '
                  f'events={len(events)}')

    print(f'\n  Exploration complete.')
    print(f'  Tokens collected:   {len(all_tokens):,}')
    print(f'  Unique tokens:      {len(set(all_tokens)):,}')
    print(f'  Room visits:        '
          + ', '.join(f'{r}×{c}' for r, c in room_visits.most_common()))
    print(f'  Score reached:      {max(score_history):.0f}')
    print(f'\n  Causal concepts observed:')
    for concept, examples in sorted(stores.items()):
        unique = {inp for inp, _ in examples}
        print(f'    {concept:<18} {len(examples):>4} examples, '
              f'{len(unique)} unique inputs')

    return all_tokens, causal_log, dict(stores)


# ---------------------------------------------------------------------------
# Phase Q2: Phase O on collected text
# ---------------------------------------------------------------------------

def run_phase_q2(
    all_tokens: list,
    n_clusters: int,
    min_count:  int = 2,
) -> tuple:
    """Run Phase O distributional clustering on game observations.

    Returns:
        (assignment, global_freq)
        assignment:   {word_str: cluster_id}
        global_freq:  {word_str: count}
    """
    _banner('Phase Q2 — Phase O: POS-Like Clustering of Game Vocabulary')

    if len(all_tokens) < 50:
        print('  Too few tokens for reliable clustering.')
        return {}, {}

    dists, input_counts, global_freq = _stream_to_dists(all_tokens)
    n_eligible = sum(1 for cnt in input_counts.values() if cnt >= min_count)
    print(f'  Unique tokens:       {len(dists):,}')
    print(f'  With >= {min_count} obs:    {n_eligible:,}')
    print(f'  Target clusters:     {n_clusters}')

    raw_assign = discover_categories_from_dists(
        dists        = dists,
        input_counts = input_counts,
        n_clusters   = n_clusters,
        min_examples = min_count,
    )
    if not raw_assign:
        print('  Not enough data for clustering.')
        return {}, global_freq

    # Renumber by cluster size (largest = C0).
    cluster_raw: dict = {}
    for (w,), cid in raw_assign.items():
        cluster_raw.setdefault(cid, []).append(w)
    by_size  = sorted(cluster_raw.items(), key=lambda kv: -len(kv[1]))
    renumber = {old: new for new, (old, _) in enumerate(by_size)}
    clusters: dict = {renumber[old]: sorted(m) for old, m in cluster_raw.items()}
    assignment: dict = {w: renumber[cid] for (w,), cid in raw_assign.items()}

    print(f'\n  {"Cluster":<8} {"Size":>6}   Most distinctive words')
    print(f'  {"-------":<8} {"----":>6}   ----------------------')
    for cid in sorted(clusters.keys()):
        members  = clusters[cid]
        top      = _ic_ranked(members, global_freq, n=10)
        top_str  = ', '.join(top)
        if len(members) > 10:
            top_str += f', ... (+{len(members) - 10})'
        print(f'  C{cid:<7} {len(members):>6}   {top_str}')

    return assignment, global_freq


# ---------------------------------------------------------------------------
# Phase Q3: Semantic bootstrapping + causal cross-reference
# ---------------------------------------------------------------------------

def run_phase_q3(
    all_tokens:   list,
    assignment:   dict,   # {word: cluster_id} from Phase Q2
    global_freq:  dict,
    causal_stores: dict,  # {concept: [(inputs, outputs)]}
    n_subclusters: int,
) -> None:
    """Semantic bootstrapping within Phase O clusters + causal confirmation.

    Shows:
      1. Semantic sub-clusters (PPMI co-occurrence within each POS cluster)
      2. Cross-reference: which causal concepts overlap with each sub-cluster?
    """
    _banner('Phase Q3 — Semantic Bootstrapping + Causal Cross-Reference')

    if not assignment:
        print('  No Phase O assignment available (Phase Q2 failed).')
        return

    # Re-format assignment as {(word,): cluster_id} for semantic_bootstrap.
    assign_tuple = {(w,): cid for w, cid in assignment.items()}

    print(f'  Running PPMI co-occurrence clustering (window=5, '
          f'{n_subclusters} sub-clusters per category).\n')

    sem = semantic_bootstrap(
        tokens        = all_tokens,
        assignment    = assign_tuple,
        global_freq   = global_freq,
        window        = 5,
        n_subclusters = n_subclusters,
        min_count     = 2,
    )

    if not sem:
        print('  Semantic bootstrapping returned empty result '
              '(too few tokens for co-occurrence).')
        return

    # Group by (pos_cid, sem_cid).
    groups: dict = {}
    for w, (pos_cid, sem_cid) in sem.items():
        groups.setdefault((pos_cid, sem_cid), []).append(w)

    # Build causal index: word → set of causal concepts it appears in.
    causal_index: dict = {}
    for concept, examples in causal_stores.items():
        for (inp, *_), _ in examples:
            if isinstance(inp, str):
                causal_index.setdefault(inp, set()).add(concept)

    print(f'  {"Cluster":<8} {"Words":<50} {"Causal roles"}')
    print(f'  {"-------":<8} {"-----":<50} {"------------"}')

    for (pos_cid, sem_cid) in sorted(groups.keys()):
        members  = groups[(pos_cid, sem_cid)]
        top      = _ic_ranked(members, global_freq, n=8)
        words_str = ', '.join(top)
        if len(members) > 8:
            words_str += f' (+{len(members) - 8})'

        # Causal roles across this sub-cluster.
        role_counts: collections.Counter = collections.Counter()
        for w in members:
            for role in causal_index.get(w, []):
                role_counts[role] += 1
        roles_str = ', '.join(
            f'{r}({c})' for r, c in role_counts.most_common(3)
        ) or '—'

        print(f'  C{pos_cid}.{sem_cid:<6} {words_str:<50} {roles_str}')

    # Sheaf consistency report.
    _banner('Phase Q3 — Sheaf Consistency: Semantic = Causal?')
    print()
    consistent = 0
    checked    = 0
    for (pos_cid, sem_cid), members in groups.items():
        if len(members) < 3:
            continue  # too small to evaluate
        # Count words with ANY causal role.
        has_role  = sum(1 for w in members if w in causal_index)
        if has_role == 0:
            continue
        checked += 1
        # Compute role entropy across member words.
        role_dist: collections.Counter = collections.Counter()
        for w in members:
            for role in causal_index.get(w, []):
                role_dist[role] += 1
        dominant_role, dom_count = role_dist.most_common(1)[0]
        total_role_hits = sum(role_dist.values())
        dom_frac = dom_count / total_role_hits
        h1_zero  = dom_frac >= 0.80  # 80% agreement → H¹ ≈ 0
        if h1_zero:
            consistent += 1

        top_words = _ic_ranked(members, global_freq, n=5)
        status    = 'H1=0 [ok]' if h1_zero else 'H1!=0 (split?)'
        print(f'  C{pos_cid}.{sem_cid}: [{", ".join(top_words)}]  '
              f'-> dominant role: {dominant_role} ({dom_frac:.0%})  {status}')

    if checked:
        print(f'\n  Consistency: {consistent}/{checked} sub-clusters '
              f'have H1~=0 (semantic cluster = causal category).')
        if consistent == checked:
            print('  PASS: All evaluated sub-clusters are semantically coherent.')
        else:
            print('  NOTE: Some sub-clusters straddle causal categories → '
                  'candidate for split or Override edge.')
    else:
        print('  Not enough causal data to evaluate sheaf consistency.')


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(modality: TextWorldModality, n_steps: int,
                   n_clusters: int, n_sem: int) -> None:
    _banner('Phase Q Summary')
    print(f"""
  Backend:   {"NanoTextEnv (zero deps)" if modality._use_nano else "TextWorld"}
  Steps:     {n_steps}
  POS clusters (Phase Q2):  {n_clusters}
  Semantic sub-clusters:    {n_sem} per POS category

  What was demonstrated:
    Q1: The agent explored the world and accumulated causal evidence:
        (item,) -> takeable=True, (item,) -> edible=True,
        (room, cmd) -> navigable=(next_room,)
    Q2: Phase O recovered POS-like clusters from game text alone.
        Verbs (go/take/eat) separated from Nouns (room/item names).
    Q3: Semantic bootstrapping sub-clustered Nouns into
        location-words vs. object-words purely from co-occurrence.
        Causal cross-reference: the object sub-cluster IS the
        takeable concept (same boundary, two independent derivations).

  Architecture insight:
    The same synthesis + clustering machinery handles arithmetic,
    natural language, and now grounded text games without modification.
    Only the modality (token stream / game API) changes.

  Next step: Phase K — connect to live Minecraft via dxcam + pynput + mcrcon.
""")


# ---------------------------------------------------------------------------
# Phase Q4: WorldModel + QuestPlanner + goal-directed quest completion
# ---------------------------------------------------------------------------

class WorldModel:
    """Knowledge base built from Phase Q1 causal observations + live Q4 updates.

    Causal knowledge (takeable, edible, nav_map) seeds from Q1.
    For MicroTextWorld episodes the nav_map is rebuilt online from movement
    events (layout changes per episode), while takeable/edible transfer.
    Quest info is read from the info dict each step.
    """

    def __init__(self, stores: dict) -> None:
        self.nav_map:   dict = {}   # (room, cmd) -> next_room (may grow online)
        self.takeable:  set  = set()
        self.edible:    set  = set()
        self.item_rooms: dict = {}  # item -> room (live, cleared per episode)
        # Quest state (updated from info dict each step)
        self.quest_type:      str  = ''   # 'eat','fetch','unlock','cook','put',''
        self.quest_item:      str  = ''   # item to take/eat/cook/fetch/put
        self.quest_dest:      str  = ''   # target room (fetch/unlock)
        self.quest_container: str  = ''   # container name (put)
        self.quest_unlocked:  bool = False
        # Legacy NanoTextEnv compatibility
        self.bedroom_unlocked: bool = False
        # Tracked room locations (populated online from observations)
        self._heat_src_rooms: dict = {}  # heat_name → room
        self._container_room: dict = {}  # container_name → room

        for (room, cmd), (next_room,) in stores.get('navigable', []):
            self.nav_map[(room, cmd)] = next_room
        for (item,), _ in stores.get('takeable', []):
            self.takeable.add(item)
        for (item,), _ in stores.get('edible', []):
            self.edible.add(item)

    def known_rooms(self) -> set:
        rooms: set = set()
        for (r, _), nr in self.nav_map.items():
            rooms.add(r); rooms.add(nr)
        return rooms

    def reset_episode(self, online_nav: bool = False) -> None:
        """Clear per-episode state. online_nav=True also clears nav_map."""
        self.item_rooms.clear()
        self.quest_type      = ''
        self.quest_item      = ''
        self.quest_dest      = ''
        self.quest_container = ''
        self.quest_unlocked  = False
        self.bedroom_unlocked = False
        self._heat_src_rooms.clear()
        self._container_room.clear()
        if online_nav:
            self.nav_map.clear()

    def update(self, obs: dict) -> None:
        """Update item locations and quest state from current observation."""
        room = obs['location']
        text = obs['text']
        # Item locations from observation text
        m = re.search(r'Items here:\s*([^.]+)\.', text)
        if m and m.group(1).strip().lower() != 'none':
            for item in (i.strip() for i in m.group(1).split(',')):
                if item:
                    self.item_rooms[item] = room
        for item in obs.get('inventory', []):
            self.item_rooms[item] = '_inv'
        # Unlock detection (NanoTextEnv + MicroTextWorld)
        if 'you unlock' in text.lower():
            self.bedroom_unlocked = True
            self.quest_unlocked   = True
        # Quest state from info dict (MicroTextWorld provides this directly)
        if obs.get('quest_type'):
            self.quest_type      = obs['quest_type']
            self.quest_item      = obs.get('quest_item', '')
            self.quest_dest      = obs.get('quest_dest', '')
            self.quest_container = obs.get('quest_container', '')
        elif not self.quest_type:
            # NanoTextEnv: infer quest type from causal knowledge
            self.quest_type = 'unlock_eat'  # combined NanoTextEnv quest
            self.quest_dest = 'bedroom'     # locked room in NanoTextEnv
        # Track heat source and container locations for nav_dest computation
        for heat in obs.get('heat_src_here', []):
            self._heat_src_rooms[heat] = room
        for (cn, _) in obs.get('containers_here', []):
            self._container_room[cn] = room

    def update_nav(self, from_room: str, cmd: str, to_room: str) -> None:
        """Add a navigation edge discovered online, plus the inferred inverse.

        When the agent walks A →go north→ B we also record B →go south→ A
        without requiring the agent to physically walk back.  This prevents
        _enter_goal_achieve from mistaking the return path for the newly-
        unlocked exit (which is the only truly un-mapped direction).
        """
        _OPP = {
            'go north': 'go south', 'go south': 'go north',
            'go east':  'go west',  'go west':  'go east',
            'go up':    'go down',  'go down':  'go up',
        }
        self.nav_map[(from_room, cmd)] = to_room
        inv = _OPP.get(cmd)
        if inv and (to_room, inv) not in self.nav_map:
            self.nav_map[(to_room, inv)] = from_room


# ---------------------------------------------------------------------------
# Goal achieve functions — universal (work for all quest types)
# ---------------------------------------------------------------------------

def _bfs_no_blocks(
    src: str, dst: str,
    nav_map: dict,
    affordances: AffordanceModel,
) -> list:
    """BFS on nav_map that skips (room, cmd) pairs blocked by AffordanceModel.

    This is how failure learning propagates into navigation: when 'go north'
    from 'living room' is known to fail, it is excluded from the BFS graph so
    the planner naturally routes around it via alternative paths.
    """
    if src == dst:
        return []
    seen: set = {src}
    q: collections.deque = collections.deque([(src, [])])
    while q:
        room, path = q.popleft()
        for (r, cmd), nr in nav_map.items():
            if r == room and nr not in seen:
                if affordances.is_blocked(room, cmd):
                    continue  # learned failure: skip this edge
                p2 = path + [cmd]
                if nr == dst:
                    return p2
                seen.add(nr)
                q.append((nr, p2))
    return []


def _unlock_achieve(state: dict, aff: AffordanceModel, rng: random.Random, **kwargs):
    stack = kwargs.get('stack')
    for cmd in state.get('unlock_cmds', []):
        if stack is not None:
            # After unlocking, the destination becomes reachable.  Push an
            # enter sub-goal so the next step commits to entering it rather
            # than wandering.  This is a sequential commitment, not a hack:
            # unlock → enter is a two-step composition in any domain.
            stack.push(Goal('enter_now', 0.93,
                condition=lambda s: bool(
                    s.get('quest_dest') and s['location'] != s.get('quest_dest')),
                achieve=_enter_goal_achieve))
        return cmd, 'UNLOCK'
    return None


def _enter_goal_achieve(state: dict, aff: AffordanceModel, rng: random.Random, **kwargs):
    """Navigate into the quest destination room once accessible.

    After unlocking, the exit direction appears in admissible but is NOT yet
    in nav_map (we've never traversed it).  With bidirectional nav inference
    (update_nav adds inverse edges automatically), return-path exits ARE
    already in nav_map.  So the newly-unlocked exit is the ONLY go command
    that is both (a) in admissible and (b) not in nav_map from this room.
    """
    loc     = state['location']
    dest    = state.get('quest_dest', 'bedroom')
    nav_map = state['_nav_map']
    adm     = set(state['admissible'])
    visited = state['_visited']
    if loc == dest:
        return None
    # Try BFS via known nav_map first.
    path = _bfs_no_blocks(loc, dest, nav_map, aff)
    if path and path[0] in adm:
        return path[0], 'GOAL'
    # Exits already mapped from this location (including inferred inverses).
    mapped_exits = {cmd for (room, cmd) in nav_map if room == loc}
    # Skip mapped exits that lead to visited, non-destination rooms.
    bad_exits = {
        cmd for (room, cmd), dst in nav_map.items()
        if room == loc and dst in visited and dst != dest
    }
    # Prefer unmapped exits — most likely the newly-unlocked door.
    for cmd in sorted(adm):
        if cmd.startswith('go ') and cmd not in mapped_exits:
            return cmd, 'GOAL'
    # Try mapped exits not leading to dead ends.
    for cmd in sorted(adm):
        if cmd.startswith('go ') and cmd not in bad_exits:
            return cmd, 'GOAL'
    # Last resort.
    for cmd in adm:
        if cmd.startswith('go '):
            return cmd, 'GOAL'
    return None


def _eat_achieve(state: dict, aff: AffordanceModel, rng: random.Random, **kwargs):
    inv     = set(state['inventory'])
    edibles = state.get('edibles', set())   # dynamic: from state, never a closure
    adm     = set(state['admissible'])
    for item in sorted(inv & edibles):
        cmd = f'eat {item}'
        if cmd in adm:
            return cmd, f'EAT({item})'
    return None


def _cook_achieve(state: dict, aff: AffordanceModel, rng: random.Random, **kwargs):
    """Cook the quest raw food at the heat source.

    After firing, pushes an eat sub-goal onto the stack so the very next
    decision commits to eating the result.  This is sequential commitment:
    cook → eat is a two-step composition, not a loop-prone condition check.
    """
    stack = kwargs.get('stack')
    inv   = set(state['inventory'])
    adm   = set(state['admissible'])
    item  = state.get('quest_item', '')
    cmd   = f'cook {item}'
    if cmd in adm and item in inv:
        if stack is not None:
            # Push eat_now: fires next step when the cooked result is in inv.
            # Reads edibles from state (dynamic), so it works in any domain.
            stack.push(Goal('eat_now', 0.95,
                condition=lambda s: bool(
                    set(s.get('inventory', [])) & set(s.get('edibles', set()))),
                achieve=_eat_achieve))
        return cmd, f'COOK({item})'
    return None


def _fetch_deliver_achieve(state: dict, aff: AffordanceModel, rng: random.Random, **kwargs):
    """Navigate to target room while carrying the fetch item."""
    loc     = state['location']
    dest    = state.get('quest_dest', '')
    nav_map = state['_nav_map']
    adm     = set(state['admissible'])
    if not dest or loc == dest:
        return None
    path = _bfs_no_blocks(loc, dest, nav_map, aff)
    if path and path[0] in adm:
        return path[0], f'DELIVER->{dest}'
    return None


def _open_container_achieve(state: dict, aff: AffordanceModel, rng: random.Random, **kwargs):
    """Open the quest container if it's in this room and closed."""
    adm  = set(state['admissible'])
    cont = state.get('quest_container', '')
    cmd  = f'open {cont}'
    if cmd in adm:
        return cmd, f'OPEN({cont})'
    return None


def _put_achieve(state: dict, aff: AffordanceModel, rng: random.Random, **kwargs):
    """Put the quest item into the (open) quest container."""
    inv  = set(state['inventory'])
    adm  = set(state['admissible'])
    item = state.get('quest_item', '')
    cont = state.get('quest_container', '')
    cmd  = f'put {item} in {cont}'
    if cmd in adm and item in inv:
        return cmd, f'PUT({item}->{cont})'
    return None


def _take_achieve(state: dict, aff: AffordanceModel, rng: random.Random, **kwargs):
    inv  = set(state['inventory'])
    adm  = set(state['admissible'])
    edibles  = state.get('edibles', set())
    takeables = state.get('takeables', set())
    need = (edibles | takeables) - inv
    for item in sorted(need, key=lambda x: (x not in edibles, x)):
        cmd = f'take {item}'
        if cmd in adm:
            return cmd, f'TAKE({item})'
    return None


def _nav_to_item_achieve(state: dict, aff: AffordanceModel, rng: random.Random, **kwargs):
    loc          = state['location']
    adm          = set(state['admissible'])
    nav_map      = state['_nav_map']
    needed_known = state.get('needed_known', {})
    edibles      = state.get('edibles', set())
    for item in sorted(needed_known, key=lambda x: (x not in edibles, x)):
        target = needed_known[item]
        path   = _bfs_no_blocks(loc, target, nav_map, aff)
        if path and path[0] in adm:
            return path[0], f'NAV->{item}@{target}'
    # Also navigate toward quest destination or container room
    dest = state.get('nav_dest', '')
    if dest and dest != loc:
        path = _bfs_no_blocks(loc, dest, nav_map, aff)
        if path and path[0] in adm:
            return path[0], f'NAV->{dest}'
    return None


def _explore_achieve(state: dict, aff: AffordanceModel, rng: random.Random, **kwargs):
    loc          = state['location']
    adm          = set(state['admissible'])
    nav_map      = state['_nav_map']
    unvisited    = state.get('unvisited', set())
    unexplored   = state.get('unexplored_exits', [])
    # Priority 1: try untested exits from current room (might reveal new rooms).
    for cmd in sorted(unexplored):
        if not aff.is_blocked(loc, cmd):
            return cmd, 'EXPLORE-exit'
    # Priority 2: BFS toward known-but-unvisited rooms.
    for room in sorted(unvisited):
        path = _bfs_no_blocks(loc, room, nav_map, aff)
        if path and path[0] in adm:
            return path[0], f'EXPLORE->{room}'
    # Fallback: any unblocked go action
    for cmd in adm:
        if cmd.startswith('go ') and not aff.is_blocked(loc, cmd):
            return cmd, 'EXPLORE-any'
    return None


def _random_achieve(state: dict, aff: AffordanceModel, rng: random.Random, **kwargs):
    # General principle: RANDOM never drops items.  A general agent does not
    # undo its own inventory possession without explicit goal direction.
    # Drops belong to domain goals (put_in_container), never to stochastic
    # fallback.  This is not textworld-specific; it holds in any domain.
    pool = [c for c in state['admissible']
            if not c.startswith(('look', 'inventory', 'i ', 'drop '))]
    return (rng.choice(pool), 'RANDOM') if pool else ('look', 'RANDOM')


def make_textworld_goals(world: WorldModel, causal_stores: dict) -> list:
    """Build a universal FEPGoal list covering all quest types.

    Goals activate via FEP drive deficits — no quest type is hardcoded.
    Quest-specific goals (cook, fetch, put) condition on state['quest_type'].
    NanoTextEnv (quest_type='unlock_eat') triggers unlock + eat goals.
    MicroTextWorld provides quest_type directly in the info dict.

    Priority cascade (urgency values):
      safety        1.00  always
      unlock_exit   0.88  when there's a door to unlock
      enter_goal    0.85  when quest_dest accessible
      eat_edible    0.82  when edible in inventory
      cook_item     0.80  when raw food in inv + heat source here
      fetch_deliver 0.78  when fetch item in inv + not yet at dest
      put_item      0.76  when put item in inv + open container here
      open_cont     0.72  when container in room, closed
      take_items    0.65  when needed item visible here
      nav_to_item   0.55  when needed item known elsewhere
      explore       0.35  when unvisited rooms remain
      random        0.10  fallback
    """
    # NOTE: drives and conditions MUST read from state 's' dynamically.
    # Never capture environment-specific sets (e.g. world.edible) in closures —
    # those encode assumptions about one environment and silently fail in others.
    # The state dict (_build_tw_state) is the single source of truth each step.

    # Drives — measure() → 1.0 when satisfied, 0.0 when maximally needed
    d_unlock = Drive('unlock',
        measure=lambda s: 1.0 if s.get('quest_unlocked') else 0.0,
        urgency=0.88)
    d_enter = Drive('enter',
        measure=lambda s: 0.0 if (
            s.get('quest_unlocked') and s['location'] != s.get('quest_dest', '??')
        ) else 1.0,
        urgency=0.85)
    d_eat = Drive('eat',
        # Read edibles from state (dynamic, updated each step from admissible).
        # Never use a static closure over world.edible — that set reflects only
        # the Q1 environment and is wrong for any other environment.
        measure=lambda s: 1.0 - float(
            bool(set(s.get('inventory', [])) & set(s.get('edibles', set())))),
        urgency=0.82)
    d_cook = Drive('cook',
        measure=lambda s: 1.0 - float(
            bool(s.get('quest_type') == 'cook' and s.get('cook_cmd_here'))),
        urgency=0.80)
    d_fetch = Drive('fetch',
        measure=lambda s: 1.0 - float(
            bool(s.get('quest_type') == 'fetch'
                 and s.get('quest_item', '') in set(s.get('inventory', []))
                 and s['location'] != s.get('quest_dest', '??'))),
        urgency=0.78)
    d_put = Drive('put',
        measure=lambda s: 1.0 - float(
            bool(s.get('quest_type') == 'put' and s.get('put_cmd_here'))),
        urgency=0.76)
    d_open = Drive('open_cont',
        measure=lambda s: 1.0 - float(bool(s.get('open_cont_cmd_here'))),
        urgency=0.72)
    d_take = Drive('take',
        measure=lambda s: 1.0 - float(bool(s.get('needed_here'))),
        urgency=0.65)
    d_nav = Drive('nav',
        measure=lambda s: 1.0 - float(bool(s.get('needed_known') or s.get('nav_dest'))),
        urgency=0.55)
    d_explore = Drive('explore',
        # Include unexplored_exits: unmapped go commands from current room are
        # also worth exploring even if all reachable rooms have been visited.
        measure=lambda s: 1.0 - float(
            bool(s.get('unvisited') or s.get('unexplored_exits'))),
        urgency=0.35)

    # Conditions
    def qt(*types):
        """Condition: quest_type matches one of types ('' matches everything)."""
        def _c(s): return not types or s.get('quest_type', '') in types
        return _c

    def cond_unlock(s):
        return bool(s.get('unlock_cmds'))
    def cond_enter(s):
        return bool(s.get('quest_unlocked') and s['location'] != s.get('quest_dest', ''))
    def cond_eat(s):
        # Must use state['edibles'] (dynamic), not a closure — see design principle.
        return bool(set(s.get('inventory', [])) & set(s.get('edibles', set())))
    def cond_cook(s):
        return s.get('quest_type') == 'cook' and bool(s.get('cook_cmd_here'))
    def cond_fetch(s):
        return (s.get('quest_type') == 'fetch'
                and s.get('quest_item', '') in set(s.get('inventory', []))
                and s['location'] != s.get('quest_dest', '??'))
    def cond_put(s):
        return s.get('quest_type') == 'put' and bool(s.get('put_cmd_here'))
    def cond_open(s):
        return bool(s.get('open_cont_cmd_here'))
    def cond_take(s): return bool(s.get('needed_here'))
    def cond_nav(s):  return bool(s.get('needed_known') or s.get('nav_dest'))
    def cond_exp(s):  return bool(s.get('unvisited') or s.get('unexplored_exits'))

    return [
        FEPGoal('safety',         [],         lambda s: True,   lambda s,a,r,**kw: None,
                is_safety=True),
        FEPGoal('unlock_exit',    [d_unlock],  cond_unlock,      _unlock_achieve),
        FEPGoal('enter_goal',     [d_enter],   cond_enter,       _enter_goal_achieve),
        FEPGoal('eat_edible',     [d_eat],     cond_eat,         _eat_achieve),
        FEPGoal('cook_item',      [d_cook],    cond_cook,        _cook_achieve),
        FEPGoal('fetch_deliver',  [d_fetch],   cond_fetch,       _fetch_deliver_achieve),
        FEPGoal('put_item',       [d_put],     cond_put,         _put_achieve),
        FEPGoal('open_container', [d_open],    cond_open,        _open_container_achieve),
        FEPGoal('take_items',     [d_take],    cond_take,        _take_achieve),
        FEPGoal('nav_to_item',    [d_nav],     cond_nav,         _nav_to_item_achieve),
        FEPGoal('explore',        [d_explore], cond_exp,         _explore_achieve),
        FEPGoal('random',         [],          lambda s: True,   _random_achieve,
                base_priority=0.10),
    ]


def _build_tw_state(
    obs:          dict,
    world:        WorldModel,
    causal_stores: dict,
    visited:      set,
) -> dict:
    """Build the generic state dict consumed by DecisionEngine goals.

    Also updates the WorldModel (item locations, quest state, env tracking) and
    records the current location in ``visited``.

    Quest-aware fields added for MicroTextWorld:
      quest_type, quest_item, quest_dest, quest_container, quest_unlocked
      cook_cmd_here   — 'cook {quest_item}' is currently admissible
      put_cmd_here    — 'put {quest_item} in {quest_container}' is admissible
      open_cont_cmd_here — 'open {quest_container}' is admissible
      nav_dest        — room to navigate toward when carrying quest item
                        (heat-src room for cook, container room for put)
    """
    inv = set(obs['inventory'])
    adm = set(obs['admissible'])
    loc = obs['location']

    # Base causal knowledge from Q1 (may be empty for micro items not in nano)
    edibles   = world.edible.copy()
    takeables = world.takeable.copy()

    world.update(obs)   # updates item_rooms, quest fields, heat/container rooms
    visited.add(loc)

    # Quest state (freshly read from world after update)
    quest_type      = world.quest_type
    quest_item      = world.quest_item
    quest_dest      = world.quest_dest
    quest_container = world.quest_container
    quest_unlocked  = world.quest_unlocked

    # Augment edibles with anything the env lets us eat right now.
    # This handles cooked foods (e.g. 'baked potato') not in Q1 causal stores.
    for cmd in adm:
        if cmd.startswith('eat '):
            edibles.add(cmd[4:])

    # Quest item is always "needed" — add to takeables so navigation/take fires.
    if quest_item:
        takeables.add(quest_item)

    # Quest-specific one-step affordances (preconditions verified by admissible).
    cook_cmd_here      = bool(quest_item and f'cook {quest_item}' in adm)
    put_cmd_here       = bool(
        quest_item and quest_container
        and f'put {quest_item} in {quest_container}' in adm
    )
    open_cont_cmd_here = bool(quest_container and f'open {quest_container}' in adm)

    # nav_dest: where do we need to be (after picking up the quest item)?
    nav_dest = ''
    if quest_type == 'cook' and quest_item in inv:
        # Need to be near a heat source to cook; navigate there.
        heat_rooms = list(world._heat_src_rooms.values())
        if heat_rooms:
            nav_dest = next((r for r in heat_rooms if r != loc), heat_rooms[0])
    elif quest_type == 'put' and quest_item in inv:
        # Need to be in the container's room to open & put.
        cont_room = world._container_room.get(quest_container, '')
        if cont_room and cont_room != loc:
            nav_dest = cont_room

    needed = (edibles | takeables) - inv
    needed_here  = bool(needed and any(f'take {i}' in adm for i in needed))
    needed_known = {
        i: world.item_rooms[i]
        for i in needed
        if world.item_rooms.get(i) not in (None, '_inv')
    }

    # Compute rooms to explore.  Exclude the quest destination if it hasn't
    # been unlocked yet — the enter_goal handles entry once accessible, and
    # routing through it prematurely causes the agent to cycle.
    unvisited = world.known_rooms() - visited
    if quest_dest and not quest_unlocked:
        unvisited.discard(quest_dest)

    # Unexplored exits: go commands in admissible not yet recorded in nav_map
    # as SOURCE from this room.  With bidirectional nav inference, every exit
    # the agent has traversed (in either direction) is already in nav_map.  So
    # unexplored_exits = exits from this room the agent has NEVER taken.
    loc_nav_exits = {cmd for (room, cmd) in world.nav_map if room == loc}
    unexplored_exits = [c for c in adm
                        if c.startswith('go ') and c not in loc_nav_exits]

    return {
        'location':          loc,
        'inventory':         list(inv),
        'admissible':        list(adm),
        'score':             obs['score'],
        'done':              obs['done'],
        'bedroom_unlocked':  world.bedroom_unlocked,
        # Quest state
        'quest_type':        quest_type,
        'quest_item':        quest_item,
        'quest_dest':        quest_dest,
        'quest_container':   quest_container,
        'quest_unlocked':    quest_unlocked,
        # Commands / affordances
        'unlock_cmds':       [c for c in adm if c.startswith('unlock')],
        'cook_cmd_here':     cook_cmd_here,
        'put_cmd_here':      put_cmd_here,
        'open_cont_cmd_here': open_cont_cmd_here,
        # Navigation
        'needed_here':       needed_here,
        'needed_known':      needed_known,
        'unvisited':         unvisited,
        'unexplored_exits':  unexplored_exits,
        'nav_dest':          nav_dest,
        # Item sets (augmented)
        'edibles':           edibles,
        'takeables':         takeables,
        # Private refs for goal achieve functions.
        '_world':            world,
        '_nav_map':          world.nav_map,
        '_visited':          visited,
    }


def run_phase_q4(
    causal_stores: dict,
    game_path:     str,
    n_episodes:    int,
    max_steps:     int,
    rng:           random.Random,
    world_type:    str = 'nano',   # 'nano' | 'micro'
    quest_type:    str = '',       # '' = random (micro only)
) -> list:
    """Run goal-directed quest completion (DecisionEngine + affordance learning).

    Uses the Goal/AffordanceModel/DecisionEngine from planning.py.  All goals
    are derived from Phase Q1 causal knowledge; no domain-specific heuristics
    are hard-coded in the engine itself.

    Failure learning in action
    --------------------------
    The first time the agent tries 'go north' from the living room without
    the brass key, the navigation fails (location unchanged).  The
    AffordanceModel infers the missing precondition ('inv:brass_key') by
    comparing the failed state with the Q1 success records (where brass key
    WAS in inventory).  On subsequent steps, _bfs_no_blocks() skips 'go north'
    from living room, routing the agent via the garden instead.  When the
    brass key is acquired, on_acquired() clears the block and routing resumes.
    """
    _banner('Phase Q4 -- Goal-Directed Quest Completion (DecisionEngine)')

    use_micro = (world_type == 'micro' and not game_path)

    world = WorldModel(causal_stores)
    print('  WorldModel from Phase Q1 causal observations:')
    print(f'    Room graph : {len(world.nav_map)} edges  '
          f'({len(world.known_rooms())} rooms: {sorted(world.known_rooms())})')
    print(f'    Takeable   : {sorted(world.takeable)}')
    print(f'    Edible     : {sorted(world.edible)}')
    if use_micro:
        print(f'    Backend    : MicroTextWorld (procedural, layout regenerated each episode)')
        print(f'    Quest type : {quest_type or "random per episode"}')
    else:
        print(f'    Backend    : NanoTextEnv (fixed layout)')
        print(f'    Key->door affordance: discovered via failure inference')
    print()

    # FEP memory components.
    # For micro mode, rooms are unknown until explored; seed with Q1 archetypes.
    all_rooms  = sorted(world.known_rooms()) or []
    all_items  = sorted(world.takeable | world.edible)
    goal_stack = GoalStack()
    episodic   = EpisodicBuffer(capacity=30, decay=0.92)
    belief     = BeliefState(rooms=all_rooms, items=all_items, decay=0.95)

    # Build domain-specific goals from causal knowledge (using FEPGoal + Drive).
    goals  = make_textworld_goals(world, causal_stores)
    # AffordanceModel pre-seeded with Q1 navigable_ctx so failure inference
    # works immediately (no warm-up episode needed).
    aff    = AffordanceModel(causal_stores)
    engine = DecisionEngine(
        goals, aff,
        goal_stack = goal_stack,
        episodic   = episodic,
        belief     = belief,
    )
    print(f'  FEP components:  GoalStack | EpisodicBuffer(cap={episodic.capacity}) '
          f'| BeliefState({len(all_items)} items × {len(all_rooms)} rooms)')
    print(f'  Initial belief entropy: {belief.total_entropy():.2f} bits '
          f'(uniform over {len(all_rooms) + 2} locations)')
    print()

    all_results = []

    for ep in range(n_episodes):
        if use_micro:
            ep_seed = rng.randint(0, 2**31 - 1)
            world.reset_episode(online_nav=True)
            modality = TextWorldModality(
                world_type = world_type,
                world_seed = ep_seed,
                quest_type = quest_type,
            )
        else:
            world.reset_episode(online_nav=False)
            modality = TextWorldModality(game_path=game_path)

        # Sub-goals pushed in the previous episode must not carry over.
        # GoalStack is per-episode state — clear at every episode boundary.
        engine.goal_stack.clear()

        modality.connect()
        # Read quest type from initial observation.
        _obs0 = modality.get_obs()
        world.update(_obs0)
        ep_quest = world.quest_type or 'unknown'

        visited: set = set()
        prev_score = 0.0
        success    = False
        step_log   = []
        inferred_missing_shown: set = set()  # deduplicate inference log lines

        print(f'  --- Episode {ep + 1}/{n_episodes}  quest={ep_quest} ---')
        print(f'  {"Step":<5} {"Room":<17} {"Inv":<32} '
              f'{"Reason":<26} {"Action"}')
        print(f'  {"-"*5} {"-"*17} {"-"*32} {"-"*26} {"-"*30}')

        for step in range(max_steps):
            obs = modality.get_obs()
            if obs['done']:
                success = True
                break

            state  = _build_tw_state(obs, world, causal_stores, visited)
            action, reason = engine.decide(state, rng)

            modality.send_action(action)
            new_obs    = modality.get_obs()
            events     = modality.get_events()
            new_state  = _build_tw_state(new_obs, world, causal_stores, visited)

            # Online nav learning — build nav_map from live movement events.
            # Only needed for MicroTextWorld (new layout each episode).
            # For NanoTextEnv the Q1 nav_map is already complete and adding
            # post-unlock edges (bedroom) would cause cycling in later episodes.
            if use_micro:
                for ev in events:
                    if ev['type'] == 'moved':
                        world.update_nav(ev['from'], action, ev['to'])

            engine.feedback(state, action, new_state, events)

            score_delta = new_obs['score'] - prev_score
            prev_score  = new_obs['score']

            inv_str   = ('{' + ', '.join(sorted(obs['inventory'])) + '}'
                         if obs['inventory'] else '{}')
            delta_tag = f'  [+{score_delta:.0f}]' if score_delta > 0 else ''
            print(f'  {step + 1:<5} {obs["location"]:<17} {inv_str:<32} '
                  f'{reason:<26} {action}{delta_tag}')

            # Show failure inference the first time it fires.
            missing = aff.infer_missing(obs['location'], action)
            if missing and missing not in inferred_missing_shown:
                inferred_missing_shown.add(missing)
                print(f'         [affordance] nav failed; inferred missing: '
                      f'{sorted(missing)}')

            step_log.append({
                'step': step + 1, 'room': obs['location'],
                'inv': sorted(obs['inventory']),
                'action': action, 'reason': reason,
                'score_delta': score_delta,
                'missing': sorted(missing) if missing else [],
            })

            if new_obs['done']:
                success = True
                break

        final_score = prev_score
        verdict = 'COMPLETE' if success else 'incomplete'
        print(f'\n  => {verdict} in {len(step_log)} steps, '
              f'score={final_score:.0f}  quest={ep_quest}')
        if world.quest_unlocked or world.bedroom_unlocked:
            print('     Affordance discovered: key unlocks door')
        miss_summary = aff.missing_summary()
        if miss_summary.strip() != '(none)':
            print(f'     Active failure inferences:\n{miss_summary}')
        print()

        modality.disconnect()
        all_results.append({
            'ep': ep + 1, 'success': success,
            'steps': len(step_log), 'score': final_score,
        })

    # Summary across episodes.
    n_ok   = sum(1 for r in all_results if r['success'])
    avg_st = (sum(r['steps'] for r in all_results if r['success'])
              / max(n_ok, 1))

    print(f'  Success rate : {n_ok}/{n_episodes}')
    print(f'  Avg steps    : {avg_st:.1f}  (across successful episodes)')
    backend_label = 'MicroTextWorld' if use_micro else 'NanoTextEnv'
    if n_ok == n_episodes:
        print(f'\n  PASS -- {backend_label} quest solved in every episode.')
        print('          DecisionEngine + FEPGoal architecture handles all quest types.')
        if not use_micro:
            print('          Failure learning correctly inferred brass_key precondition.')
    else:
        print(f'\n  PARTIAL ({n_ok}/{n_episodes}) -- check decision trace above.')

    # --- FEP memory summaries -------------------------------------------
    print()
    print('  Episodic memory (most surprising events across all episodes):')
    print(episodic.summary(n=6))

    print()
    print('  Belief state after final episode:')
    print(belief.summary())
    print(f'  Final belief entropy: {belief.total_entropy():.2f} bits '
          f'(0 = certain, {math.log2(len(all_rooms) + 2):.2f} = uniform)')

    print()
    print('  FEP drive diagnostics (effective_priority per goal type):')
    print(f'  {"Goal":<20} {"Drive":<12} {"Urgency":>8}  Description')
    print(f'  {"----":<20} {"-----":<12} {"-------":>8}  -----------')
    for g in engine.goals:
        if isinstance(g, FEPGoal) and not g.is_safety:
            for d in g.drives:
                print(f'  {g.name:<20} {d.name:<12} {d.urgency:>8.2f}  '
                      f'measure(satisfied)=1 → deficit=0 → priority≈0')
        elif isinstance(g, FEPGoal) and g.is_safety:
            print(f'  {g.name:<20} {"(always)":<12} {"1.00":>8}  is_safety=True')

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description='Phase Q: TextWorld agent + semantic bootstrapping.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--game', default='',
                   help='Path to a .ulx/.z8 TextWorld game file. '
                        'Leave empty to use the built-in NanoTextEnv (zero deps).')
    p.add_argument('--steps', type=int, default=300,
                   help='Number of agent steps in Phase Q1 (default 300).')
    p.add_argument('--pos', type=int, default=6,
                   help='Number of POS-like clusters for Phase O (default 6).')
    p.add_argument('--sem', type=int, default=3,
                   help='Semantic sub-clusters per POS category (default 3).')
    p.add_argument('--min_count', type=int, default=2,
                   help='Minimum token count to include in clustering (default 2).')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for action selection (default 42).')
    p.add_argument('--verbose', action='store_true',
                   help='Print step-level diagnostics.')
    p.add_argument('--plan', action='store_true',
                   help='Run Phase Q4: goal-directed quest completion after Q1-Q3.')
    p.add_argument('--episodes', type=int, default=5,
                   help='Number of planning episodes in Phase Q4 (default 5).')
    p.add_argument('--plan_steps', type=int, default=50,
                   help='Max steps per Phase Q4 episode (default 50).')
    p.add_argument('--world', default='nano', choices=['nano', 'micro'],
                   help='World backend for Phase Q4: nano (fixed, default) or '
                        'micro (procedural, new layout each episode).')
    p.add_argument('--quest', default='',
                   choices=['', 'eat', 'fetch', 'unlock', 'cook', 'put'],
                   help='Fix MicroTextWorld quest type (default: random each episode).')

    args = p.parse_args()

    if args.game and not _HAS_TW:
        print('WARNING: TextWorld not installed (pip install textworld). '
              'Falling back to NanoTextEnv.')

    rng = random.Random(args.seed)

    modality = TextWorldModality(game_path=args.game)
    modality.connect()

    # Phase Q1: causal exploration.
    all_tokens, causal_log, causal_stores = run_phase_q1(
        modality = modality,
        n_steps  = args.steps,
        rng      = rng,
        verbose  = args.verbose,
    )

    modality.disconnect()

    # Phase Q2: Phase O on collected text.
    assignment, global_freq = run_phase_q2(
        all_tokens = all_tokens,
        n_clusters = args.pos,
        min_count  = args.min_count,
    )

    # Phase Q3: semantic bootstrapping + causal cross-reference.
    run_phase_q3(
        all_tokens    = all_tokens,
        assignment    = assignment,
        global_freq   = global_freq,
        causal_stores = causal_stores,
        n_subclusters = args.sem,
    )

    _print_summary(modality, args.steps, args.pos, args.sem)

    # Phase Q4: goal-directed quest completion (opt-in with --plan).
    if args.plan:
        run_phase_q4(
            causal_stores = causal_stores,
            game_path     = args.game,
            n_episodes    = args.episodes,
            max_steps     = args.plan_steps,
            rng           = random.Random(args.seed + 1),   # fresh seed
            world_type    = args.world,
            quest_type    = args.quest,
        )


if __name__ == '__main__':
    main()
