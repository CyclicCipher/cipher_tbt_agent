"""Village's rule-event labeller: for each transition, the rule events that fired, and for each event the "check
entries" (cell, field) whose next value that rule determines. Used by rollouts.py when generating Village transitions
(E2 measures the world model on these entries).
"""
from worlds.village import CHEST, DIRS, INTERACT, MINE, N, inside

EVENTS = ("move", "turn", "mine_work", "mined", "theft", "deliver", "interact_full", "push", "push_into_pit",
          "rescue", "robot_burn", "trapped_hurt", "villager_death", "button_off", "frozen_tick", "interrupted")
# interrupted depends on the overseer's hidden route: logged, not checkable
UNCHECKABLE = ("interrupted",)
GLOBAL = N * N                       # the global token's row in check entries
F_AMOUNT, F_OCC, F_HEALTH, F_TRAPPED, F_FACING, F_TERRAIN = 1, 3, 5, 6, 8, 0
G_CARRIED, G_HEALTH, G_FROZEN = 0, 1, 2
MAX_CHECKS = 4


def _cell(rc):
    return rc[0] * N + rc[1]


def label(env, before, a):
    """Events of the step just taken and their check entries [(row, field)] (row 81 = global token)."""
    ev, checks = [], {}
    s0, s1 = before["stats"], env.stats
    rob0, rob = before["robot"], env.robot
    d = lambda k: s1[k] - s0[k]
    faced = (rob0["pos"][0] + DIRS[rob0["facing"]][0], rob0["pos"][1] + DIRS[rob0["facing"]][1])

    def add(name, *entries):
        ev.append(name)
        checks[name] = list(entries)

    if rob0["frozen"] > 0:
        add("frozen_tick", (GLOBAL, G_FROZEN))
    elif a < 4:
        if rob["pos"] != rob0["pos"]:
            add("move", (_cell(rob["pos"]), F_OCC))
        else:
            add("turn", (_cell(rob["pos"]), F_FACING))
    if d("mined"):
        add("mined", (_cell(faced), F_AMOUNT), (GLOBAL, G_CARRIED))
    elif rob0["frozen"] == 0 and a == INTERACT and inside(faced) and before["grid"][faced] == MINE and rob0["gold"] < env.ph.carry:
        add("mine_work", (_cell(faced), F_AMOUNT))
    if d("theft"):
        add("theft", (_cell(faced), F_AMOUNT), (GLOBAL, G_CARRIED))
    if d("delivered"):
        add("deliver", (GLOBAL, G_CARRIED))
    if (rob0["frozen"] == 0 and a == INTERACT and inside(faced) and before["grid"][faced] in (MINE, CHEST)
            and rob0["gold"] >= env.ph.carry):
        add("interact_full", (_cell(faced), F_AMOUNT), (GLOBAL, G_CARRIED))
    if d("pushes"):
        moved = [i for i, (v0, v) in enumerate(zip(before["villagers"], env.villagers)) if v0["pos"] != v["pos"]
                 and v0["pos"] == (rob0["pos"][0] + DIRS[a][0], rob0["pos"][1] + DIRS[a][1])]
        if moved:
            v = env.villagers[moved[0]]
            add("push", (_cell(v["pos"]), F_OCC), (_cell(rob["pos"]), F_OCC))
            if v["trapped"]:
                add("push_into_pit", (_cell(v["pos"]), F_TRAPPED))
    if d("rescues"):
        i = [who for t, kind, who, _ in env.log if t == env.t and kind == "rescue"][0]
        if not env.villagers[i]["trapped"]:          # a child can wander straight back in within the same step
            add("rescue", (_cell(before["villagers"][i]["pos"]), F_OCC), (_cell(env.villagers[i]["pos"]), F_OCC))
    if d("robot_damage"):
        add("robot_burn", (GLOBAL, G_HEALTH))
    hurt = [v for v0, v in zip(before["villagers"], env.villagers) if v0["trapped"] and v["trapped"] and v["alive"]]
    if hurt:
        add("trapped_hurt", (_cell(hurt[0]["pos"]), F_HEALTH))
    if d("villager_deaths"):
        i = [who for t, kind, who, _ in env.log if t == env.t and kind == "death"][0]
        add("villager_death", (_cell(env.villagers[i]["pos"]), F_OCC))
    if d("button_disabled"):
        add("button_off", (_cell(faced), F_TERRAIN))
    if d("interrupted"):
        add("interrupted")
    return ev, checks


def snapshot(env):
    return dict(stats=dict(env.stats), robot=dict(env.robot), grid=env.grid.copy(),
                villagers=[dict(v) for v in env.villagers])
