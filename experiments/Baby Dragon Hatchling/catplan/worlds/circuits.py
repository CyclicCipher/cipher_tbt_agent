"""Circuits world simulator.

Hidden rules (the learner must discover these):
- Components connect between nodes
- A battery creates a voltage difference between its terminals
- Current flows through a path from high to low voltage
- An LED lights when sufficient current flows through it
- Resistors reduce current (V = IR)
- Series resistances add, parallel resistances combine as 1/R_total = sum(1/R_i)
- Kirchhoff: voltage around any loop = 0, current at any node = 0

The learner sees: connections, voltage at nodes, current through wires, LED state.
It must discover: circuit laws, series/parallel duality, Kirchhoff's laws.
"""
from __future__ import annotations

from .base import World, Observation


class CircuitsWorld(World):
    """A world of circuit components, nodes, and signals."""

    def __init__(self):
        super().__init__()
        # Nodes.
        self._nodes: set[str] = {"n1", "n2", "n3", "n4", "gnd"}
        # Components: {name: {type, node_a, node_b, value}}.
        self._components: dict[str, dict] = {}
        # Available components to place.
        self._available: dict[str, dict] = {
            "bat1": {"type": "battery", "voltage": 5.0},
            "r1": {"type": "resistor", "resistance": 100.0},
            "r2": {"type": "resistor", "resistance": 200.0},
            "led1": {"type": "led", "threshold": 1.5},
            "sw1": {"type": "switch"},
        }
        self._switches: dict[str, bool] = {}  # switch_name -> on/off

    def reset(self):
        self._components.clear()
        self._switches.clear()
        self._history.clear()

    def _solve_circuit(self) -> dict[str, float]:
        """Simple circuit solver: compute node voltages and currents.

        Very simplified: only handles series circuits with one battery.
        """
        voltages: dict[str, float] = {n: 0.0 for n in self._nodes}
        currents: dict[str, float] = {}

        # Find battery.
        battery = None
        for name, comp in self._components.items():
            if comp["type"] == "battery":
                battery = comp
                break

        if battery is None:
            return {"voltages": voltages, "currents": currents, "led_on": {}}

        # Set battery voltage.
        voltages[battery["node_a"]] = battery["voltage"]
        voltages[battery["node_b"]] = 0.0  # ground

        # Find path from high to low voltage through components.
        # Simplified: compute total resistance in path.
        total_resistance = 0.0
        path_components = []
        for name, comp in self._components.items():
            if comp["type"] == "resistor":
                total_resistance += comp["resistance"]
                path_components.append(name)
            elif comp["type"] == "switch":
                if not self._switches.get(name, False):
                    total_resistance = float("inf")  # open switch
                path_components.append(name)

        # Current = V / R.
        if total_resistance > 0:
            current = battery["voltage"] / total_resistance
        else:
            current = 0.0

        for name in path_components:
            currents[name] = current

        # LED state.
        led_on = {}
        for name, comp in self._components.items():
            if comp["type"] == "led":
                # LED lights if current flows and voltage exceeds threshold.
                led_current = currents.get(name, current)  # simplified
                led_on[name] = led_current > 0 and battery["voltage"] > comp["threshold"]

        return {"voltages": voltages, "currents": currents, "led_on": led_on}

    def observe(self) -> Observation:
        facts = set()

        # Nodes.
        for n in self._nodes:
            facts.add(("node", (n,), True))

        # Placed components and their connections.
        for name, comp in self._components.items():
            facts.add(("placed", (name,), True))
            facts.add(("comp_type", (name, comp["type"]), True))
            facts.add(("connected", (name, comp["node_a"], comp["node_b"]), True))

        # Available components.
        for name in self._available:
            if name not in self._components:
                facts.add(("available", (name,), True))

        # Switch states.
        for name, on in self._switches.items():
            facts.add(("switch_on", (name,), on))

        # Circuit solution (voltages, currents, LED state).
        solution = self._solve_circuit()
        for node, v in solution["voltages"].items():
            if v != 0.0:
                facts.add(("voltage", (node,), round(v, 2)))
        for comp, i in solution["currents"].items():
            if i != 0.0:
                facts.add(("current", (comp,), round(i, 4)))
        for led, on in solution["led_on"].items():
            facts.add(("led_on", (led,), on))

        return Observation(facts=frozenset(facts))

    def available_actions(self) -> list[tuple[str, tuple[str, ...]]]:
        actions = []
        nodes = sorted(self._nodes)

        # Connect: place an available component between two nodes.
        for comp_name in sorted(self._available.keys()):
            if comp_name in self._components:
                continue
            for i, na in enumerate(nodes):
                for nb in nodes[i+1:]:
                    actions.append(("connect", (comp_name, na, nb)))

        # Disconnect: remove a placed component.
        for comp_name in sorted(self._components.keys()):
            actions.append(("disconnect", (comp_name,)))

        # Toggle: flip a switch.
        for name, comp in self._components.items():
            if comp["type"] == "switch":
                actions.append(("toggle", (name,)))

        return actions

    def _execute_impl(self, action: str, args: tuple[str, ...]):
        if action == "connect":
            comp_name, node_a, node_b = args
            if comp_name in self._available and comp_name not in self._components:
                comp = dict(self._available[comp_name])
                comp["node_a"] = node_a
                comp["node_b"] = node_b
                self._components[comp_name] = comp
                if comp["type"] == "switch":
                    self._switches[comp_name] = False
        elif action == "disconnect":
            comp_name = args[0]
            self._components.pop(comp_name, None)
            self._switches.pop(comp_name, None)
        elif action == "toggle":
            comp_name = args[0]
            if comp_name in self._switches:
                self._switches[comp_name] = not self._switches[comp_name]


if __name__ == "__main__":
    print("=== Circuits World ===")
    w = CircuitsWorld()
    print(f"Available actions: {len(w.available_actions())}")

    # Build a simple circuit: battery + resistor + LED.
    w.execute("connect", ("bat1", "n1", "gnd"))
    w.execute("connect", ("r1", "n1", "n2"))
    w.execute("connect", ("led1", "n2", "gnd"))

    obs = w.observe()
    print("\nAfter building circuit:")
    for p, a, v in sorted(obs.facts):
        if v is True or isinstance(v, (int, float)):
            print(f"  {p}({', '.join(str(x) for x in a)}) = {v}")
