"""

Control Interface for Predictive Coding Agent

This module handles user commands and model responses.

Commands are sent via keyboard hotkeys while the agent is running.

Hotkeys:

    F5  - Start/resume agent

    F6  - Pause agent (freezes motor output, continues perception)

    F7  - Stop agent completely

    F8  - Save checkpoint

    F9  - Trigger consolidation (sleep mode)

    F10 - Open query interface (for text-pretrained models)

    F12 - Emergency stop (kills process)

For text-pretrained models, the query interface allows:

    - Typing a question

    - Model responds by predicting text tokens

    - Response is displayed in a simple GUI window

"""

from enum import Enum, auto

from typing import Optional, Callable

import threading

from pynput import keyboard


class AgentState(Enum):

    STOPPED = auto()

    RUNNING = auto()

    PAUSED = auto()

    CONSOLIDATING = auto()

    QUERYING = auto()


class ControlInterface:

    """

    Manages user control of the agent via hotkeys.

    

    Usage:

        control = ControlInterface()

        control.on_state_change(callback_function)

        control.start_listening()

    """

    

    def __init__(self):

        self.state = AgentState.STOPPED

        self._callbacks: list[Callable[[AgentState], None]] = []

        self._listener: Optional[keyboard.Listener] = None

        self._consolidation_callback: Optional[Callable] = None

        self._query_callback: Optional[Callable[[str], str]] = None

    

    def on_state_change(self, callback: Callable[[AgentState], None]):

        """Register a callback for state changes."""

        self._callbacks.append(callback)

    

    def on_consolidation_request(self, callback: Callable):

        """Register callback for consolidation trigger."""

        self._consolidation_callback = callback

    

    def on_query(self, callback: Callable[[str], str]):

        """Register callback for text queries (text-pretrained models only)."""

        self._query_callback = callback

    

    def _notify_state_change(self):

        for callback in self._callbacks:

            callback(self.state)

    

    def _on_key_press(self, key):

        try:

            if key == keyboard.Key.f5:

                if self.state in (AgentState.STOPPED, AgentState.PAUSED):

                    self.state = AgentState.RUNNING

                    self._notify_state_change()

                    print("[Control] Agent RUNNING")

            

            elif key == keyboard.Key.f6:

                if self.state == AgentState.RUNNING:

                    self.state = AgentState.PAUSED

                    self._notify_state_change()

                    print("[Control] Agent PAUSED")

            

            elif key == keyboard.Key.f7:

                self.state = AgentState.STOPPED

                self._notify_state_change()

                print("[Control] Agent STOPPED")

            

            elif key == keyboard.Key.f8:

                print("[Control] Checkpoint save requested")

                # Checkpoint saving handled by main loop

            

            elif key == keyboard.Key.f9:

                if self.state == AgentState.RUNNING:

                    self.state = AgentState.CONSOLIDATING

                    self._notify_state_change()

                    print("[Control] Consolidation STARTED")

                    if self._consolidation_callback:

                        # Run consolidation in background thread

                        thread = threading.Thread(target=self._run_consolidation)

                        thread.start()

            

            elif key == keyboard.Key.f10:

                if self._query_callback and self.state == AgentState.PAUSED:

                    self.state = AgentState.QUERYING

                    self._notify_state_change()

                    self._open_query_interface()

            

            elif key == keyboard.Key.f12:

                print("[Control] EMERGENCY STOP")

                import sys

                sys.exit(1)

                

        except Exception as e:

            print(f"[Control] Error handling key: {e}")

    

    def _run_consolidation(self):

        """Run consolidation and return to running state when done."""

        try:

            if self._consolidation_callback:

                self._consolidation_callback()

        finally:

            self.state = AgentState.RUNNING

            self._notify_state_change()

            print("[Control] Consolidation COMPLETE, resuming")

    

    def _open_query_interface(self):

        """

        Open a simple text input for querying the model.

        

        For text-pretrained models only. The model responds by

        generating text tokens, which are decoded and displayed.

        """

        # TODO: Implement simple tkinter dialog for text input/output

        # For now, use console input

        print("\n" + "="*50)

        print("QUERY MODE (type 'exit' to return to game)")

        print("="*50)

        

        while self.state == AgentState.QUERYING:

            try:

                query = input("\nYou: ").strip()

                if query.lower() == 'exit':

                    break

                if query and self._query_callback:

                    response = self._query_callback(query)

                    print(f"\nAgent: {response}")

            except EOFError:

                break

        

        self.state = AgentState.PAUSED

        self._notify_state_change()

        print("\n[Control] Exited query mode")

    

    def start_listening(self):

        """Start listening for hotkey commands."""

        self._listener = keyboard.Listener(on_press=self._on_key_press)

        self._listener.start()

        print("[Control] Hotkey listener started")

        print("  F5=Start  F6=Pause  F7=Stop  F8=Save  F9=Sleep  F10=Query  F12=Kill")

    

    def stop_listening(self):

        """Stop listening for hotkey commands."""

        if self._listener:

            self._listener.stop()

            self._listener = None


# Convenience function for main script

def create_control_interface() -> ControlInterface:

    """Create and return a configured control interface."""

    return ControlInterface()
