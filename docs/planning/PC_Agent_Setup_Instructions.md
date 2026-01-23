# Predictive Coding Agent: Setup Instructions

Complete setup guide for Windows with NVIDIA GPU.

---

## Prerequisites

Before starting, ensure you have:
- Windows 10 or 11
- NVIDIA GPU with updated drivers
- Administrator access
- Danganronpa: Trigger Happy Havoc installed (Steam version recommended)
- ~10 GB free disk space

---

## Part 1: Install Core Software

### 1.1 Install Python 3.11

1. Go to https://www.python.org/downloads/release/python-3119/
2. Scroll down and download **Windows installer (64-bit)**
3. Run the installer
4. **IMPORTANT**: Check the box that says **"Add python.exe to PATH"**
5. Click **"Install Now"**
6. When complete, click "Close"

**Verify installation** — Open Command Prompt (press `Win+R`, type `cmd`, press Enter):

```cmd
python --version
```

Expected output: `Python 3.11.x`

```cmd
pip --version
```

Expected output: `pip 23.x.x from ...`

---

### 1.2 Install Git

1. Go to https://git-scm.com/download/win
2. Download the 64-bit installer
3. Run the installer
4. Accept all default options (click Next repeatedly)
5. Click Install, then Finish

**Verify installation**:

```cmd
git --version
```

Expected output: `git version 2.x.x`

---

### 1.3 Check NVIDIA Driver and CUDA

Open Command Prompt and run:

```cmd
nvidia-smi
```

Expected output: A table showing your GPU (RTX 3050 Ti) and CUDA Version (should be 12.x).

If this command fails:
1. Go to https://www.nvidia.com/Download/index.aspx
2. Find and download the latest driver for RTX 3050 Ti Laptop
3. Install and restart your computer

---

## Part 2: Create Project

### 2.1 Create Project Directory

Open Command Prompt and run these commands one at a time:

```cmd
cd %USERPROFILE%
mkdir Projects
cd Projects
mkdir predictive-coding-agent
cd predictive-coding-agent
```

You are now in: `C:\Users\<YourName>\Projects\predictive-coding-agent`

---

### 2.2 Initialize Git Repository

```cmd
git init
```

Expected output: `Initialized empty Git repository in ...`

---

### 2.3 Create Virtual Environment

```cmd
python -m venv venv
```

This creates a `venv` folder. Now activate it:

```cmd
venv\Scripts\activate
```

Your prompt should now show `(venv)` at the beginning:
```
(venv) C:\Users\<YourName>\Projects\predictive-coding-agent>
```

**IMPORTANT**: Every time you open a new Command Prompt to work on this project, you must:
1. Navigate to the project folder: `cd %USERPROFILE%\Projects\predictive-coding-agent`
2. Activate the venv: `venv\Scripts\activate`

---

## Part 3: Install Dependencies

Make sure your venv is activated (you see `(venv)` in your prompt).

### 3.1 Install PyTorch with CUDA

This is the largest download (~2.5 GB). Run:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Wait for it to complete (may take several minutes).

**Verify GPU access**:

```cmd
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

If CUDA available shows False, your NVIDIA drivers may need updating.

---

### 3.2 Install Screen Capture and Input Libraries

```cmd
pip install mss
pip install pynput
pip install PyGetWindow
pip install pillow
```

---

### 3.3 Install Audio Library

```cmd
pip install soundcard
```

---

### 3.4 Install Utility Libraries

```cmd
pip install numpy
pip install tqdm
pip install tensorboard
pip install pyyaml
pip install matplotlib
```

---

### 3.5 Save Requirements

```cmd
pip freeze > requirements.txt
```

This creates a file listing all installed packages for reproducibility.

---

## Part 4: Create Project Structure

Run these commands to create the folder structure:

```cmd
mkdir src
mkdir src\network
mkdir src\wrapper
mkdir src\hippocampus
mkdir src\pretraining
mkdir src\control
mkdir configs
mkdir logs
mkdir checkpoints
mkdir data
mkdir data\recordings
mkdir data\text_corpus
mkdir data\rendered_text
mkdir tests
```

---

## Part 5: Create Initial Files

### 5.1 Create .gitignore

Create a file called `.gitignore` in the project root. You can do this with:

```cmd
notepad .gitignore
```

Paste the following content, save, and close:

```
# Virtual environment
venv/

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs and checkpoints
logs/
checkpoints/
*.pt
*.pth
*.ckpt

# Data (too large for git)
data/recordings/
data/text_corpus/
data/rendered_text/

# OS files
.DS_Store
Thumbs.db

# Environment variables
.env

# Temporary files
*.tmp
*.temp
```

---

### 5.2 Create Main Configuration File

```cmd
notepad configs\default.yaml
```

Paste the following, save, and close:

```yaml
# Predictive Coding Agent Configuration

# Hardware
device: "cuda"
dtype: "float32"  # Use float16 if memory is tight

# Network Architecture
network:
  num_layers: 10
  neurons_per_layer: 1500
  temporal_window: 8
  
  # Neuron parameters
  neuron:
    apical_kernel_size: 8
    basal_kernel_size: 8
    initial_gate: 0.5
    divisive_norm_sigma: 0.01

# Foveal Vision
vision:
  fovea_size: 320          # pixels, square
  periphery_size: 96       # pixels, square
  frame_buffer_size: 16    # number of frames
  
# Audio
audio:
  sample_rate: 16000
  chunk_duration_ms: 100
  mel_bands: 64
  buffer_size: 16          # number of chunks

# Learning
learning:
  backbone_lr_awake: 0.001
  backbone_lr_consolidation: 0.01
  overlay_lr_awake: 0.001
  overlay_lr_consolidation: 0.01
  
# Hippocampus
hippocampus:
  buffer_size: 2000        # number of episodes
  latent_dim: 512
  temporal_context: 8      # timesteps before/after
  salience_percentile: 80  # top 20% get stored
  consolidation_replay_speed: 10  # 10x faster

# Motor Control
motor:
  cursor_gain: 5.0
  click_threshold: 0.7
  gaze_gain: 50.0
  
# Overlay (experimental)
overlay:
  enabled: true
  sparsity: 0.03           # 3% of connections
  refinement_steps: 2

# Consolidation
consolidation:
  trigger: "manual"        # "manual" or "pause_detected"
  min_duration_seconds: 60
  interruptible: true
  
# Logging
logging:
  tensorboard: true
  log_interval: 100        # steps
  checkpoint_interval: 1000
```

---

### 5.3 Create Package Init Files

These empty files tell Python these folders are packages:

```cmd
type nul > src\__init__.py
type nul > src\network\__init__.py
type nul > src\wrapper\__init__.py
type nul > src\hippocampus\__init__.py
type nul > src\pretraining\__init__.py
type nul > src\control\__init__.py
type nul > tests\__init__.py
```

---

### 5.4 Create Control Interface Stub

```cmd
notepad src\control\interface.py
```

Paste the following, save, and close:

```python
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
```

---

## Part 6: Verify Setup

### 6.1 Test Screen Capture

Start Danganronpa in **windowed mode** (not fullscreen). Then run:

```cmd
notepad test_screen.py
```

Paste:

```python
import mss
import numpy as np
from PIL import Image

print("Testing screen capture...")
print("Make sure Danganronpa is running in windowed mode.")
input("Press Enter to capture...")

with mss.mss() as sct:
    # Capture primary monitor
    monitor = sct.monitors[1]
    img = sct.grab(monitor)
    arr = np.array(img)
    
    print(f"Captured shape: {arr.shape}")
    print(f"Monitor info: {monitor}")
    
    # Save a test image
    Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX").save("test_capture.png")
    print("Saved test_capture.png")

print("\nScreen capture working!")
```

Save, close, and run:

```cmd
python test_screen.py
```

Check that `test_capture.png` was created and shows your screen.

---

### 6.2 Test Audio Capture

```cmd
notepad test_audio.py
```

Paste:

```python
import soundcard as sc
import numpy as np

print("Testing audio capture (WASAPI loopback)...")
print("Play some audio on your computer.")

# Get the default speaker as a microphone (loopback)
speaker = sc.default_speaker()
print(f"Default speaker: {speaker.name}")

mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
print(f"Loopback mic: {mic.name}")

print("\nRecording 2 seconds of audio...")
with mic.recorder(samplerate=16000, channels=1) as rec:
    audio = rec.record(numframes=32000)  # 2 seconds at 16kHz

print(f"Recorded shape: {audio.shape}")
print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

if np.abs(audio).max() > 0.001:
    print("\nAudio capture working! Sound detected.")
else:
    print("\nWarning: Very quiet audio. Make sure something is playing.")

print("Audio test complete!")
```

Save, close, and run (make sure some audio is playing):

```cmd
python test_audio.py
```

---

### 6.3 Test Input Injection

**WARNING**: This will move your mouse. Be ready.

```cmd
notepad test_input.py
```

Paste:

```python
from pynput.mouse import Controller as MouseController
from pynput.keyboard import Controller as KeyboardController, Key
import time

print("Testing input injection...")
print("Your mouse will move in 3 seconds!")
time.sleep(3)

mouse = MouseController()
keyboard = KeyboardController()

# Record starting position
start_pos = mouse.position
print(f"Starting position: {start_pos}")

# Move mouse
mouse.position = (100, 100)
time.sleep(0.5)
print(f"Moved to: {mouse.position}")

# Move back
mouse.position = start_pos
print(f"Returned to: {mouse.position}")

print("\nInput injection working!")
```

Save, close, and run:

```cmd
python test_input.py
```

---

### 6.4 Test PyTorch GPU Memory

```cmd
notepad test_gpu.py
```

Paste:

```python
import torch

print("Testing PyTorch GPU...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# Check available memory
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Total GPU memory: {total:.2f} GB")

# Try allocating a test tensor
print("\nAllocating test tensor (500MB)...")
x = torch.randn(128, 1024, 1024, device='cuda')
allocated = torch.cuda.memory_allocated() / 1e9
print(f"Memory allocated: {allocated:.2f} GB")

del x
torch.cuda.empty_cache()
print("Freed memory.")

print("\nGPU test complete!")
```

Save, close, and run:

```cmd
python test_gpu.py
```

---

## Part 7: Clean Up Test Files

After verifying everything works:

```cmd
del test_screen.py
del test_audio.py
del test_input.py
del test_gpu.py
del test_capture.png
```

---

## Part 8: Initial Git Commit

```cmd
git add .
git commit -m "Initial project setup"
```

---

## Part 9: Connect to GitHub (Optional)

If you want to back up your project to GitHub:

1. Go to https://github.com and sign in
2. Click the "+" in the top right, then "New repository"
3. Name it `predictive-coding-agent`
4. Keep it private (recommended)
5. Do NOT initialize with README (we already have files)
6. Click "Create repository"
7. Copy the URL (looks like `https://github.com/yourusername/predictive-coding-agent.git`)

Then run:

```cmd
git remote add origin https://github.com/yourusername/predictive-coding-agent.git
git branch -M main
git push -u origin main
```

---

## Part 10: Ready for Claude Code

Your project is now set up. The folder structure should look like:

```
predictive-coding-agent/
├── venv/                    # Virtual environment (not in git)
├── src/
│   ├── __init__.py
│   ├── network/
│   │   └── __init__.py
│   ├── wrapper/
│   │   └── __init__.py
│   ├── hippocampus/
│   │   └── __init__.py
│   ├── pretraining/
│   │   └── __init__.py
│   └── control/
│       ├── __init__.py
│       └── interface.py
├── configs/
│   └── default.yaml
├── logs/                    # Empty, for tensorboard logs
├── checkpoints/             # Empty, for model checkpoints
├── data/
│   ├── recordings/          # Empty, for gameplay recordings
│   ├── text_corpus/         # Empty, for pretraining text
│   └── rendered_text/       # Empty, for rendered text images
├── tests/
│   └── __init__.py
├── .gitignore
└── requirements.txt
```

To use Claude Code with this project:

1. Open Claude Code or your terminal with Claude Code access
2. Navigate to the project: `cd %USERPROFILE%\Projects\predictive-coding-agent`
3. Activate the venv: `venv\Scripts\activate`
4. Provide Claude Code with:
   - The Planning Document (Predictive_Coding_Agent_Planning_Document_v2.docx)
   - This setup document
   - The instruction to begin implementation

---

## Quick Reference: Daily Workflow

Every time you work on the project:

```cmd
cd %USERPROFILE%\Projects\predictive-coding-agent
venv\Scripts\activate
```

To run the agent (once implemented):

```cmd
python -m src.main
```

Hotkeys while running:
- **F5** — Start/resume
- **F6** — Pause
- **F7** — Stop
- **F8** — Save checkpoint
- **F9** — Trigger consolidation (manual sleep)
- **F10** — Query mode (text-pretrained models only)
- **F12** — Emergency stop

---

## Troubleshooting

**"python is not recognized"**
- Reinstall Python and check "Add to PATH"
- Or use full path: `C:\Users\<YourName>\AppData\Local\Programs\Python\Python311\python.exe`

**"CUDA not available"**
- Update NVIDIA drivers
- Reinstall PyTorch with correct CUDA version

**Screen capture shows black screen**
- Some games block capture in fullscreen; use windowed mode
- Try running Command Prompt as Administrator

**Audio capture silent**
- Check Windows sound settings
- Make sure the game audio is routed to the default speaker

**Permission denied on input injection**
- Some games block synthetic input; Danganronpa should work
- Try running as Administrator if needed
