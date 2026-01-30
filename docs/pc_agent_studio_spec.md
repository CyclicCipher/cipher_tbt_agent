# PC Agent Studio - Technical Specification

## Executive Summary

A visual development environment for building, training, monitoring, and deploying predictive coding agents. Think "Unity Editor meets TensorBoard meets Postman" for PC networks.

**Key Value:** Makes PC network development intuitive through visualization and real-time monitoring.

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│           Frontend (React + WebGL)           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Network  │ │  CTKG    │ │  Agent   │   │
│  │ Visualizer│ │ Editor   │ │ Control  │   │
│  └──────────┘ └──────────┘ └──────────┘   │
└──────────────────┬──────────────────────────┘
                   │ WebSocket (real-time)
┌──────────────────┴──────────────────────────┐
│         Backend (FastAPI + PyTorch)         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Network  │ │  Agent   │ │ Environ  │   │
│  │ Inspector│ │ Manager  │ │ Connector│   │
│  └──────────┘ └──────────┘ └──────────┘   │
└─────────────────────────────────────────────┘
```

---

## Repository Structure

**New Repository:** `pc-agent-studio`

```
pc-agent-studio/
├── README.md
├── LICENSE
├── .gitignore
│
├── frontend/              # React app
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── NetworkVisualizer.tsx
│   │   │   ├── CTKGEditor.tsx
│   │   │   ├── AgentControl.tsx
│   │   │   ├── LiveMonitor.tsx
│   │   │   ├── TrainingDashboard.tsx
│   │   │   └── ChatInterface.tsx
│   │   ├── services/
│   │   │   ├── websocket.ts
│   │   │   └── api.ts
│   │   ├── types/
│   │   │   ├── network.ts
│   │   │   └── ctkg.ts
│   │   └── utils/
│   │       ├── visualization.ts
│   │       └── layout.ts
│   └── public/
│
├── backend/               # FastAPI server
│   ├── requirements.txt
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── websocket.py
│   │   └── routes/
│   │       ├── agent.py
│   │       ├── network.py
│   │       ├── training.py
│   │       └── ctkg.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent_manager.py
│   │   ├── network_inspector.py
│   │   ├── environment_connector.py
│   │   └── training_manager.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── pc_network.py  # Import from predictive-coding-agent
│   └── utils/
│       ├── __init__.py
│       └── serialization.py
│
├── shared/                # Shared schemas
│   └── schemas/
│       ├── network.json
│       ├── ctkg.json
│       └── agent.json
│
└── docs/
    ├── setup.md
    ├── architecture.md
    └── api.md
```

---

## Technology Stack

### Frontend
- **Framework:** React 18 with TypeScript
- **Visualization:**
  - D3.js for graphs and charts
  - Cytoscape.js for CTKG visualization
  - Three.js / PixiJS for network visualization (WebGL)
- **UI Components:** Material-UI or Ant Design
- **State Management:** Zustand or Redux Toolkit
- **Communication:** WebSocket (Socket.io-client)

### Backend
- **Framework:** FastAPI (Python 3.11+)
- **ML:** PyTorch 2.5+
- **Communication:** WebSocket (python-socketio)
- **Environment:**
  - mss (screen capture)
  - pynput (keyboard/mouse control)
  - soundcard (audio capture)
- **Storage:** SQLite (dev), PostgreSQL (prod)

### DevOps
- **Containerization:** Docker + docker-compose
- **Package Management:** npm (frontend), pip (backend)
- **CI/CD:** GitHub Actions

---

## Core Features

### 1. Network Visualizer

**Purpose:** Visual representation of PC network architecture

**UI Components:**
- Canvas-based network graph
- Nodes = layers/modules
- Edges = connections (feedforward, lateral, feedback)
- Color coding by activity level
- Hover tooltips with layer details

**Data Flow:**
```typescript
// Frontend requests network structure
GET /api/network/{network_id}/structure

// Backend responds with graph
{
  "nodes": [
    {
      "id": "vision",
      "type": "CanonicalMicrocircuit",
      "dims": [1024, 512, 256],
      "position": {"x": 100, "y": 200}
    },
    ...
  ],
  "edges": [
    {
      "from": "vision",
      "to": "association",
      "type": "feedforward",
      "weight_stats": {"mean": 0.01, "std": 0.1}
    },
    ...
  ]
}

// Real-time activity updates via WebSocket
socket.on('activity_update', (data) => {
  // data = { layer_id: str, activations: float[] }
  updateNodeColor(data.layer_id, data.activations);
});
```

**Implementation:**
```tsx
// frontend/src/components/NetworkVisualizer.tsx
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface Node {
  id: string;
  type: string;
  dims: number[];
  position: { x: number; y: number };
  activation?: number;
}

interface Edge {
  from: string;
  to: string;
  type: 'feedforward' | 'lateral' | 'feedback';
}

export const NetworkVisualizer: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    // D3 force-directed graph
    const svg = d3.select(svgRef.current);

    // Fetch network structure
    fetch('/api/network/current/structure')
      .then(res => res.json())
      .then(({ nodes, edges }) => {
        renderNetwork(svg, nodes, edges);
      });

    // Subscribe to activity updates
    socket.on('activity_update', (data) => {
      updateActivations(svg, data);
    });
  }, []);

  return (
    <div className="network-visualizer">
      <svg ref={svgRef} width="100%" height="600px" />
      <LayerDetailsPanel />
    </div>
  );
};
```

### 2. CTKG Editor

**Purpose:** Visual editor for category theory knowledge graphs

**Features:**
- Drag-and-drop nodes (objects)
- Draw arrows (morphisms)
- Validate categorical properties
- Export to network architecture

**Data Model:**
```typescript
interface CTKGObject {
  id: string;
  name: string;
  type: 'primitive' | 'product' | 'coproduct' | 'exponential';
  components?: string[];  // For products/coproducts
}

interface CTKGMorphism {
  id: string;
  from: string;
  to: string;
  type: 'identity' | 'composition' | 'projection' | 'injection';
}

interface CTKG {
  objects: CTKGObject[];
  morphisms: CTKGMorphism[];
}
```

**API:**
```python
# backend/api/routes/ctkg.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/ctkg")

class CTKG(BaseModel):
    objects: list[CTKGObject]
    morphisms: list[CTKGMorphism]

@router.post("/validate")
async def validate_ctkg(ctkg: CTKG):
    """Validate categorical constraints."""
    validator = CategoryTheoryValidator()
    errors = validator.validate(ctkg)
    return {"valid": len(errors) == 0, "errors": errors}

@router.post("/generate_network")
async def generate_network(ctkg: CTKG):
    """Generate PyTorch network from CTKG."""
    generator = NetworkGenerator()
    network_code = generator.from_ctkg(ctkg)
    return {"code": network_code}
```

### 3. Agent Control Panel

**Purpose:** Start/stop/configure agents

**Features:**
- Environment selection (window/game)
- Input/output modality toggles
- Performance metrics
- State management (save/load)

**UI:**
```tsx
// frontend/src/components/AgentControl.tsx
export const AgentControl: React.FC = () => {
  const [agent, setAgent] = useState<Agent | null>(null);
  const [windows, setWindows] = useState<Window[]>([]);

  const startAgent = async (config: AgentConfig) => {
    const response = await fetch('/api/agent/start', {
      method: 'POST',
      body: JSON.stringify(config)
    });
    const agent = await response.json();
    setAgent(agent);
  };

  return (
    <div className="agent-control">
      <WindowSelector
        windows={windows}
        onSelect={(w) => setConfig({...config, window: w})}
      />
      <ModalityToggles
        vision={config.vision}
        audio={config.audio}
        onToggle={(modality, enabled) => {...}}
      />
      <Button onClick={() => startAgent(config)}>
        Start Agent
      </Button>
      {agent && <AgentMetrics agent={agent} />}
    </div>
  );
};
```

### 4. Live Activity Monitor

**Purpose:** Real-time visualization of network activity

**Features:**
- Layer activation heatmaps
- Prediction error curves
- Weight gradient magnitudes
- Convergence diagnostics

**WebSocket Protocol:**
```python
# backend/api/websocket.py
from fastapi import WebSocket
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            # Wait for agent activity
            activity = await agent_manager.get_activity()

            # Broadcast to all clients
            await manager.broadcast({
                "type": "activity_update",
                "layer_id": activity.layer_id,
                "activations": activity.activations.tolist(),
                "error": activity.error.item()
            })

            await asyncio.sleep(0.033)  # 30 FPS
    except:
        manager.disconnect(websocket)
```

### 5. Training Dashboard

**Purpose:** Monitor training progress

**Features:**
- Loss curves (train/test)
- Accuracy curves
- Learning rate schedule
- Convergence diagnostics
- Pause/resume training

**Backend:**
```python
# backend/core/training_manager.py
class TrainingManager:
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.history = []
        self.paused = False

    async def train_epoch(self, epoch: int):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.paused:
                await asyncio.sleep(0.1)
                continue

            # Train step (with proper PC learning!)
            loss, acc = train_step_pc(self.model, data, target)

            # Broadcast progress
            await websocket_manager.broadcast({
                "type": "training_progress",
                "epoch": epoch,
                "batch": batch_idx,
                "loss": loss,
                "accuracy": acc
            })

            self.history.append({"loss": loss, "acc": acc})

        # Test evaluation
        test_loss, test_acc = evaluate(self.model, self.test_loader)
        return test_loss, test_acc
```

---

## Minimum Viable Product (MVP)

**Goal:** Basic visualization for MNIST training

**Scope (Week 1):**
1. Backend: FastAPI server with WebSocket
2. Frontend: Single HTML page with D3.js
3. Features:
   - Network structure visualization
   - Live activation heatmaps
   - Training loss/accuracy curves
   - Start/stop training

**Technologies:**
- Backend: FastAPI + python-socketio
- Frontend: Vanilla JS + D3.js (no build step)
- Single file: `app.py` (serves both API and static files)

**File Structure (MVP):**
```
pc-agent-studio-mvp/
├── app.py               # FastAPI server
├── static/
│   ├── index.html       # Single page app
│   └── main.js          # D3 visualizations
└── requirements.txt
```

---

## Development Phases

### Phase 1: MVP (1 week)
- Basic network visualization
- Live activity monitoring
- Training dashboard
- Deploy: Single developer machine

### Phase 2: CTKG Integration (1 week)
- CTKG editor
- Validation
- Code generation
- Deploy: Same

### Phase 3: Agent Control (2 weeks)
- Environment connector (screen capture, keyboard/mouse)
- Agent lifecycle management
- Save/load states
- Deploy: Same

### Phase 4: Production (1 week)
- Docker containerization
- Multi-user support
- Cloud deployment
- Database persistence

---

## Integration with predictive-coding-agent Repo

**Option 1: Submodule**
```bash
cd predictive-coding-agent
git submodule add https://github.com/USER/pc-agent-studio tools/studio
```

**Option 2: Pip Install**
```bash
pip install pc-agent-studio
pc-studio serve  # Starts server
```

**Option 3: Import Models**
```python
# In pc-agent-studio/backend/models/pc_network.py
import sys
sys.path.append('/path/to/predictive-coding-agent')
from experiments.categorical_pc.categorical_network_impl import (
    CategoricalPCNetwork,
    CanonicalMicrocircuit
)
```

---

## API Reference (Key Endpoints)

### Network
- `GET /api/network/{id}/structure` - Get network graph
- `GET /api/network/{id}/weights` - Get weight statistics
- `POST /api/network/create` - Create new network from config

### Agent
- `POST /api/agent/start` - Start agent
- `POST /api/agent/stop` - Stop agent
- `GET /api/agent/{id}/status` - Get agent status
- `GET /api/agent/{id}/metrics` - Get performance metrics

### Training
- `POST /api/training/start` - Start training
- `POST /api/training/pause` - Pause training
- `GET /api/training/{id}/history` - Get training history

### CTKG
- `POST /api/ctkg/validate` - Validate CTKG
- `POST /api/ctkg/generate_network` - Generate network from CTKG

### WebSocket Events
- `activity_update` - Layer activations
- `training_progress` - Training metrics
- `inference_step` - Inference iteration details

---

## Next Steps to Build

1. **Create new repository:** `pc-agent-studio`
2. **Start with MVP:**
   - Single Python file (FastAPI)
   - Single HTML page (D3.js)
   - WebSocket connection
   - Load MNIST training script
3. **Test integration:** Import from predictive-coding-agent
4. **Iterate:** Add features based on usage

**You can develop this in a separate Claude chat by:**
1. Providing this spec
2. Starting with MVP
3. Testing incrementally

**I can use it by:**
1. Running `python app.py`
2. Open browser to `localhost:8000`
3. Watch training in real-time

Would you like me to create the MVP starter code now?
