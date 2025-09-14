# Intent-based Networking with LLMs

This project integrates Large Language Models (LLMs) into Software Defined Networking (SDN) for intent-based network management. It enables users to control and configure SDN infrastructure (via RYU controller) using natural language intents.

## Features
- **Natural Language Intent Processing**: Parse and understand network operation intents using LLMs (DeepSeek, Ollama, OpenAI).
- **Automatic Flow Rule Generation**: Generate OpenFlow rules for network policies.
- **Real-time Network Monitoring**: Monitor status, traffic, and anomalies.
- **RESTful API**: Full API with OpenAPI documentation.
- **SDN Controller Integration**: Seamless with RYU controller.
- **Topology Management**: Auto-discover and manage network topology.
- **Web UI Interface**: Modern web interface for visualizing and managing all functions.
- **Local LLM Support**: Run models locally with Ollama for privacy and cost savings.

## Project Structure
```
llm-intent-sdn/
├── src/llm_intent_sdn/          # Main source code
│   ├── api/                     # API routes
│   │   └── static/              # Web UI files
│   ├── models/                  # Data models
│   ├── services/                # Business logic
│   └── utils/                   # Utilities
├── scripts/                     # Startup & management scripts
├── examples/                    # Usage examples
├── docs/                        # Documentation
```

## Getting Started

### Requirements
- Python 3.8+
- Ubuntu 20.04+ / CentOS 8+ / macOS 10.15+
- 4GB+ RAM

### Installation
```bash
# Clone the repo
git clone <repository-url>
cd llm-intent-sdn

# Install dependencies (eventlet must be <=0.30.2 for RYU compatibility)
pip install -e .
# If you install eventlet manually, use:
pip install 'eventlet<=0.30.2'  # Required for RYU 4.x

# Configure environment variables
cp env.example .env
# Edit .env and add your DeepSeek API key, or configure Ollama for local models
```

### Running the System

#### Using Web UI
```bash
# Terminal 1: Start RYU controller (REST + Topology APIs)
# Requires ryu >= 4.x. This starts REST stats and topology endpoints used by the backend
ryu-manager --observe-links ryu.app.ofctl_rest ryu.app.rest_topology

# Terminal 2: Start Mininet topology (sudo required)
sudo python scripts/start_mininet.py

# Terminal 3: Start API server with Web UI
python run.py
# Open http://localhost:8000/static/index.html
```

## Web UI Features

The Web UI provides a comprehensive interface for all system functions:

### 1. Dashboard Overview
- Real-time network statistics
- System health indicators  
- Quick status of devices, flows, and intents

### 2. Intent Processing Tab
- **Natural language input**: Enter network intents in plain English
- **Quick examples**: Pre-defined intent templates for common operations
- **Full mesh connectivity**: Use "allow all the hosts to reach each other" for complete network connectivity
- **Analysis mode**: Analyze intents without executing them
- **Real-time results**: View processing results and generated actions
- **Intent history**: Track all processed intents

### 3. Network Topology Tab
- **Interactive visualization**: Visual network topology with vis.js
- **Multiple layouts**: Hierarchical, physics-based, and circular layouts
- **Device information**: Detailed switch and host information
- **Link details**: Network connectivity and port mappings

### 4. Flow Rules Tab
- **Flow table management**: View and filter OpenFlow rules
- **Switch filtering**: Filter rules by specific switches
- **Search functionality**: Find specific flow rules
- **Detailed view**: Complete flow rule information including matches and actions

### 5. Monitoring Tab
- **Network health**: Overall system health indicators
- **Anomaly detection**: Real-time network anomaly alerts
- **Statistics dashboard**: Performance metrics and utilization data
- **Real-time updates**: Automatic refresh of monitoring data

## Full Mesh Connectivity

The system supports automatic full mesh connectivity setup for all hosts in the network. This feature provides deterministic, reliable connectivity without relying on LLM generation.

### Usage
Simply enter one of these intent phrases in the Web UI:
- `allow all the hosts to reach each other`
- `allow all hosts to reach each other`
- `allow all the hosts to connect`
- `allow all hosts to communicate`

### How It Works
1. **Automatic Detection**: The system detects full mesh connectivity intents
2. **Host Discovery**: Automatically discovers all available hosts (h1, h2, h3, etc.)
3. **Path Calculation**: Uses BFS algorithm to compute shortest paths between all host pairs
4. **Flow Rule Installation**: Installs bidirectional flow rules with:
   - `priority=900` (higher than default rules)
   - `cookie=0xA1A1` (for easy management and deletion)
   - `idle_timeout=0` (permanent rules)
5. **Complete Coverage**: Ensures every host can reach every other host

### Benefits
- **Reliability**: Deterministic path calculation, no LLM dependency
- **Performance**: Optimized shortest paths for all pairs
- **Management**: Unified cookie-based rule management
- **Scalability**: Efficient for small to medium networks

### Example
For a 6-host network (h1-h6), this creates 15 bidirectional paths with optimized flow rules on all intermediate switches.

## LLM Options

### Cloud LLMs (DeepSeek, OpenAI)
```bash
# Set your API key in .env
LLM_SDN_OPENAI_API_KEY=your_api_key_here
LLM_SDN_OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_SDN_LLM_MODEL=deepseek/deepseek-chat
```

### Local LLMs (Ollama)
```bash
# Install Ollama and download a model
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2:latest

# Configure in .env
LLM_SDN_USE_OLLAMA=true
LLM_SDN_OLLAMA_HOST=127.0.0.1
LLM_SDN_OLLAMA_PORT=11434
LLM_SDN_OLLAMA_TIMEOUT=60
```

### Google Gemini
```bash
LLM_SDN_USE_GEMINI=true
LLM_SDN_GEMINI_API_KEY=your_gemini_api_key
LLM_SDN_GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
LLM_SDN_GEMINI_MODEL=models/gemini-1.5-flash
```

## Testing

### Using Web UI
1. Access the Web UI at http://localhost:8000/static/index.html
2. Try the intent examples or enter your own natural language commands
3. Monitor network topology and flow rules in real-time
4. Check system health and anomalies

### Notes
- Most workflows are driven from the Web UI. Example scripts may vary by branch; prefer the UI for end‑to‑end testing.

## Intent Examples

Try these examples in the Web UI:

- **Security**: "Block all traffic from host 10.0.0.1 to host 10.0.0.2"
- **Routing**: "Route traffic from 10.0.0.1 to 10.0.0.3 through the fastest path"
- **Monitoring**: "Monitor bandwidth usage on all network links"
- **QoS**: "Set QoS priority for video traffic on port 80"

## Advanced Settings

- Mininet executor (optional): the backend can verify connectivity and bandwidth via a helper executed inside Mininet. It is enabled by default; to disable and use controller‑only heuristics:
```bash
LLM_SDN_USE_MININET_EXECUTOR=false
```

- Timeouts:
  - Ollama requests use `LLM_SDN_OLLAMA_TIMEOUT` (default 60s) for local models
  - Other LLM requests use `intent_timeout` from settings (default 30s)

## QoS Behavior Notes

- QoS rules match TCP destination port only (e.g., `tcp_dst=80`) because clients use ephemeral source ports.
- QoS rules are installed deterministically along the shortest path; they align output ports with existing routing to avoid blackholes.

## API Documentation

- **Web UI**: http://localhost:8000/static/index.html
- **API Docs**: http://localhost:8000/docs  
- **Alternative API Docs**: http://localhost:8000/redoc



