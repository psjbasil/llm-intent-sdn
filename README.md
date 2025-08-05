# Intent-based Networking with LLMs

This project integrates Large Language Models (LLMs) into Software Defined Networking (SDN) for intent-based network management. It enables users to control and configure SDN infrastructure (via RYU controller) using natural language intents.

## Features
- **Natural Language Intent Processing**: Parse and understand network operation intents using DeepSeek LLM.
- **Automatic Flow Rule Generation**: Generate OpenFlow rules for network policies.
- **Real-time Network Monitoring**: Monitor status, traffic, and anomalies.
- **RESTful API**: Full API with OpenAPI documentation.
- **SDN Controller Integration**: Seamless with RYU controller.
- **Topology Management**: Auto-discover and manage network topology.
- **Web UI Interface**: Modern web interface for visualizing and managing all functions.

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
# Edit .env and add your DeepSeek API key
```

### Running the System

#### Using Web UI
```bash
# Terminal 1: Start RYU controller
python scripts/start_ryu.py

# Terminal 2: Start Mininet topology (sudo required)
sudo python scripts/start_mininet.py

# Terminal 3: Start API server with Web UI
python run.py
# This will automatically open http://localhost:8000/static/index.html
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

## Testing

### Using Web UI
1. Access the Web UI at http://localhost:8000/static/index.html
2. Try the intent examples or enter your own natural language commands
3. Monitor network topology and flow rules in real-time
4. Check system health and anomalies

### Using Scripts
```bash
# Run system tests
python scripts/test_system.py

# Or run example intents
python examples/intent_examples.py
```

## Intent Examples

Try these examples in the Web UI:

- **Security**: "Block all traffic from host 10.0.0.1 to host 10.0.0.2"
- **Routing**: "Route traffic from 10.0.0.1 to 10.0.0.3 through the fastest path"
- **Monitoring**: "Monitor bandwidth usage on all network links"
- **QoS**: "Set QoS priority for video traffic on port 80"

## API Documentation

- **Web UI**: http://localhost:8000/static/index.html
- **API Docs**: http://localhost:8000/docs  
- **Alternative API Docs**: http://localhost:8000/redoc



