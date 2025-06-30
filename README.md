# Intent-based Networking with LLMs

This project integrates Large Language Models (LLMs) into Software Defined Networking (SDN) for intent-based network management. It enables users to control and configure SDN infrastructure (via RYU controller) using natural language intents.

## Features
- **Natural Language Intent Processing**: Parse and understand network operation intents using DeepSeek LLM.
- **Automatic Flow Rule Generation**: Generate OpenFlow rules for network policies.
- **Real-time Network Monitoring**: Monitor status, traffic, and anomalies.
- **RESTful API**: Full API with OpenAPI documentation.
- **SDN Controller Integration**: Seamless with RYU controller.
- **Topology Management**: Auto-discover and manage network topology.

## Project Structure
```
llm-intent-sdn/
├── src/llm_intent_sdn/          # Main source code
│   ├── api/                     # API routes
│   ├── models/                  # Data models
│   ├── services/                # Business logic
│   └── utils/                   # Utilities
├── scripts/                     # Startup & management scripts
├── examples/                    # Usage examples
├── docs/                        # Documentation
```

## Getting Started

### Requirements
- Python 3.10+
- Ubuntu 20.04+ / CentOS 8+ / macOS 10.15+
- 4GB+ RAM

### Installation
```bash
# Clone the repo
git clone <repository-url>
cd llm-intent-sdn

# Install dependencies
python setup.py

# Configure environment variables
cp env.example .env
# Edit .env and add your DeepSeek API key
```

### Running the System
```bash
# Terminal 1: Start RYU controller
python scripts/start_ryu.py

# Terminal 2: Start Mininet topology (sudo required)
sudo python scripts/start_mininet.py

# Terminal 3: Start API server
python run.py
```

## Testing
```bash
# Run system tests
python scripts/test_system.py

# Or run example intents
python examples/intent_examples.py
```

## API Docs
Visit http://localhost:8000/docs for interactive API documentation.



