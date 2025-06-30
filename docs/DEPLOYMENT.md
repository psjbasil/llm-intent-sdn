# LLM Intent-based SDN Deployment Guide

This is a complete deployment guide to set up and run the LLM intent-driven SDN network management system.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Ubuntu 20.04+ / CentOS 8+ / macOS 10.15+
- **Python**: 3.10 or higher
- **Memory**: At least 4GB RAM
- **Network**: Internet connection for LLM API access

### Required Software
- **Python 3.10+**: Programming language runtime
- **Mininet**: Network emulation platform
- **Open vSwitch**: Virtual switching platform
- **RYU**: SDN controller framework

## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd llm-intent-sdn

# Run setup script
python setup.py
```

### 2. Configure Environment
```bash
# Copy environment template
cp env.example .env

# Edit configuration (IMPORTANT!)
nano .env
```

**Required Configuration:**
- `LLM_SDN_OPENAI_API_KEY`: Your DeepSeek API key from OpenRouter
- Other settings can use default values for testing

### 3. Start System Components

#### Terminal 1: Start RYU Controller
```bash
python scripts/start_ryu.py
```

#### Terminal 2: Start Mininet (requires sudo)
```bash
sudo python scripts/start_mininet.py
```

#### Terminal 3: Start API Server
```bash
python run.py
```

### 4. Verify Installation
```bash
python scripts/test_system.py
```

## 📖 Detailed Setup

### Step 1: Install System Dependencies

#### Ubuntu/Debian
```bash
# Update package manager
sudo apt update

# Install Python and pip
sudo apt install python3.10 python3.10-pip python3.10-venv

# Install Mininet
sudo apt install mininet

# Install Open vSwitch
sudo apt install openvswitch-switch
```

#### CentOS/RHEL
```bash
# Install Python
sudo dnf install python3.10 python3.10-pip

# Install Mininet (may require building from source)
sudo dnf install mininet

# Install Open vSwitch
sudo dnf install openvswitch
```

#### macOS
```bash
# Install Python using Homebrew
brew install python@3.10

# Note: Mininet on macOS requires special setup or VM
```

### Step 2: Setup Python Environment
```bash
# Create virtual environment
python3.10 -m venv llm-sdn-env

# Activate environment
source llm-sdn-env/bin/activate  # Linux/macOS
# llm-sdn-env\Scripts\activate     # Windows

# Install project
python setup.py
```

### Step 3: Configure Services

#### Environment Variables (.env)
```bash
# API Configuration
LLM_SDN_API_HOST=0.0.0.0
LLM_SDN_API_PORT=8000
LLM_SDN_API_DEBUG=false

# RYU Controller
LLM_SDN_RYU_HOST=127.0.0.1
LLM_SDN_RYU_PORT=8080

# LLM Configuration (REQUIRED)
LLM_SDN_OPENAI_API_KEY=your_deepseek_api_key_here
LLM_SDN_OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_SDN_LLM_MODEL=deepseek/deepseek-chat

# Logging
LLM_SDN_LOG_LEVEL=INFO
LLM_SDN_LOG_FILE=logs/llm_intent_sdn.log
```

### Step 4: Start Services in Order

#### 1. Start RYU Controller
```bash
# Method 1: Using provided script
python scripts/start_ryu.py

# Method 2: Manual command
ryu-manager ryu.app.rest_topology ryu.app.ws_topology ryu.app.ofctl_rest ryu.app.rest_conf_switch ryu.app.simple_switch_13
```

**Expected Output:**
```
loading app ryu.app.rest_topology
loading app ryu.app.ws_topology
loading app ryu.app.ofctl_rest
instantiating app ryu.app.rest_topology
instantiating app ryu.app.ws_topology
instantiating app ryu.app.ofctl_rest
```

#### 2. Start Mininet Network
```bash
# Using provided script (recommended)
sudo python scripts/start_mininet.py

# Manual Mininet setup
sudo mn --topo single,3 --controller remote,ip=127.0.0.1,port=6653 --switch ovsk,protocols=OpenFlow13
```

**Expected Output:**
```
*** Creating network
*** Adding controller
*** Adding hosts:
h1 h2 h3 h4
*** Adding switches:
s1 s2 s3
*** Adding links:
(h1, s1) (h2, s1) (h3, s2) (h4, s3) (s1, s2) (s1, s3) (s2, s3)
*** Configuring hosts
*** Starting controller
c0
*** Starting 3 switches
s1 s2 s3 ...
*** Starting CLI:
mininet>
```

#### 3. Start API Server
```bash
# Using provided script
python run.py

# Manual uvicorn command
uvicorn llm_intent_sdn.api.main:app --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 🔍 Verification and Testing

### Health Check
```bash
# Check API health
curl http://localhost:8000/health

# Check RYU controller
curl http://localhost:8080/stats/switches

# Run comprehensive tests
python scripts/test_system.py
```

### API Documentation
Visit: http://localhost:8000/docs

### Test Intent Processing
```bash
curl -X POST "http://localhost:8000/api/v1/intent/process" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Block traffic between host 10.0.0.1 and 10.0.0.2",
       "context": "Security policy"
     }'
```

## 🐛 Troubleshooting

### Common Issues

#### 1. RYU Controller Connection Failed
**Problem**: Cannot connect to RYU controller
**Solution**:
```bash
# Check if RYU is running
ps aux | grep ryu

# Check port availability
netstat -tulpn | grep 8080

# Restart RYU
pkill -f ryu
python scripts/start_ryu.py
```

#### 2. Mininet Permission Denied
**Problem**: Mininet requires root privileges
**Solution**:
```bash
# Run with sudo
sudo python scripts/start_mininet.py

# Or add user to sudoers for mininet commands
sudo visudo
# Add: username ALL=(ALL) NOPASSWD: /usr/bin/mn
```

#### 3. LLM API Key Error
**Problem**: Invalid or missing API key
**Solution**:
```bash
# Check .env file
cat .env | grep API_KEY

# Get API key from OpenRouter
# Visit: https://openrouter.ai/keys
# Add to .env file
```

#### 4. Port Already in Use
**Problem**: Port 8000 or 8080 already occupied
**Solution**:
```bash
# Find process using port
sudo lsof -i :8000
sudo lsof -i :8080

# Kill process or change port in .env
```

### Log Analysis
```bash
# View API logs
tail -f logs/llm_intent_sdn.log

# View RYU logs
journalctl -u ryu-manager -f

# View system logs
dmesg | grep ovs
```

## 🔧 Configuration Options

### API Configuration
- `LLM_SDN_API_HOST`: API server bind address
- `LLM_SDN_API_PORT`: API server port
- `LLM_SDN_API_DEBUG`: Enable debug mode

### RYU Configuration
- `LLM_SDN_RYU_HOST`: RYU controller address
- `LLM_SDN_RYU_PORT`: RYU REST API port

### LLM Configuration
- `LLM_SDN_OPENAI_API_KEY`: API key for LLM service
- `LLM_SDN_LLM_MODEL`: Model name (deepseek/deepseek-chat)
- `LLM_SDN_LLM_TEMPERATURE`: Model temperature (0.1)

### Network Configuration
- `LLM_SDN_MININET_HOST`: Mininet host address
- `LLM_SDN_MININET_PORT`: OpenFlow controller port

## 📊 Monitoring and Maintenance

### Health Monitoring
```bash
# API health
curl http://localhost:8000/health

# System metrics
curl http://localhost:8000/api/v1/monitoring/health

# Network stats
curl http://localhost:8000/api/v1/monitoring/stats
```

### Log Rotation
```bash
# Setup logrotate for application logs
sudo tee /etc/logrotate.d/llm-intent-sdn << EOF
/path/to/logs/llm_intent_sdn.log {
    daily
    missingok
    rotate 30
    compress
    notifempty
    copytruncate
}
EOF
```

### Backup Configuration
```bash
# Backup configuration files
cp .env .env.backup
cp -r logs logs.backup

# Backup network topology
curl http://localhost:8000/api/v1/network/topology > topology_backup.json
```

## 🔄 Production Deployment

### Using Docker (Optional)
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -e .
CMD ["python", "run.py"]
```

### Systemd Service
```ini
[Unit]
Description=LLM Intent-based SDN API
After=network.target

[Service]
Type=simple
User=llm-sdn
WorkingDirectory=/opt/llm-intent-sdn
ExecStart=/opt/llm-intent-sdn/llm-sdn-env/bin/python run.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Reverse Proxy (Nginx)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```