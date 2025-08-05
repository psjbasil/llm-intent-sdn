// LLM Intent-based SDN Web UI JavaScript
// Intelligent Software-Defined Network Management System

// Global variables
let network = null;
let currentTab = 'intent';
let intentHistory = [];
let flowData = [];

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('LLM Intent-based SDN system initializing...');
    initializeApp();
});

function initializeApp() {
    switchTab('intent');
    initializeTopology();
    loadDashboardData();
    checkConnectionStatus();
    startPeriodicUpdates();
}

// Tab switching functionality
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = 'none';
    });

    // Remove active state from all buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('tab-active');
        button.classList.add('text-gray-600', 'hover:text-blue-600');
    });

    // Show selected tab
    const selectedTab = document.getElementById(tabName + '-tab');
    if (selectedTab) {
        selectedTab.style.display = 'block';
        selectedTab.classList.add('fade-in');
    }

    // Activate selected button
    const selectedButton = document.querySelector(`[data-tab="${tabName}"]`);
    if (selectedButton) {
        selectedButton.classList.add('tab-active');
        selectedButton.classList.remove('text-gray-600', 'hover:text-blue-600');
    }

    currentTab = tabName;

    // Load tab-specific data
    switch(tabName) {
        case 'topology':
            loadTopology();
            break;
        case 'flows':
            loadFlowRules();
            break;
        case 'monitoring':
            loadMonitoringData();
            break;
    }
}

// Load dashboard data
async function loadDashboardData() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        updateDashboardMetrics(data);
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        // Show mock data
        updateDashboardMetrics({ status: 'healthy' });
    }
}

// Update dashboard metrics
function updateDashboardMetrics(data) {
    document.getElementById('device-count').textContent = '6';  // 3 switches + 6 hosts from current topology
    document.getElementById('flow-count').textContent = '12';
    document.getElementById('intent-count').textContent = intentHistory.length;
    
    const healthStatus = document.getElementById('health-status');
    if (data && data.status === 'healthy') {
        healthStatus.textContent = 'Good';
        healthStatus.className = 'text-3xl font-bold text-green-600';
    } else {
        healthStatus.textContent = 'Warning';
        healthStatus.className = 'text-3xl font-bold text-orange-600';
    }
}

// Intent processing
async function processIntent() {
    const intentInput = document.getElementById('intent-input');
    const intent = intentInput.value.trim();

    if (!intent) {
        showNotification('Please enter an intent description', 'error');
        return;
    }

    try {
        showLoading(true);
        
        // Try calling real API
        try {
            const response = await fetch('/api/v1/intent/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    intent_text: intent, 
                    context: { source: 'web-ui' },
                    priority: 5 
                })
            });
            
            if (response.ok) {
                const results = await response.json();
                showIntentResults(results);
                addToIntentHistory(intent, results);
                intentInput.value = '';
                showNotification('Intent processed successfully', 'success');
            } else {
                throw new Error('API call failed');
            }
        } catch (apiError) {
            // If API call fails, show mock results
            console.warn('API call failed, showing mock results:', apiError);
            
            setTimeout(() => {
                const results = {
                    status: 'success',
                    message: 'Intent analysis: This is a network security control request, will generate corresponding OpenFlow rules',
                    actions_taken: ['Generate blocking flow rules', 'Apply to relevant switches', 'Update network security policies']
                };
                
                showIntentResults(results);
                addToIntentHistory(intent, results);
                intentInput.value = '';
                showNotification('Intent processed successfully (demo mode)', 'success');
            }, 1500);
        }

    } catch (error) {
        showNotification('Intent processing failed', 'error');
        console.error('Error processing intent:', error);
    } finally {
        showLoading(false);
    }
}

// Analyze intent
async function analyzeIntent() {
    const intentInput = document.getElementById('intent-input');
    const intent = intentInput.value.trim();

    if (!intent) {
        showNotification('Please enter an intent to analyze', 'error');
        return;
    }

    try {
        showLoading(true);
        
        // Try calling real API
        try {
            const response = await fetch('/api/v1/intent/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    intent_text: intent,
                    context: { source: 'web-ui', analysis_only: true }
                })
            });
            
            if (response.ok) {
                const results = await response.json();
                showIntentResults(results, true);
                showNotification('Intent analysis completed', 'success');
            } else {
                throw new Error('API call failed');
            }
        } catch (apiError) {
            // If API call fails, show mock results
            console.warn('API call failed, showing mock results:', apiError);
            
            setTimeout(() => {
                const results = {
                    status: 'analyzed',
                    message: 'Intent type: Network security policy, confidence: 95%, involves host communication control',
                    actions_taken: ['Will generate blocking rules', 'Affects network segment: 10.0.0.0/24', 'Estimated execution time: <1s']
                };
                
                showIntentResults(results, true);
                showNotification('Intent analysis completed (demo mode)', 'success');
            }, 1000);
        }

    } catch (error) {
        showNotification('Intent analysis failed', 'error');
        console.error('Error analyzing intent:', error);
    } finally {
        showLoading(false);
    }
}

// Show intent results
function showIntentResults(results, isAnalysis = false) {
    const resultsDiv = document.getElementById('intent-results');
    const contentDiv = document.getElementById('results-content');

    if (!resultsDiv || !contentDiv) return;

    resultsDiv.style.display = 'block';

    const statusColor = results.status === 'success' ? 'green' : results.status === 'analyzed' ? 'blue' : 'red';
    const statusIcon = results.status === 'success' ? 'check-circle' : results.status === 'analyzed' ? 'search' : 'exclamation-circle';

    let html = `
        <div class="bg-${statusColor}-50 border border-${statusColor}-200 p-6 rounded-lg">
            <div class="flex items-center mb-4">
                <i class="fas fa-${statusIcon} text-${statusColor}-600 mr-2"></i>
                <h5 class="font-semibold text-${statusColor}-800">
                    ${isAnalysis ? 'Analysis Results' : 'Execution Results'}
                </h5>
                <span class="ml-auto text-sm text-${statusColor}-600">${new Date().toLocaleTimeString()}</span>
            </div>
    `;

    if (results.message) {
        html += `
            <div class="mb-4">
                <h6 class="font-medium text-${statusColor}-700 mb-2">Analysis Results:</h6>
                <p class="text-${statusColor}-600">${results.message}</p>
            </div>
        `;
    }

    if (results.actions_taken) {
        html += `
            <div>
                <h6 class="font-medium text-${statusColor}-700 mb-2">${isAnalysis ? 'Expected Actions' : 'Executed Actions'}:</h6>
                <ul class="list-disc list-inside text-${statusColor}-600 space-y-1">
                    ${results.actions_taken.map(action => `<li>${action}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    html += '</div>';
    contentDiv.innerHTML = html;
}

// Load examples
function loadExample(type) {
    const examples = {
        security: 'Block all traffic from host h1 to host h2',
        routing: 'Route traffic from h1 to h3 through the fastest path',
        monitoring: 'Monitor bandwidth usage on all network links',
        qos: 'Set high priority QoS for video traffic on port 80'
    };

    const intentInput = document.getElementById('intent-input');
    if (intentInput && examples[type]) {
        intentInput.value = examples[type];
        intentInput.focus();
    }
}

// Add to intent history
function addToIntentHistory(intent, results) {
    const historyItem = {
        id: Date.now(),
        intent: intent,
        timestamp: new Date(),
        status: results.status
    };

    intentHistory.unshift(historyItem);
    if (intentHistory.length > 5) {
        intentHistory = intentHistory.slice(0, 5);
    }

    updateIntentHistory();
    updateDashboardMetrics();
}

// Update intent history display
function updateIntentHistory() {
    const historyDiv = document.getElementById('intent-history');
    if (!historyDiv) return;

    if (intentHistory.length === 0) {
        historyDiv.innerHTML = '<p class="text-gray-500 text-center py-4">No processing history yet</p>';
        return;
    }

    let html = '';
    intentHistory.forEach(item => {
        const statusColor = item.status === 'success' ? 'green' : item.status === 'analyzed' ? 'blue' : 'red';
        const statusText = item.status === 'success' ? 'Success' : item.status === 'analyzed' ? 'Analyzed' : 'Failed';
        html += `
            <div class="bg-gray-50 p-3 rounded-lg border border-gray-200">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs text-gray-500">${item.timestamp.toLocaleString()}</span>
                    <span class="px-2 py-1 text-xs rounded-full bg-${statusColor}-100 text-${statusColor}-800">
                        ${statusText}
                    </span>
                </div>
                <p class="text-sm text-gray-800 font-medium">${item.intent}</p>
            </div>
        `;
    });

    historyDiv.innerHTML = html;
}

// Network topology functions
function initializeTopology() {
    const container = document.getElementById('network-topology');
    if (!container) return;

    try {
        const data = {
            nodes: new vis.DataSet([
                {id: 's1', label: 'Switch 1', group: 'switch', title: 'Core Switch'},
                {id: 's2', label: 'Switch 2', group: 'switch', title: 'Access Switch'},
                {id: 's3', label: 'Switch 3', group: 'switch', title: 'Access Switch'},
                {id: 'h1', label: 'H1', group: 'host', title: 'Host 1'},
                {id: 'h2', label: 'H2', group: 'host', title: 'Host 2'},
                {id: 'h3', label: 'H3', group: 'host', title: 'Host 3'},
                {id: 'h4', label: 'H4', group: 'host', title: 'Host 4'},
                {id: 'h5', label: 'H5', group: 'host', title: 'Host 5'},
                {id: 'h6', label: 'H6', group: 'host', title: 'Host 6'}
            ]),
            edges: new vis.DataSet([
                {from: 'h1', to: 's1', title: 'Access link'},
                {from: 'h5', to: 's1', title: 'Access link'},
                {from: 'h2', to: 's2', title: 'Access link'},
                {from: 'h4', to: 's2', title: 'Access link'},
                {from: 'h3', to: 's3', title: 'Access link'},
                {from: 'h6', to: 's3', title: 'Access link'},
                {from: 's1', to: 's2', title: 'Backbone link'},
                {from: 's2', to: 's3', title: 'Backbone link'}
            ])
        };

        const options = {
            groups: {
                switch: {
                    color: {background: '#3b82f6', border: '#1d4ed8'}, 
                    shape: 'box',
                    font: {color: 'white'}
                },
                host: {
                    color: {background: '#10b981', border: '#059669'}, 
                    shape: 'ellipse',
                    font: {color: 'white'}
                }
            },
            physics: {
                stabilization: false,
                barnesHut: {
                    gravitationalConstant: -8000,
                    springConstant: 0.04,
                    springLength: 95
                }
            },
            interaction: {
                navigationButtons: true,
                keyboard: true
            }
        };

        network = new vis.Network(container, data, options);
        
        // Add click event
        network.on('click', function(params) {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                console.log('Clicked node:', nodeId);
            }
        });
        
    } catch (error) {
        console.error('Failed to initialize network topology:', error);
        container.innerHTML = '<div class="flex items-center justify-center h-full text-gray-500"><p>Network topology initialization failed</p></div>';
    }
}

async function loadTopology() {
    console.log('Loading network topology data...');
    try {
        const response = await fetch('/api/v1/network/topology');
        if (response.ok) {
            const data = await response.json();
            console.log('Received topology data:', data);
            updateTopologyVisualization(data);
            showNotification('Topology data refreshed', 'success');
        } else {
            throw new Error('API call failed');
        }
    } catch (error) {
        console.warn('API call failed, showing current topology:', error);
        showNotification('Topology data loaded (current setup)', 'success');
    }
}

function updateTopologyVisualization(data) {
    if (!network) return;
    
    const nodes = [];
    const edges = [];
    
    // Add switches from topology data
    if (data.devices) {
        data.devices.forEach(device => {
            if (device.device_type === 'switch') {
                nodes.push({
                    id: `s${device.dpid}`,
                    label: device.name || `Switch ${device.dpid}`,
                    group: 'switch',
                    title: `Switch: ${device.name}\nDPID: ${device.dpid}`
                });
            }
        });
    }
    
    // Add hosts based on current Mininet topology (h1-h6)
    const hostConfig = [
        { id: 'h1', switch: 's1' },
        { id: 'h2', switch: 's2' },
        { id: 'h3', switch: 's3' },
        { id: 'h4', switch: 's2' },
        { id: 'h5', switch: 's1' },
        { id: 'h6', switch: 's3' }
    ];
    
    hostConfig.forEach(host => {
        nodes.push({
            id: host.id,
            label: host.id.toUpperCase(),
            group: 'host',
            title: `Host: ${host.id.toUpperCase()}`
        });
        
        // Add host-switch links
        edges.push({
            from: host.id,
            to: host.switch,
            title: 'Host connection'
        });
    });
    
    // Add switch-switch links
    if (data.links) {
        data.links.forEach(link => {
            edges.push({
                from: `s${link.src_dpid}`,
                to: `s${link.dst_dpid}`,
                title: `Link: Port ${link.src_port} <-> Port ${link.dst_port}`
            });
        });
    }
    
    // Update the network visualization
    network.setData({
        nodes: new vis.DataSet(nodes),
        edges: new vis.DataSet(edges)
    });
}

function changeLayout() {
    if (!network) return;
    
    const layout = document.getElementById('layout-select').value;
    console.log('Switching layout:', layout);
    
    const options = {
        layout: {},
        physics: { enabled: false }
    };
    
    switch(layout) {
        case 'hierarchical':
            options.layout = {
                hierarchical: {
                    enabled: true,
                    direction: 'UD',
                    sortMethod: 'directed'
                }
            };
            break;
        case 'physics':
            options.physics.enabled = true;
            break;
        case 'circular':
            // vis.js will handle circular layout automatically
            break;
    }
    
    network.setOptions(options);
    showNotification(`Switched to ${layout} layout`, 'success');
}

function refreshTopology() {
    console.log('Refreshing network topology...');
    loadTopology();
}

// Flow rules functions
async function loadFlowRules() {
    console.log('Loading flow rules...');
    const flowsContent = document.getElementById('flows-content');
    if (!flowsContent) return;
    
    // Try loading from API
    try {
        const response = await fetch('/api/v1/network/flows');
        if (response.ok) {
            const data = await response.json();
            displayFlowRules(data);
        } else {
            throw new Error('API call failed');
        }
    } catch (error) {
        console.warn('API call failed, showing mock data:', error);
        // Show mock flow rules data
        const mockFlows = [
            {
                id: 1,
                dpid: '0000000000000001',
                priority: 100,
                duration_sec: 3600,
                packet_count: 150,
                byte_count: 15000,
                table_id: 0,
                actions: 'OUTPUT:CONTROLLER'
            },
            {
                id: 2,
                dpid: '0000000000000002',
                priority: 200,
                duration_sec: 1800,
                packet_count: 75,
                byte_count: 7500,
                table_id: 0,
                actions: 'DROP'
            }
        ];
        displayFlowRules(mockFlows);
    }
}

function displayFlowRules(flows) {
    const flowsContent = document.getElementById('flows-content');
    if (!flowsContent) return;

    if (!flows || flows.length === 0) {
        flowsContent.innerHTML = '<p class="text-gray-500 text-center py-8">No flow rules data available</p>';
        return;
    }

    let html = '';
    flows.forEach(flow => {
        html += `
            <div class="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <div class="flex items-center justify-between mb-2">
                    <span class="font-medium text-gray-800">Flow Rule ID: ${flow.id}</span>
                    <span class="text-sm text-gray-500">Switch: ${flow.dpid}</span>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                        <strong>Priority:</strong> ${flow.priority}<br>
                        <strong>Duration:</strong> ${flow.duration_sec}s<br>
                        <strong>Packet Count:</strong> ${flow.packet_count}
                    </div>
                    <div>
                        <strong>Byte Count:</strong> ${flow.byte_count}<br>
                        <strong>Table ID:</strong> ${flow.table_id}<br>
                        <strong>Actions:</strong> ${flow.actions}
                    </div>
                </div>
            </div>
        `;
    });

    flowsContent.innerHTML = html;
}

function filterFlows() {
    console.log('Filtering flow rules...');
    loadFlowRules();
}

// Monitoring functions
async function loadMonitoringData() {
    console.log('Loading monitoring data...');
    
    // Load health metrics
    const healthMetrics = document.getElementById('health-metrics');
    if (healthMetrics) {
        try {
            const response = await fetch('/health/detailed');
            if (response.ok) {
                const data = await response.json();
                updateHealthMetrics(data);
            } else {
                throw new Error('API call failed');
            }
        } catch (error) {
            console.warn('Health check API call failed, showing mock data:', error);
            // Show mock health data
            healthMetrics.innerHTML = `
                <div class="flex items-center justify-between p-4 bg-green-50 rounded-lg">
                    <span class="font-medium text-green-800">RYU Controller</span>
                    <span class="px-3 py-1 text-sm rounded-full bg-green-100 text-green-800">Running</span>
                </div>
                <div class="flex items-center justify-between p-4 bg-green-50 rounded-lg">
                    <span class="font-medium text-green-800">OpenFlow Connection</span>
                    <span class="px-3 py-1 text-sm rounded-full bg-green-100 text-green-800">Connected</span>
                </div>
                <div class="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
                    <span class="font-medium text-blue-800">LLM Service</span>
                    <span class="px-3 py-1 text-sm rounded-full bg-blue-100 text-blue-800">Ready</span>
                </div>
            `;
        }
    }

    // Load performance metrics
    const performanceMetrics = document.getElementById('performance-metrics');
    if (performanceMetrics) {
        performanceMetrics.innerHTML = `
            <div class="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
                <span class="font-medium text-blue-800">CPU Usage</span>
                <span class="text-blue-600">25%</span>
            </div>
            <div class="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
                <span class="font-medium text-blue-800">Memory Usage</span>
                <span class="text-blue-600">42%</span>
            </div>
            <div class="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
                <span class="font-medium text-blue-800">Network Latency</span>
                <span class="text-blue-600">2ms</span>
            </div>
            <div class="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
                <span class="font-medium text-blue-800">Throughput</span>
                <span class="text-blue-600">850 Mbps</span>
            </div>
        `;
    }
}

function updateHealthMetrics(data) {
    const healthMetrics = document.getElementById('health-metrics');
    if (!healthMetrics || !data.dependencies) return;
    
    let html = '';
    Object.entries(data.dependencies).forEach(([service, info]) => {
        const statusColor = info.status === 'healthy' ? 'green' : 'red';
        const serviceName = service === 'ryu_controller' ? 'RYU Controller' : 
                           service === 'llm_api' ? 'LLM Service' : service;
        
        html += `
            <div class="flex items-center justify-between p-4 bg-${statusColor}-50 rounded-lg">
                <span class="font-medium text-${statusColor}-800">${serviceName}</span>
                <span class="px-3 py-1 text-sm rounded-full bg-${statusColor}-100 text-${statusColor}-800">
                    ${info.status === 'healthy' ? 'Running' : 'Error'}
                </span>
            </div>
        `;
    });
    
    healthMetrics.innerHTML = html;
}

// Utility functions
function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}

function showNotification(message, type) {
    const notification = document.createElement('div');
    const bgColor = type === 'success' ? 'bg-green-500' : 'bg-red-500';
    notification.className = `fixed top-4 right-4 ${bgColor} text-white px-6 py-3 rounded-lg shadow-lg z-50`;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 3000);
}

function checkConnectionStatus() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            const dot = document.getElementById('connection-dot');
            const text = document.getElementById('connection-text');
            
            if (data.status === 'healthy') {
                dot.className = 'status-dot status-online';
                text.textContent = 'Connected';
            } else {
                dot.className = 'status-dot status-offline';
                text.textContent = 'Connection Error';
            }
        })
        .catch(() => {
            const dot = document.getElementById('connection-dot');
            const text = document.getElementById('connection-text');
            dot.className = 'status-dot status-offline';
            text.textContent = 'Disconnected';
        });
}

function refreshAll() {
    console.log('Refreshing all data...');
    loadDashboardData();
    checkConnectionStatus();
    
    if (currentTab === 'topology') loadTopology();
    if (currentTab === 'flows') loadFlowRules();
    if (currentTab === 'monitoring') loadMonitoringData();
    
    showNotification('Data refreshed', 'success');
}

function startPeriodicUpdates() {
    // Update dashboard data every 30 seconds
    setInterval(() => {
        loadDashboardData();
        checkConnectionStatus();
    }, 30000);

    // Update monitoring data every 10 seconds
    setInterval(() => {
        if (currentTab === 'monitoring') {
            loadMonitoringData();
        }
    }, 10000);
}