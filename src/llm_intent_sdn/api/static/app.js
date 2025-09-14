// LLM Intent-based SDN Web UI JavaScript
// Intelligent Software-Defined Network Management System

// Global variables
let network = null;
let currentTab = 'intent';
let intentHistory = [];
let flowData = [];
let chatMessages = [];
let selectedCategory = 'connectivity';

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
    // Initialize chat with welcome message
    appendChatMessage('system', 'Welcome. Select a category and describe your intent.');
    updateCategoryButtons();
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

// Chat helpers
function appendChatMessage(role, text) {
    chatMessages.push({ role, text, time: new Date() });
    const box = document.getElementById('chat-messages');
    if (!box) return;
    const isUser = role === 'user';
    const bubble = document.createElement('div');
    bubble.className = `flex ${isUser ? 'justify-end' : 'justify-start'}`;
    const inner = document.createElement('div');
    inner.className = `${isUser ? 'bg-blue-600 text-white' : 'bg-white text-gray-800'} px-3 py-2 rounded-lg shadow border ${isUser ? 'border-blue-700' : 'border-gray-200'}`;
    inner.textContent = text;
    bubble.appendChild(inner);
    box.appendChild(bubble);
    box.scrollTop = box.scrollHeight;
}

function selectCategory(cat) {
    selectedCategory = cat;
    updateCategoryButtons();
    const hints = {
        connectivity: 'Example: Allow h1 to reach h3; Deny h2 -> h5',
        performance: 'Example: Give video conference traffic highest priority from h1 to h2',
        security: 'Example: Monitor anomalies on s1; Block suspicious h5 -> h1'
    };
    appendChatMessage('system', `Category set to ${cat}. ${hints[cat]}`);
}

function updateCategoryButtons() {
    const ids = ['connectivity','performance','security'];
    ids.forEach(id => {
        const btn = document.getElementById(`cat-${id}`);
        if (!btn) return;
        if (id === selectedCategory) {
            btn.className = 'px-3 py-2 rounded-lg border text-sm bg-blue-50 border-blue-200 text-blue-700';
        } else {
            btn.className = 'px-3 py-2 rounded-lg border text-sm border-gray-200 text-gray-700 hover:bg-gray-50';
        }
    });
}

function insertChatTemplate(cat) {
    const templates = {
        connectivity: 'Deny h1 from reaching h2',
        performance: 'Give video conference traffic highest priority from h1 to h2',
        security: 'Monitor for anomalies and block suspicious traffic from h5 to h1'
    };
    const input = document.getElementById('chat-input');
    if (input) input.value = templates[cat] || '';
    selectCategory(cat);
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;
    appendChatMessage('user', text);
    input.value = '';

    // Map category to suggested intent_type
    const categoryToType = {
        connectivity: /allow|permit|reach|connect/i.test(text) ? 'allow' : /deny|block|forbid/i.test(text) ? 'deny' : 'allow',
        performance: 'qos',
        security: /monitor|detect|anomaly/i.test(text) ? 'monitor' : 'deny'
    };
    const intentType = categoryToType[selectedCategory] || 'analyze';

    try {
        showLoading(true);
        const payload = {
            intent_text: text,
            context: { source: 'chat', category: selectedCategory },
            priority: 5
        };
        const url = intentType === 'analyze' ? '/api/v1/intent/analyze' : '/api/v1/intent/process';
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (response.ok) {
            const results = await response.json();
            const summary = summarizeResultsForChat(results);
            appendChatMessage('assistant', summary);
            showIntentResults(results, url.includes('analyze'));
            addToIntentHistory(text, results);
        } else {
            throw new Error('API call failed');
        }
    } catch (e) {
        console.warn('Chat API failed, fallback to demo message', e);
        appendChatMessage('assistant', 'Processing in demo mode. Intent has been analyzed and corresponding actions are suggested.');
    } finally {
        showLoading(false);
    }
}

function summarizeResultsForChat(results) {
    const status = (results && results.status ? String(results.status).toLowerCase() : '');
    const actions = results.applied_actions || results.actions_taken || [];
    const actionText = actions.length ? `Actions: ${actions.join('; ')}` : '';
    const msg = results.llm_interpretation || results.message || '';
    return `Status: ${status || 'ok'}. ${msg} ${actionText}`.trim();
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
                
                // Check actual processing status (normalize first)
                const normalizedStatus = (results && results.status ? String(results.status).toLowerCase() : '');
                console.log('processIntent - final status:', results.status, 'normalized:', normalizedStatus);
                if (normalizedStatus === 'completed' || normalizedStatus === 'success') {
                    showNotification('Intent processed successfully', 'success');
                } else if (normalizedStatus === 'failed') {
                    showNotification(`Intent processing failed: ${results.error_message || 'Unknown error'}`, 'error');
                } else if (normalizedStatus === 'analyzed') {
                    showNotification('Intent analysis completed', 'success');
                } else {
                    showNotification(`Intent status: ${normalizedStatus || 'unknown'}`, 'info');
                }
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

    // Normalize status from backend (enum/string/uppercase) and log for debugging
    const normalizedStatus = (results && results.status ? String(results.status).toLowerCase() : '');
    console.log('showIntentResults - status:', results.status, 'normalized:', normalizedStatus, 'isAnalysis:', isAnalysis);

    // Determine status color and icon based on normalized status and analysis mode
    let statusColor, statusIcon;
    
    if (normalizedStatus === 'completed' || normalizedStatus === 'success') {
        statusColor = 'green';
        statusIcon = 'check-circle';
    } else if (normalizedStatus === 'processing') {
        statusColor = 'yellow';
        statusIcon = 'clock';
    } else if (normalizedStatus === 'analyzed') {
        statusColor = 'blue';
        statusIcon = 'search';
    } else if (normalizedStatus === 'failed') {
        statusColor = 'red';
        statusIcon = 'exclamation-circle';
    } else {
        // Default to green for unknown status
        statusColor = 'green';
        statusIcon = 'check-circle';
    }

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

    // Show LLM interpretation or error message
    if (results.llm_interpretation) {
        html += `
            <div class="mb-4">
                <h6 class="font-medium text-${statusColor}-700 mb-2">Analysis Results:</h6>
                <p class="text-${statusColor}-600">${results.llm_interpretation}</p>
            </div>
        `;
    } else if (results.message) {
        html += `
            <div class="mb-4">
                <h6 class="font-medium text-${statusColor}-700 mb-2">Analysis Results:</h6>
                <p class="text-${statusColor}-600">${results.message}</p>
            </div>
        `;
    }
    
    // Show error message if failed
    if (results.status === 'failed' && results.error_message) {
        html += `
            <div class="mb-4">
                <h6 class="font-medium text-red-700 mb-2">Error Details:</h6>
                <p class="text-red-600">${results.error_message}</p>
            </div>
        `;
    }

    // Show applied actions
    if (results.applied_actions && results.applied_actions.length > 0) {
        html += `
            <div class="mb-4">
                <h6 class="font-medium text-green-700 mb-2">Successfully Applied Actions:</h6>
                <ul class="list-disc list-inside text-green-600 space-y-1">
                    ${results.applied_actions.map(action => `<li>${action}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    // Show installed flow rules (details)
    if (results.flow_rules && results.flow_rules.length > 0) {
        html += `
            <div class="mb-4">
                <h6 class="font-medium text-${statusColor}-700 mb-2">Installed Flow Rules:</h6>
                <div class="overflow-x-auto">
                    <table class="min-w-full text-sm">
                        <thead>
                            <tr class="text-${statusColor}-700">
                                <th class="text-left pr-4">dpid</th>
                                <th class="text-left pr-4">priority</th>
                                <th class="text-left pr-4">match</th>
                                <th class="text-left pr-4">actions</th>
                                <th class="text-left pr-4">timeouts</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${results.flow_rules.map(r => `
                                <tr class="text-${statusColor}-600">
                                    <td class="pr-4">${r.dpid}</td>
                                    <td class="pr-4">${r.priority ?? '-'}</td>
                                    <td class="pr-4">${r.match ? Object.entries(r.match).map(([k,v]) => `${k}=${v}`).join(', ') : '-'}</td>
                                    <td class="pr-4">${r.actions ? r.actions.map(a => `${a.type}${a.port!==undefined?':'+a.port:''}`).join(', ') : '-'}</td>
                                    <td class="pr-4">${(r.idle_timeout ?? 0)}/${(r.hard_timeout ?? 0)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }
    
    // Show failed actions
    if (results.failed_actions && results.failed_actions.length > 0) {
        html += `
            <div class="mb-4">
                <h6 class="font-medium text-red-700 mb-2">Failed Actions:</h6>
                <ul class="list-disc list-inside text-red-600 space-y-1">
                    ${results.failed_actions.map(action => `<li>${action}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    // Fallback for legacy actions_taken field
    if (results.actions_taken && !results.applied_actions) {
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
        qos: 'Give video conference traffic highest priority from h1 to h2'
    };

    const intentInput = document.getElementById('intent-input');
    if (intentInput && examples[type]) {
        intentInput.value = examples[type];
        intentInput.focus();
    }
}

// Add to intent history
function addToIntentHistory(intent, results) {
    const normalizedStatus = (results && results.status ? String(results.status).toLowerCase() : '');
    console.log('addToIntentHistory - status:', results.status, 'normalized:', normalizedStatus);
    console.log('addToIntentHistory - full results object:', results);

    const historyItem = {
        id: results && results.intent_id ? results.intent_id : String(Date.now()),
        intent: intent,
        timestamp: new Date(),
        status: normalizedStatus
    };
    
    console.log('addToIntentHistory - created history item:', historyItem);

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
        // Normalize stored status for safety
        const normalizedStatus = (item && item.status ? String(item.status).toLowerCase() : '');
        let statusColor, statusText, actionButton;
        
        // Debug logging
        console.log(`Debug: Intent ${item.id} - Original status: "${item.status}", Normalized: "${normalizedStatus}"`);
        
        if (normalizedStatus === 'completed' || normalizedStatus === 'success') {
            statusColor = 'green';
            statusText = normalizedStatus === 'completed' ? 'Completed' : 'Success';
            // Show Delete button for completed intents
            actionButton = `<button class="text-xs px-2 py-1 rounded border border-blue-200 text-blue-700 hover:bg-blue-50"
                                    onclick="deleteFlowRules('${item.id}')">Delete</button>`;
            console.log(`Debug: Showing Delete button for ${item.id}`);
        } else if (normalizedStatus === 'analyzed') {
            statusColor = 'blue';
            statusText = 'Analyzed';
            actionButton = `<button class="text-xs px-2 py-1 rounded border border-red-200 text-red-700 hover:bg-red-50"
                                    onclick="cancelIntent('${item.id}')">Cancel</button>`;
        } else if (normalizedStatus === 'processing') {
            statusColor = 'yellow';
            statusText = 'Processing';
            actionButton = `<button class="text-xs px-2 py-1 rounded border border-red-200 text-red-700 hover:bg-red-50"
                                    onclick="cancelIntent('${item.id}')">Cancel</button>`;
        } else if (normalizedStatus === 'deleted') {
            statusColor = 'gray';
            statusText = 'Deleted';
            actionButton = ''; // No action button for deleted intents
        } else {
            statusColor = 'red';
            statusText = 'Failed';
            actionButton = `<button class="text-xs px-2 py-1 rounded border border-red-200 text-red-700 hover:bg-red-50"
                                    onclick="cancelIntent('${item.id}')">Cancel</button>`;
        }
        html += `
            <div class="bg-gray-50 p-3 rounded-lg border border-gray-200">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs text-gray-500">${item.timestamp.toLocaleString()}</span>
                    <div class="flex items-center gap-2">
                        <span class="px-2 py-1 text-xs rounded-full bg-${statusColor}-100 text-${statusColor}-800">
                            ${statusText}
                        </span>
                        ${actionButton}
                    </div>
                </div>
                <p class="text-sm text-gray-800 font-medium">${item.intent}</p>
                <p class="text-xs text-gray-500">ID: ${item.id}</p>
            </div>
        `;
    });

    historyDiv.innerHTML = html;
}

async function cancelIntent(intentId) {
    try {
        showLoading(true);
        const res = await fetch(`/api/v1/intent/${intentId}/cancel`, { method: 'POST' });
        if (!res.ok) throw new Error('Cancel API failed');
        showNotification(`Intent ${intentId} cancelled`, 'success');
        // Update local history status to cancelled
        intentHistory = intentHistory.map(h => h.id === intentId ? { ...h, status: 'cancelled' } : h);
        updateIntentHistory();
    } catch (e) {
        console.error('Cancel intent failed', e);
        showNotification('Cancel intent failed', 'error');
    } finally {
        showLoading(false);
    }
}

async function deleteFlowRules(intentId) {
    try {
        showLoading(true);
        const res = await fetch(`/api/v1/intent/${intentId}/delete`, { method: 'POST' });
        if (!res.ok) throw new Error('Delete API failed');
        showNotification(`Flow rules for intent ${intentId} deleted successfully`, 'success');
        // Update local history status to deleted
        intentHistory = intentHistory.map(h => h.id === intentId ? { ...h, status: 'deleted' } : h);
        updateIntentHistory();
    } catch (e) {
        console.error('Delete flow rules failed', e);
        showNotification('Delete flow rules failed', 'error');
    } finally {
        showLoading(false);
    }
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
                {id: 's4', label: 'Switch 4', group: 'switch', title: 'Access Switch'},
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
                {from: 'h6', to: 's4', title: 'Access link'},
                {from: 's1', to: 's2', title: 'Path 1'},
                {from: 's2', to: 's3', title: 'Path 1'},
                {from: 's1', to: 's4', title: 'Path 2'},
                {from: 's4', to: 's3', title: 'Path 2'}
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
    try {
        const response = await fetch('/api/v1/network/topology');
        if (response.ok) {
            const data = await response.json();
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
    
    // Helper: normalize RYU DPID like "0000...0001" or number to short integer string
    function normalizeSwitchId(dpid) {
        const s = String(dpid || '').trim();
        // If like '0000...0001' or '1', strip leading zeros then fallback to parseInt
        const stripped = s.replace(/^0+/, '');
        if (stripped && /^\d+$/.test(stripped)) return stripped;
        // If hex-like, try parse as int
        if (/^[0-9a-fA-F]+$/.test(s)) {
            try { return String(parseInt(s, 16)); } catch (_) {}
        }
        // If name like 's1', capture numeric suffix
        const m = s.match(/(\d+)$/);
        return m ? m[1] : s;
    }
    
    // Add switches from topology data
    if (data.devices) {
        data.devices.forEach(device => {
            if (device.device_type === 'switch') {
                const sid = normalizeSwitchId(device.dpid);
                nodes.push({
                    id: `s${sid}`,
                    label: device.name && /\d+$/.test(device.name) ? device.name.toUpperCase() : `Switch ${sid}`,
                    group: 'switch',
                    title: `Switch: ${device.name || 's'+sid}\nDPID: ${device.dpid}`
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
        { id: 'h6', switch: 's4' }
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
    
    // Add switch-switch links (merge bidirectional links)
    if (data.links) {
        const linkMap = new Map();
        
        // Process all links and deduplicate
        data.links.forEach(link => {
            const from = `s${normalizeSwitchId(link.src_dpid)}`;
            const to = `s${normalizeSwitchId(link.dst_dpid)}`;
            
            // Create a normalized link identifier (smaller ID first)
            const linkId = from < to ? `${from}-${to}` : `${to}-${from}`;
            
            if (!linkMap.has(linkId)) {
                // Store link with consistent direction (smaller ID first)
                linkMap.set(linkId, {
                    from: from < to ? from : to,
                    to: from < to ? to : from,
                    fromPort: from < to ? link.src_port_no : link.dst_port_no,
                    toPort: from < to ? link.dst_port_no : link.src_port_no
                });
            }
        });
        
        // Create edges from unique links
        linkMap.forEach(linkInfo => {
            edges.push({
                from: linkInfo.from,
                to: linkInfo.to,
                title: `${linkInfo.from} Port ${linkInfo.fromPort} ↔ ${linkInfo.to} Port ${linkInfo.toPort}`,
                arrows: undefined,  // No arrows for undirected links
                smooth: { enabled: false }  // Straight lines for cleaner look
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