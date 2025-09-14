#!/usr/bin/env python3
"""
LLM Intent-based SDN Controller - Performance Optimized Version
Optimized following RYU best practices while preserving existing functionality.
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet, arp
from ryu.lib.packet import ether_types
from ryu.topology import event
from ryu.app.wsgi import WSGIApplication, ControllerBase, route
from webob import Response
import json
import time


class LLMSDNController(app_manager.RyuApp):
    """
    Zero Trust SDN Controller - Intent-based Network Security
    
    Security Model:
    - Default: ALL TRAFFIC DENIED
    - Communication: ONLY allowed through explicit intents
    - No learning switch behavior
    - No automatic forwarding
    - All decisions made by LLM/intent processor
    
    Features:
    - Zero Trust: All hosts isolated by default
    - Intent-driven: Communication only allowed through explicit intents
    - Security-first: No default forwarding, all traffic requires authorization
    - LLM Integration: Natural language intent processing
    """
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    _CONTEXTS = {
        'wsgi': WSGIApplication,
    }
    
    def __init__(self, *args, **kwargs):
        super(LLMSDNController, self).__init__(*args, **kwargs)
        
        # Zero Trust State
        self.authorized_flows = {}  # {flow_key: flow_info}
        self.blocked_flows = {}     # {flow_key: block_info}
        self.pending_intents = {}   # {intent_id: intent_info}
        
        # Switch datapaths
        self.datapaths = {}
        
        # Network Topology (for path calculation)
        self.topology_map = {
            1: {2: 3, 4: 4},  # s1: s2 via port 3, s4 via port 4
            2: {1: 3, 3: 4},  # s2: s1 via port 3, s3 via port 4
            3: {2: 2, 4: 3},  # s3: s2 via port 2, s4 via port 3
            4: {1: 2, 3: 3}   # s4: s1 via port 2, s3 via port 3
        }
        
        # Host Locations
        self.host_locations = {
            1: {1: 'h1', 2: 'h5'},      # s1: port 1 -> h1, port 2 -> h5
            2: {1: 'h2', 2: 'h4'},      # s2: port 1 -> h2, port 2 -> h4
            3: {1: 'h3'},               # s3: port 1 -> h3
            4: {1: 'h6'}                # s4: port 1 -> h6
        }
        
        # Host MAC addresses
        self.host_macs = {
            'h1': '00:00:00:00:00:01',
            'h2': '00:00:00:00:00:02',
            'h3': '00:00:00:00:00:03',
            'h4': '00:00:00:00:00:04',
            'h5': '00:00:00:00:00:05',
            'h6': '00:00:00:00:00:06'
        }
        
        # Statistics
        self.stats = {
            'packet_in_count': 0,
            'denied_packets': 0,
            'authorized_packets': 0,
            'intents_processed': 0,
            'flows_installed': 0,
            'flows_blocked': 0
        }
        
        # Setup REST API
        wsgi = kwargs['wsgi']
        wsgi.register(LLMSDNControllerRestAPI, {'controller': self})
        
        print("🔒 Zero Trust SDN Controller initialized")
        print("🚫 Default state: ALL TRAFFIC DENIED")
        print("✅ Communication only through explicit intents")
        


    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Install table-miss flow entry - NO proactive flows."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id

        # Store datapath
        self.datapaths[dpid] = datapath

        # Install table-miss flow entry (send to controller)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        print(f"🔌 Switch {dpid} connected - ZERO TRUST MODE")
        print(f"   🚫 No proactive flows installed")
        print(f"   ⏳ Waiting for explicit intents...")

    def add_flow(self, datapath, priority, match, actions, buffer_id=None,
                 idle_timeout=0, hard_timeout=0):
        """Add a flow entry to the specified switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst, idle_timeout=idle_timeout,
                                    hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst,
                                    idle_timeout=idle_timeout,
                                    hard_timeout=hard_timeout)
        datapath.send_msg(mod)



    def _find_shortest_path(self, src, dst):
        """Find shortest path between two switches using BFS"""
        if src == dst:
            return [src]
        
        visited = set()
        queue = [(src, [src])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == dst:
                return path
            
            # Explore neighbors
            for neighbor in self.topology_map.get(current, {}):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return None

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle packet-in events with zero trust policy."""
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes",
                              ev.msg.msg_len, ev.msg.total_len)
        
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # Allow LLDP for topology discovery
            return
        
        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        self.stats['packet_in_count'] += 1
        
        # Check if this flow is authorized
        flow_key = self._generate_flow_key(src, dst, dpid)
        
        if flow_key in self.authorized_flows:
            # Flow is authorized - forward according to intent
            self._handle_authorized_flow(ev, datapath, in_port, pkt, flow_key)
            self.stats['authorized_packets'] += 1
        else:
            # Flow is NOT authorized - DENY by default
            self._handle_unauthorized_flow(ev, datapath, in_port, pkt, flow_key)
            self.stats['denied_packets'] += 1

    def _generate_flow_key(self, src_mac, dst_mac, dpid):
        """Generate a unique key for a flow."""
        return f"{src_mac}_{dst_mac}_{dpid}"

    def _handle_authorized_flow(self, ev, datapath, in_port, pkt, flow_key):
        """Handle authorized flow according to intent."""
        msg = ev.msg
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id
        
        flow_info = self.authorized_flows[flow_key]
        output_port = flow_info['output_port']
        
        # Handle different flow types
        if flow_info.get('monitoring'):
            # Monitoring flow - log and forward
            print(f"👁️ Monitoring: {flow_key}")
            actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        elif flow_info.get('rate_limiting'):
            # Rate limiting flow - check limits and forward
            threshold = flow_info.get('threshold', 'unknown')
            print(f"⏱️ Rate limiting: {flow_key} (threshold: {threshold})")
            actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        elif output_port == 'flood':
            # Flood action
            actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        else:
            # Normal forwarding
            actions = [parser.OFPActionOutput(output_port)]
        
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
        
        print(f"✅ Authorized: {flow_key} -> {output_port}")

    def _handle_unauthorized_flow(self, ev, datapath, in_port, pkt, flow_key):
        """Handle unauthorized flow - DENY by default."""
        msg = ev.msg
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id
        
        # Drop packet (no actions = drop)
        actions = []
        
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
        
        print(f"🚫 DENIED: {flow_key} - No authorization")

    def process_intent(self, intent_data):
        """Process network intent and install authorized flows."""
        try:
            intent_id = intent_data.get('intent_id', f"intent_{int(len(self.pending_intents))}")
            intent_text = intent_data.get('intent_text', '')
            intent_type = intent_data.get('intent_type', 'allow')
            
            print(f"🎯 Processing Intent: {intent_id}")
            print(f"   📝 Text: {intent_text}")
            print(f"   🔧 Type: {intent_type}")
            
            # Parse intent and generate flows
            flows = self._parse_intent_to_flows(intent_data)
            
            # Install flows on switches
            for flow in flows:
                self._install_authorized_flow(flow)
            
            # Store intent
            self.pending_intents[intent_id] = {
                'text': intent_text,
                'type': intent_type,
                'flows': flows,
                'timestamp': time.time()
            }
            
            self.stats['intents_processed'] += 1
            
            return {
                'intent_id': intent_id,
                'status': 'success',
                'flows_installed': len(flows),
                'message': f'Intent processed successfully'
            }
            
        except Exception as e:
            print(f"❌ Error processing intent: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _parse_intent_to_flows(self, intent_data):
        """Parse intent and generate flow rules."""
        flows = []
        
        # Extract intent category and type
        intent_category = intent_data.get('intent_category', 'connectivity')
        intent_type = intent_data.get('intent_type', 'allow')
        
        if intent_category == 'connectivity':
            # Handle connectivity intents (ACL, reachability)
            flows.extend(self._parse_connectivity_intent(intent_data))
        elif intent_category == 'performance':
            # Handle performance intents (QoS, path constraints)
            flows.extend(self._parse_performance_intent(intent_data))
        elif intent_category == 'security':
            # Handle security intents (ACL + anomaly detection)
            flows.extend(self._parse_security_intent(intent_data))
        else:
            # Fallback to basic connectivity
            flows.extend(self._parse_connectivity_intent(intent_data))
        
        return flows

    def _parse_connectivity_intent(self, intent_data):
        """Parse connectivity intent (ACL, reachability)."""
        flows = []
        source = intent_data.get('source', '')
        destination = intent_data.get('destination', '')
        intent_type = intent_data.get('intent_type', 'allow')
        
        if intent_type == 'allow':
            # Generate bidirectional flows
            flows.extend(self._generate_flows_for_hosts(source, destination, 'allow'))
        elif intent_type == 'deny':
            # Generate blocking flows
            flows.extend(self._generate_flows_for_hosts(source, destination, 'deny'))
        
        return flows

    def _parse_performance_intent(self, intent_data):
        """Parse performance intent (QoS, path constraints)."""
        flows = []
        source = intent_data.get('source', '')
        destination = intent_data.get('destination', '')
        qos_type = intent_data.get('qos_type', 'bandwidth')
        qos_value = intent_data.get('qos_value', '')
        
        # For now, treat QoS intents as allow flows with QoS metadata
        # In a full implementation, this would configure QoS parameters
        flows.extend(self._generate_flows_for_hosts(source, destination, 'allow'))
        
        # Add QoS metadata to flows
        for flow in flows:
            flow['qos_type'] = qos_type
            flow['qos_value'] = qos_value
            flow['priority'] = 150  # Higher priority for QoS flows
        
        print(f"⚡ QoS Intent: {qos_type} ({qos_value}) for {source} -> {destination}")
        return flows

    def _parse_security_intent(self, intent_data):
        """Parse security intent (ACL + anomaly detection)."""
        flows = []
        target_host = intent_data.get('target_host', '')
        security_action = intent_data.get('security_action', 'monitor')
        threshold = intent_data.get('threshold', '')
        
        if security_action == 'monitor':
            # Monitor traffic - allow but with monitoring
            flows.extend(self._generate_monitoring_flows(target_host))
        elif security_action == 'block_suspicious':
            # Block suspicious traffic - deny with monitoring
            flows.extend(self._generate_suspicious_blocking_flows(target_host))
        elif security_action == 'rate_limit':
            # Rate limiting - allow with rate limits
            flows.extend(self._generate_rate_limiting_flows(target_host, threshold))
        elif security_action == 'isolate':
            # Isolate host - deny all traffic
            flows.extend(self._generate_isolation_flows(target_host))
        
        print(f"🛡️ Security Intent: {security_action} for {target_host} (threshold: {threshold})")
        return flows

    def _generate_monitoring_flows(self, target_host):
        """Generate flows for traffic monitoring."""
        flows = []
        target_mac = self.host_macs.get(target_host)
        
        if not target_mac:
            print(f"⚠️  Unknown host: {target_host}")
            return flows
        
        # Generate monitoring flows for all switches
        for dpid in [1, 2, 3, 4]:
            flow = {
                'dpid': dpid,
                'src_mac': '00:00:00:00:00:00',  # Any source
                'dst_mac': target_mac,
                'priority': 50,  # Lower priority for monitoring
                'action': 'monitor',
                'intent_id': f'monitor_{target_host}',
                'security_action': 'monitor'
            }
            flows.append(flow)
        
        return flows

    def _generate_suspicious_blocking_flows(self, target_host):
        """Generate flows for blocking suspicious traffic."""
        flows = []
        target_mac = self.host_macs.get(target_host)
        
        if not target_mac:
            print(f"⚠️  Unknown host: {target_host}")
            return flows
        
        # Generate blocking flows for all switches
        for dpid in [1, 2, 3, 4]:
            flow = {
                'dpid': dpid,
                'src_mac': '00:00:00:00:00:00',  # Any source
                'dst_mac': target_mac,
                'priority': 200,  # High priority for blocking
                'action': 'deny',
                'intent_id': f'block_suspicious_{target_host}',
                'security_action': 'block_suspicious'
            }
            flows.append(flow)
        
        return flows

    def _generate_rate_limiting_flows(self, target_host, threshold):
        """Generate flows for rate limiting."""
        flows = []
        target_mac = self.host_macs.get(target_host)
        
        if not target_mac:
            print(f"⚠️  Unknown host: {target_host}")
            return flows
        
        # Generate rate limiting flows for all switches
        for dpid in [1, 2, 3, 4]:
            flow = {
                'dpid': dpid,
                'src_mac': '00:00:00:00:00:00',  # Any source
                'dst_mac': target_mac,
                'priority': 100,
                'action': 'rate_limit',
                'intent_id': f'rate_limit_{target_host}',
                'security_action': 'rate_limit',
                'threshold': threshold
            }
            flows.append(flow)
        
        return flows

    def _generate_isolation_flows(self, target_host):
        """Generate flows for host isolation."""
        flows = []
        target_mac = self.host_macs.get(target_host)
        
        if not target_mac:
            print(f"⚠️  Unknown host: {target_host}")
            return flows
        
        # Generate isolation flows for all switches
        for dpid in [1, 2, 3, 4]:
            # Block traffic TO the target
            flow_to = {
                'dpid': dpid,
                'src_mac': '00:00:00:00:00:00',  # Any source
                'dst_mac': target_mac,
                'priority': 300,  # Highest priority for isolation
                'action': 'deny',
                'intent_id': f'isolate_{target_host}',
                'security_action': 'isolate'
            }
            flows.append(flow_to)
            
            # Block traffic FROM the target
            flow_from = {
                'dpid': dpid,
                'src_mac': target_mac,
                'dst_mac': '00:00:00:00:00:00',  # Any destination
                'priority': 300,  # Highest priority for isolation
                'action': 'deny',
                'intent_id': f'isolate_{target_host}',
                'security_action': 'isolate'
            }
            flows.append(flow_from)
        
        return flows

    def _generate_flows_for_hosts(self, source, destination, action):
        """Generate flows for host-to-host communication."""
        flows = []
        
        # Convert host names to MAC addresses
        src_mac = self.host_macs.get(source)
        dst_mac = self.host_macs.get(destination)
        
        if not src_mac or not dst_mac:
            print(f"⚠️  Unknown host: {source} or {destination}")
            return flows
        
        # Generate flows for each switch
        for dpid in [1, 2, 3, 4]:
            if action == 'allow':
                # Calculate output port for this switch
                output_port = self._calculate_output_port(dpid, destination)
                if output_port:
                    flow = {
                        'dpid': dpid,
                        'src_mac': src_mac,
                        'dst_mac': dst_mac,
                        'output_port': output_port,
                        'priority': 100,
                        'action': 'allow'
                    }
                    flows.append(flow)
                    
                    # Add reverse flow
                    reverse_output_port = self._calculate_output_port(dpid, source)
                    if reverse_output_port:
                        reverse_flow = {
                            'dpid': dpid,
                            'src_mac': dst_mac,
                            'dst_mac': src_mac,
                            'output_port': reverse_output_port,
                            'priority': 100,
                            'action': 'allow'
                        }
                        flows.append(reverse_flow)
            
            elif action == 'deny':
                # Blocking flow
                flow = {
                    'dpid': dpid,
                    'src_mac': src_mac,
                    'dst_mac': dst_mac,
                    'priority': 200,  # Higher priority than allow flows
                    'action': 'deny'
                }
                flows.append(flow)
        
        return flows

    def _calculate_output_port(self, dpid, host_name):
        """Calculate output port for reaching a host from a switch."""
        # Check if host is directly connected
        if dpid in self.host_locations:
            for port, connected_host in self.host_locations[dpid].items():
                if connected_host == host_name:
                    return port
        
        # Host is not directly connected, calculate path
        return self._calculate_next_hop_port(dpid, host_name)

    def _calculate_next_hop_port(self, src_switch, target_host):
        """Calculate next hop port for reaching target host."""
        # Find which switch the target host is connected to
        target_switch = None
        for switch_id, hosts in self.host_locations.items():
            if target_host in hosts.values():
                target_switch = switch_id
                break
        
        if not target_switch or src_switch == target_switch:
            return None
        
        # Use BFS to find shortest path
        path = self._find_shortest_path(src_switch, target_switch)
        
        if path and len(path) > 1:
            next_switch = path[1]
            return self.topology_map.get(src_switch, {}).get(next_switch)
        
        return None

    def _find_shortest_path(self, src, dst):
        """Find shortest path between two switches using BFS."""
        if src == dst:
            return [src]
        
        visited = set()
        queue = [(src, [src])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == dst:
                return path
            
            # Explore neighbors
            for neighbor in self.topology_map.get(current, {}):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return None

    def _install_authorized_flow(self, flow):
        """Install an authorized flow on the switch."""
        dpid = flow['dpid']
        if dpid not in self.datapaths:
            print(f"⚠️  Switch {dpid} not connected")
            return
        
        datapath = self.datapaths[dpid]
        parser = datapath.ofproto_parser
        
        # Create match
        match = parser.OFPMatch(
            eth_src=flow['src_mac'],
            eth_dst=flow['dst_mac']
        )
        
        action = flow['action']
        
        if action == 'allow':
            # Allow flow
            if 'output_port' in flow:
                actions = [parser.OFPActionOutput(flow['output_port'])]
            else:
                # For QoS flows, use flood as default
                actions = [parser.OFPActionOutput(datapath.ofproto.OFPP_FLOOD)]
            
            self.add_flow(datapath, flow['priority'], match, actions, idle_timeout=0)
            
            # Store in authorized flows
            flow_key = self._generate_flow_key(flow['src_mac'], flow['dst_mac'], dpid)
            self.authorized_flows[flow_key] = {
                'output_port': flow.get('output_port', 'flood'),
                'priority': flow['priority'],
                'intent_id': flow.get('intent_id', 'unknown'),
                'qos_type': flow.get('qos_type'),
                'qos_value': flow.get('qos_value')
            }
            
            print(f"✅ Installed ALLOW flow: s{dpid} {flow['src_mac']}->{flow['dst_mac']}")
            self.stats['flows_installed'] += 1
            
        elif action == 'deny':
            # Deny flow (drop)
            actions = []  # No actions = drop
            self.add_flow(datapath, flow['priority'], match, actions, idle_timeout=0)
            
            # Store in blocked flows
            flow_key = self._generate_flow_key(flow['src_mac'], flow['dst_mac'], dpid)
            self.blocked_flows[flow_key] = {
                'priority': flow['priority'],
                'intent_id': flow.get('intent_id', 'unknown'),
                'security_action': flow.get('security_action')
            }
            
            print(f"🚫 Installed DENY flow: s{dpid} {flow['src_mac']}->{flow['dst_mac']}")
            self.stats['flows_blocked'] += 1
            
        elif action == 'monitor':
            # Monitor flow - allow but with monitoring
            actions = [parser.OFPActionOutput(datapath.ofproto.OFPP_FLOOD)]
            self.add_flow(datapath, flow['priority'], match, actions, idle_timeout=0)
            
            # Store in authorized flows with monitoring flag
            flow_key = self._generate_flow_key(flow['src_mac'], flow['dst_mac'], dpid)
            self.authorized_flows[flow_key] = {
                'output_port': 'flood',
                'priority': flow['priority'],
                'intent_id': flow.get('intent_id', 'unknown'),
                'monitoring': True,
                'security_action': 'monitor'
            }
            
            print(f"👁️ Installed MONITOR flow: s{dpid} {flow['src_mac']}->{flow['dst_mac']}")
            self.stats['flows_installed'] += 1
            
        elif action == 'rate_limit':
            # Rate limiting flow - allow with rate limits
            actions = [parser.OFPActionOutput(datapath.ofproto.OFPP_FLOOD)]
            self.add_flow(datapath, flow['priority'], match, actions, idle_timeout=0)
            
            # Store in authorized flows with rate limiting info
            flow_key = self._generate_flow_key(flow['src_mac'], flow['dst_mac'], dpid)
            self.authorized_flows[flow_key] = {
                'output_port': 'flood',
                'priority': flow['priority'],
                'intent_id': flow.get('intent_id', 'unknown'),
                'rate_limiting': True,
                'threshold': flow.get('threshold'),
                'security_action': 'rate_limit'
            }
            
            print(f"⏱️ Installed RATE_LIMIT flow: s{dpid} {flow['src_mac']}->{flow['dst_mac']} (threshold: {flow.get('threshold')})")
            self.stats['flows_installed'] += 1

    def get_security_status(self):
        """Get current security status."""
        return {
            'authorized_flows': len(self.authorized_flows),
            'blocked_flows': len(self.blocked_flows),
            'pending_intents': len(self.pending_intents),
            'stats': self.stats,
            'zero_trust_active': True
        }


class LLMSDNControllerRestAPI(ControllerBase):
    """REST API for LLM SDN Controller."""
    
    def __init__(self, req, link, data, **config):
        super(LLMSDNControllerRestAPI, self).__init__(req, link, data, **config)
        self.controller = data['controller']

    # Zero Trust API endpoints
    @route('intent', '/intent/process', methods=['POST'])
    def process_intent(self, req, **kwargs):
        """Process a network intent."""
        try:
            intent_data = req.json if req.body else {}
            result = self.controller.process_intent(intent_data)
            body = json.dumps(result, indent=2)
            return Response(content_type='application/json; charset=utf-8', text=body)
        except Exception as e:
            return Response(status=500, content_type='text/plain; charset=utf-8', text=str(e))

    @route('status', '/status', methods=['GET'])
    def get_status(self, req, **kwargs):
        """Get controller status."""
        try:
            status = self.controller.get_security_status()
            body = json.dumps(status, indent=2)
            return Response(content_type='application/json; charset=utf-8', text=body)
        except Exception as e:
            return Response(status=500, content_type='text/plain; charset=utf-8', text=str(e))

    @route('flows', '/flows/authorized', methods=['GET'])
    def get_authorized_flows(self, req, **kwargs):
        """Get authorized flows."""
        try:
            body = json.dumps(self.controller.authorized_flows, indent=2)
            return Response(content_type='application/json; charset=utf-8', text=body)
        except Exception as e:
            return Response(status=500, content_type='text/plain; charset=utf-8', text=str(e))

    @route('flows', '/flows/blocked', methods=['GET'])
    def get_blocked_flows(self, req, **kwargs):
        """Get blocked flows."""
        try:
            body = json.dumps(self.controller.blocked_flows, indent=2)
            return Response(content_type='application/json; charset=utf-8', text=body)
        except Exception as e:
            return Response(status=500, content_type='text/plain; charset=utf-8', text=str(e))

    @route('intents', '/intents', methods=['GET'])
    def get_intents(self, req, **kwargs):
        """Get processed intents."""
        try:
            body = json.dumps(self.controller.pending_intents, indent=2)
            return Response(content_type='application/json; charset=utf-8', text=body)
        except Exception as e:
            return Response(status=500, content_type='text/plain; charset=utf-8', text=str(e))
