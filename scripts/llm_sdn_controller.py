#!/usr/bin/env python3
"""
Custom RYU controller application for LLM Intent-based SDN.
Handles basic switching with reduced logging and better flow management.
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.lib.packet import lldp
from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link
from ryu.app.wsgi import WSGIApplication, ControllerBase, route
from webob import Response
import json
import logging
import time
import threading


class LLMSDNController(app_manager.RyuApp):
    """
    Custom SDN controller for LLM Intent-based SDN system.
    
    Features:
    - Basic learning switch functionality
    - Reduced logging for better performance
    - Flow table management
    - Topology awareness
    - REST API for topology information
    """
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    _CONTEXTS = {
        'wsgi': WSGIApplication,
    }
    
    def __init__(self, *args, **kwargs):
        super(LLMSDNController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.switches = {}
        self.links = {}
        self.switch_ports = {}  # Store discovered ports for each switch
        self.lldp_probe_interval = 10  # Send LLDP probes every 10 seconds
        
        # Set logging level to reduce noise
        self.logger.setLevel(logging.WARNING)
        
        print("LLM SDN Controller initialized with LLDP topology discovery")
        
        # Setup REST API
        self.wsgi = kwargs['wsgi']
        self.wsgi.register(LLMSDNControllerRestAPI, self)
        
        # Start LLDP probe thread
        self.lldp_thread = threading.Thread(target=self._lldp_probe_loop)
        self.lldp_thread.daemon = True
        self.lldp_thread.start()
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch connection and install table-miss flow entry."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Store switch info
        self.switches[datapath.id] = datapath
        print(f"Switch connected: DPID {datapath.id}")
        
        # Initialize ports for this switch
        self.switch_ports[datapath.id] = []
        
        # Request port description to discover ports
        self._request_port_description(datapath)
        
        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                        ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
    
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
                                  instructions=inst,
                                  idle_timeout=idle_timeout,
                                  hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                  match=match, instructions=inst,
                                  idle_timeout=idle_timeout,
                                  hard_timeout=hard_timeout)
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle packet-in events with learning switch logic."""
        # Reduce packet-in logging by only logging significant events
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        # Handle LLDP packets for topology discovery
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            self._handle_lldp(datapath, in_port, pkt)
            return
            
        # Ignore multicast packets to reduce noise
        if eth.dst.startswith('33:33:'):  # IPv6 multicast
            return
        if eth.dst.startswith('01:00:5e:'):  # IPv4 multicast
            return
        
        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        # Initialize MAC table for this switch
        self.mac_to_port.setdefault(dpid, {})
        
        # Learn source MAC address
        if src not in self.mac_to_port[dpid]:
            print(f"Learned: SW{dpid} Port{in_port} -> {src}")
        self.mac_to_port[dpid][src] = in_port
        
        # Determine output port
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        actions = [parser.OFPActionOutput(out_port)]
        
        # Install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            
            # Verify if we have a valid buffer_id
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, 
                            buffer_id=msg.buffer_id, idle_timeout=60)
                return
            else:
                self.add_flow(datapath, 1, match, actions, idle_timeout=60)
        
        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
    
    @set_ev_cls(event.EventSwitchEnter)
    def get_topology_data(self, ev):
        """Handle topology discovery events."""
        switch_list = get_switch(self, None)
        switches = [switch.dp.id for switch in switch_list]
        
        print(f"Switch entered: {len(switches)} switches total")
        
        # Update switches state (keep datapath objects)
        for switch in switch_list:
            if switch.dp.id not in self.switches:
                self.switches[switch.dp.id] = switch.dp
        
        # Links are managed by our LLDP discovery
        print(f"Current topology: {len(self.switches)} switches, {len(self.links)} links")
    
    @set_ev_cls(event.EventSwitchLeave)
    def switch_leave_handler(self, ev):
        """Handle switch disconnection."""
        switch = ev.switch
        print(f"Switch disconnected: DPID {switch.dp.id}")
        
        # Clean up
        if switch.dp.id in self.switches:
            del self.switches[switch.dp.id]
        if switch.dp.id in self.mac_to_port:
            del self.mac_to_port[switch.dp.id]
        if switch.dp.id in self.switch_ports:
            del self.switch_ports[switch.dp.id]
    
    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def port_status_handler(self, ev):
        """Handle port status change events"""
        try:
            msg = ev.msg
            datapath = msg.datapath
            dpid = datapath.id
            reason = msg.reason
            port_no = msg.desc.port_no
            
            if reason == datapath.ofproto.OFPPR_ADD:
                print(f"Port {port_no} added to switch {dpid}")
                # Request updated port description
                self._request_port_description(datapath)
            elif reason == datapath.ofproto.OFPPR_DELETE:
                print(f"Port {port_no} deleted from switch {dpid}")
                # Remove port from storage
                if dpid in self.switch_ports:
                    self.switch_ports[dpid] = [p for p in self.switch_ports[dpid] if p['port_no'] != port_no]
            elif reason == datapath.ofproto.OFPPR_MODIFY:
                print(f"Port {port_no} modified on switch {dpid}")
                # Request updated port description
                self._request_port_description(datapath)
                
        except Exception as e:
            print(f"Error handling port status event: {e}")
    
    def _lldp_probe_loop(self):
        """Send LLDP probe packets periodically"""
        while True:
            try:
                time.sleep(self.lldp_probe_interval)
                self._send_lldp_probes()
            except Exception as e:
                print(f"Error in LLDP probe loop: {e}")
    
    def _send_lldp_probes(self):
        """Send LLDP probe packets to all switches"""
        for dpid, datapath in self.switches.items():
            try:
                # Get ports for this switch
                ports = self._get_switch_ports(datapath)
                for port in ports:
                    port_no = port['port_no']
                    hw_addr = port['hw_addr']
                    if port_no != datapath.ofproto.OFPP_LOCAL:
                        print(f"Sending LLDP probe to switch {dpid}, port {port_no}")
                        self._send_lldp_packet(datapath, port_no, hw_addr)
            except Exception as e:
                print(f"Error sending LLDP probes to switch {dpid}: {e}")
    
    def _request_port_description(self, datapath):
        """Request port description from switch"""
        try:
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser
            
            # Send port description request
            req = parser.OFPPortDescStatsRequest(datapath, 0)
            datapath.send_msg(req)
            print(f"Requested port description from switch {datapath.id}")
            
        except Exception as e:
            print(f"Error requesting port description from switch {datapath.id}: {e}")
    
    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def port_desc_stats_reply_handler(self, ev):
        """Handle port description reply"""
        try:
            msg = ev.msg
            datapath = msg.datapath
            dpid = datapath.id
            
            # Parse port descriptions
            ports = []
            for stat in msg.body:
                if stat.port_no != datapath.ofproto.OFPP_LOCAL:
                    port_info = {
                        'port_no': stat.port_no,
                        'hw_addr': stat.hw_addr,
                        'name': stat.name.decode('utf-8', errors='ignore'),
                        'config': stat.config,
                        'state': stat.state,
                        'curr': stat.curr,
                        'advertised': stat.advertised,
                        'supported': stat.supported,
                        'peer': stat.peer
                    }
                    ports.append(port_info)
            
            # Store discovered ports
            self.switch_ports[dpid] = ports
            print(f"Discovered {len(ports)} ports for switch {dpid}: {[p['port_no'] for p in ports]}")
            
        except Exception as e:
            print(f"Error handling port description reply: {e}")
    
    def _get_switch_ports(self, datapath):
        """Get ports for a switch"""
        try:
            # Return discovered ports from storage
            dpid = datapath.id
            if dpid in self.switch_ports:
                ports = self.switch_ports[dpid]
                print(f"Found {len(ports)} ports for switch {dpid}")
                return ports
            else:
                print(f"No ports discovered for switch {dpid} yet")
                return []
            
        except Exception as e:
            print(f"Error getting ports for switch {datapath.id}: {e}")
            return []
    
    def _send_lldp_packet(self, datapath, port_no, hw_addr):
        """Send LLDP packet to a specific port"""
        try:
            # Create LLDP packet with switch and port information
            pkt = packet.Packet()
            
            # Add Ethernet header
            eth = ethernet.ethernet(
                dst='01:80:c2:00:00:0e',  # LLDP multicast address
                src=hw_addr,
                ethertype=ether_types.ETH_TYPE_LLDP
            )
            pkt.add_protocol(eth)
            
            # Add LLDP header with required TLVs
            from ryu.lib.packet.lldp import ChassisID, PortID, TTL, End
            
            # Create TLVs
            chassis_id = ChassisID(subtype=1, chassis_id=str(datapath.id).encode())
            port_id = PortID(subtype=2, port_id=str(port_no).encode())
            ttl = TTL(ttl=120)
            end = End()
            
            # Create LLDP packet with TLVs
            lldp_pkt = lldp.lldp(tlvs=[chassis_id, port_id, ttl, end])
            pkt.add_protocol(lldp_pkt)
            
            # Serialize packet
            pkt.serialize()
            
            # Send packet out
            actions = [datapath.ofproto_parser.OFPActionOutput(port_no)]
            out = datapath.ofproto_parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=datapath.ofproto.OFP_NO_BUFFER,
                in_port=datapath.ofproto.OFPP_CONTROLLER,
                actions=actions,
                data=pkt.data
            )
            datapath.send_msg(out)
            
            print(f"Sent LLDP packet from switch {datapath.id}, port {port_no}")
            
        except Exception as e:
            print(f"Error sending LLDP packet: {e}")
    
    def _handle_lldp(self, datapath, in_port, pkt):
        """Handle received LLDP packet"""
        try:
            # Parse LLDP packet
            lldp_pkt = pkt.get_protocol(lldp.lldp)
            if lldp_pkt:
                # Extract switch and port information from LLDP packet
                src_dpid = datapath.id
                src_port = in_port
                
                # Parse LLDP TLVs to get source switch and port info
                src_switch_id = None
                src_port_id = None
                
                for tlv in lldp_pkt.tlvs:
                    if hasattr(tlv, 'tlv_type'):
                        if tlv.tlv_type == 1:  # Chassis ID
                            try:
                                src_switch_id = int(tlv.chassis_id.decode())
                            except:
                                pass
                        elif tlv.tlv_type == 2:  # Port ID
                            try:
                                src_port_id = int(tlv.port_id.decode())
                            except:
                                pass
                
                if src_switch_id and src_port_id:
                    print(f"LLDP: Switch {src_switch_id}:{src_port_id} -> Switch {src_dpid}:{src_port}")
                    
                    # Store link information
                    link_key = (src_switch_id, src_dpid)
                    reverse_key = (src_dpid, src_switch_id)
                    
                    # Only store if not already stored
                    if link_key not in self.links and reverse_key not in self.links:
                        link_info = {
                            'src_switch': src_switch_id,
                            'src_port': src_port_id,
                            'dst_switch': src_dpid,
                            'dst_port': src_port
                        }
                        self.links[link_key] = link_info
                        print(f"Link discovered: {src_switch_id}:{src_port_id} <-> {src_dpid}:{src_port}")
                
                # Update topology information
                self._update_topology()
                
        except Exception as e:
            print(f"Error handling LLDP packet: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_topology(self):
        """Update topology information"""
        try:
            # Get current switches
            switch_list = get_switch(self, None)
            
            # Update switches state (keep datapath objects)
            for switch in switch_list:
                if switch.dp.id not in self.switches:
                    self.switches[switch.dp.id] = switch.dp
            
            # Links are managed by our LLDP discovery
            print(f"Topology updated: {len(self.switches)} switches, {len(self.links)} links")
            
            # Print discovered links
            if self.links:
                print("Discovered links:")
                for (src, dst), link_info in self.links.items():
                    print(f"  {src}:{link_info['src_port']} <-> {dst}:{link_info['dst_port']}")
            
        except Exception as e:
            print(f"Error updating topology: {e}")
    
    @set_ev_cls(event.EventLinkAdd)
    def link_add_handler(self, ev):
        """Handle link add event"""
        link = ev.link
        src_dpid = link.src.dpid
        dst_dpid = link.dst.dpid
        src_port_no = link.src.port_no
        dst_port_no = link.dst.port_no
        
        print(f"Link added: {src_dpid}:{src_port_no} <-> {dst_dpid}:{dst_port_no}")
        
        # Update topology
        self._update_topology()
    
    @set_ev_cls(event.EventLinkDelete)
    def link_delete_handler(self, ev):
        """Handle link delete event"""
        link = ev.link
        src_dpid = link.src.dpid
        dst_dpid = link.dst.dpid
        
        print(f"Link deleted: {src_dpid} <-> {dst_dpid}")
        
        # Update topology
        self._update_topology()


class LLMSDNControllerRestAPI(ControllerBase):
    """REST API controller for LLM SDN Controller."""
    
    def __init__(self, req, link, data, **config):
        super(LLMSDNControllerRestAPI, self).__init__(req, link, data, **config)
        # In Ryu WSGI, the data is passed directly as the controller instance
        self.controller = data
    
    @route('lldp', '/custom/links', methods=['GET'])
    def get_links(self, req, **kwargs):
        """Get discovered links from LLDP discovery."""
        try:
            links = []
            for (src, dst), link_info in self.controller.links.items():
                links.append({
                    'src_switch': link_info['src_switch'],
                    'src_port': link_info['src_port'],
                    'dst_switch': link_info['dst_switch'],
                    'dst_port': link_info['dst_port']
                })
            
            body = json.dumps(links)
            return Response(content_type='application/json; charset=utf-8', body=body)
            
        except Exception as e:
            import traceback
            error_msg = f"Error in get_links: {str(e)}\n{traceback.format_exc()}"
            return Response(status=500, body=error_msg)
    
    @route('lldp', '/custom/topology', methods=['GET'])
    def get_topology(self, req, **kwargs):
        """Get complete topology information."""
        try:
            topology = {
                'switches': list(self.controller.switches.keys()),
                'links': [],
                'ports': {}
            }
            
            # Add links
            for (src, dst), link_info in self.controller.links.items():
                topology['links'].append({
                    'src_switch': link_info['src_switch'],
                    'src_port': link_info['src_port'],
                    'dst_switch': link_info['dst_switch'],
                    'dst_port': link_info['dst_port']
                })
            
            # Add ports
            for dpid, ports in self.controller.switch_ports.items():
                topology['ports'][str(dpid)] = [
                    {
                        'port_no': port['port_no'],
                        'name': port['name'],
                        'hw_addr': port['hw_addr']
                    }
                    for port in ports
                ]
            
            body = json.dumps(topology)
            return Response(content_type='application/json; charset=utf-8', body=body)
            
        except Exception as e:
            import traceback
            error_msg = f"Error in get_topology: {str(e)}\n{traceback.format_exc()}"
            return Response(status=500, body=error_msg)