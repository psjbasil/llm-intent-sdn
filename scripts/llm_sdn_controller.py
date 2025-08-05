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
from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link
import logging


class LLMSDNController(app_manager.RyuApp):
    """
    Custom SDN controller for LLM Intent-based SDN system.
    
    Features:
    - Basic learning switch functionality
    - Reduced logging for better performance
    - Flow table management
    - Topology awareness
    """
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        super(LLMSDNController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.switches = {}
        self.links = {}
        
        # Set logging level to reduce noise
        self.logger.setLevel(logging.WARNING)
        
        print("LLM SDN Controller initialized")
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch connection and install table-miss flow entry."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Store switch info
        self.switches[datapath.id] = datapath
        print(f"Switch connected: DPID {datapath.id}")
        
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
        
        # Ignore LLDP and IPv6 multicast packets to reduce noise
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
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
        
        links_list = get_link(self, None)
        links = [(link.src.dpid, link.dst.dpid, 
                 {'port': link.src.port_no}) for link in links_list]
        
        print(f"Topology updated: {len(switches)} switches, {len(links)} links")
        
        # Store topology info
        self.switches = {sw: None for sw in switches}
        self.links = {(src, dst): port_info for src, dst, port_info in links}
    
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