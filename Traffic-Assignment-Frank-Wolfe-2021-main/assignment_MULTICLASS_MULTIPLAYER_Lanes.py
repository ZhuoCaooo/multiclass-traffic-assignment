import math
import time
import heapq
from typing import Tuple, Any
from mpl_toolkits import mplot3d
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings

from network_import import *
from utils import PathUtils
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

''' define the proportion of CAVs in total demand'''



class FlowTransportNetwork:

    def __init__(self):
        self.linkSet = {}
        self.nodeSet = {}

        self.tripSet = {}
        self.zoneSet = {}
        self.originZones = {}

        self.networkx_graph = None

    def to_networkx(self):
        if self.networkx_graph is None:
            self.networkx_graph = nx.DiGraph([(int(begin), int(end)) for (begin, end) in self.linkSet.keys()])
        return self.networkx_graph

    def reset_flow(self):
        for link in self.linkSet.values():
            link.reset_flow()

    def reset(self):
        for link in self.linkSet.values():
            link.reset()


class Zone:
    def __init__(self, zoneId: str):
        self.zoneId = zoneId

        self.lat = 0
        self.lon = 0
        self.destList = []  # list of zone ids (strs)


class Node:
    """
    This class has attributes associated with any node
    """

    def __init__(self, nodeId: str):
        self.Id = nodeId

        self.lat = 0
        self.lon = 0

        self.outLinks = []  # list of node ids (strs)
        self.inLinks = []  # list of node ids (strs)

        # For Dijkstra
        self.label = np.inf
        # label for CAV
        self.label8 = np.inf
        # label for HDV
        self.label7 = np.inf

        self.predCAV = None
        self.predHDV = None


class Link:
    """
    This class has attributes associated with any link
    """

    def __init__(self,
                 init_node: str,
                 term_node: str,
                 capacity: float,
                 length: float,
                 fft: float,
                 b: float,
                 power: float,
                 speed_limit: float,
                 toll: float,
                 linkType
                 ):
        self.init_node = init_node
        self.term_node = term_node
        self.max_capacity = float(capacity)  # veh per hour
        self.length = float(length)  # Length
        self.fft = float(fft)  # Free flow travel time (min)
        self.beta = float(power)
        self.alpha = float(b)
        self.speedLimit = float(speed_limit)
        self.toll = float(toll)
        self.linkType = linkType

        self.curr_capacity_percentage = 1
        self.capacity = self.max_capacity
        '''used in single class traffic assignment'''
        self.flow = 0.0
        ''' define cav_flow_initial with corresponding hdv flow, multiclass'''
        self.flowCAV = 0.0
        self.flowHDV = 0.0
        ''' pre- flows in iterations to calculate the gap'''
        self.preflowCAV = 0.0
        self.preflowHDV = 0.0
        '''used in single class traffic assignment'''
        self.cost = self.fft
        ''' define multiclass initial travel cost'''
        self.costCAV = self.fft
        self.costHDV = self.fft
        ''' define multiclass initial travel cost'''
        self.precostCAV = self.fft
        self.precostHDV = self.fft
        '''vehicle headway settings'''


    # Method not used for assignment
    def modify_capacity(self, delta_percentage: float):
        assert -1 <= delta_percentage <= 1
        self.curr_capacity_percentage += delta_percentage
        self.curr_capacity_percentage = max(0, min(1, self.curr_capacity_percentage))
        self.capacity = self.max_capacity * self.curr_capacity_percentage

    def reset(self):
        self.curr_capacity_percentage = 1
        self.capacity = self.max_capacity
        self.reset_flow()

    def reset_flow(self):
        self.flow = 0.0
        self.cost = self.fft


class Demand:
    def __init__(self,
                 init_node: str,
                 term_node: str,
                 demand: float,
                 demandCAV: float,
                 demandHDV: float
                 ):
        self.fromZone = init_node
        self.toNode = term_node
        self.demand = float(demand)
        self.demandCAV = float(demandCAV)
        self.demandHDV = float(demandHDV)
        # print('demandHDV=', demandHDV)
        # print('demandCAV=', demandCAV)


def DijkstraHeapCAV(origin, network: FlowTransportNetwork):
    """
    Calculates shortest path from an origin to all other destinations.
    The labels and preds are stored in node instances.
    """
    for n in network.nodeSet:
        network.nodeSet[n].label8 = np.inf
        network.nodeSet[n].predCAV = None
    network.nodeSet[origin].label8 = 0.0
    network.nodeSet[origin].predCAV = None
    SE = [(0, origin)]
    while SE:
        currentNode = heapq.heappop(SE)[1]
        currentLabel = network.nodeSet[currentNode].label8
        for toNode in network.nodeSet[currentNode].outLinks:
            link = (currentNode, toNode)
            newNode = toNode
            newPredCAV = currentNode
            existingLabel = network.nodeSet[newNode].label8
            newLabel8 = currentLabel + network.linkSet[link].costCAV
            if newLabel8 < existingLabel:
                heapq.heappush(SE, (newLabel8, newNode))
                network.nodeSet[newNode].label8 = newLabel8
                network.nodeSet[newNode].predCAV = newPredCAV


def DijkstraHeapHDV(origin, network: FlowTransportNetwork):
    """
    Calculates shortest path from an origin to all other destinations.
    The labels and preds are stored in node instances.
    """
    for n in network.nodeSet:
        network.nodeSet[n].label7 = np.inf
        network.nodeSet[n].predHDV = None
    network.nodeSet[origin].label7 = 0.0
    network.nodeSet[origin].predHDV = None
    SE = [(0, origin)]
    while SE:
        currentNode = heapq.heappop(SE)[1]
        currentLabel = network.nodeSet[currentNode].label7
        for toNode in network.nodeSet[currentNode].outLinks:
            link = (currentNode, toNode)
            newNode = toNode
            newPredHDV = currentNode
            existingLabel = network.nodeSet[newNode].label7
            newLabel7 = currentLabel + network.linkSet[link].costHDV
            if newLabel7 < existingLabel:
                heapq.heappush(SE, (newLabel7, newNode))
                network.nodeSet[newNode].label7 = newLabel7
                network.nodeSet[newNode].predHDV = newPredHDV


def BPRcostFunctionCAV(optimal: bool,
                       fft: float,
                       alpha: float,
                       flowCAV: float,
                       capacity: float,
                       beta: float,
                       length: float,
                       flowHDV: float,
                       maxSpeed: float,
                       toll: float
                       ) -> float:
    if flowHDV+flowCAV == 0:
        linkportionCAV=0
        linkportionHDV=1
    else:
        linkportionCAV = flowCAV / (flowCAV + flowHDV)
        linkportionHDV = flowHDV / (flowCAV + flowHDV)

    capacityCAV = capacity * (2 / 3)
    capacityHDV = capacity * (1 / 3)
    capacitynew = capacity

    h00=1.8
    h11=0.5
    h10=1.5
    h01=1.0
    h10_hat=(h10+h01)/2
    headway_a = h11+h00-2*h10_hat
    headway_b = h10_hat-h00
    capacity_hat=1/(headway_a*linkportionCAV*(1-linkportionCAV)*intensity+headway_a*linkportionCAV*linkportionCAV+2*headway_b*linkportionCAV+h00)
    capacity_hdv=1/(headway_a*linkportionCAV*linkportionCAV+2*headway_b*linkportionCAV+h00)
    capacity_toll=capacity_hat/capacity_hdv*capacitynew*tollportion
    capacity_untoll = capacity_hat / capacity_hdv * capacitynew*(1-tollportion)

    if capacity < 1e-3:
        return np.finfo(np.float32).max
    if optimal:
        if toll == 1:
            return fft * (1 + (alpha * math.pow(((flowCAV * 1.0 + flowHDV * 1.0) / capacity_toll), beta)) * (beta + 1))
        elif toll == 2:
            return fft * (1 + (alpha * math.pow(((flowCAV * 1.0 + flowHDV * 1.0) / capacity_untoll), beta)) * (beta + 1))
        else:
            return fft * (1 + (alpha * math.pow(((flowCAV * 1.0 + flowHDV * 1.0) / capacitynew), beta)) * (beta + 1))

    if not optimal:
        if toll == 1:
            return fft * (1 + math.pow(((flowCAV * 1 + 1 * flowHDV) / capacity_toll), beta)) * VOTCAV
        if toll == 2:
            return fft * (1 + math.pow(((flowCAV * 1 + 1 * flowHDV) / capacity_untoll), beta))* VOTCAV
        else:
            return fft * (1 + alpha * math.pow(((flowHDV * 1 + 1 * flowCAV) / capacitynew), beta))* VOTCAV




def BPRcostFunctionHDV(optimal: bool,
                       fft: float,
                       alpha: float,
                       flowCAV: float,
                       capacity: float,
                       beta: float,
                       length: float,
                       flowHDV: float,
                       maxSpeed: float,
                       toll: float
                       ) -> float:
    if flowHDV+flowCAV == 0:
        linkportionCAV=0
        linkportionHDV=1
    else:
        linkportionCAV = flowCAV / (flowCAV + flowHDV)
        linkportionHDV = flowHDV / (flowCAV + flowHDV)

    capacityCAV = capacity * (2 / 3)
    capacityHDV = capacity * (1 / 3)
    capacitynew = capacity

    h00=1.8
    h11=0.5
    h10=1.5
    h01=1.0
    h10_hat=(h10+h01)/2
    headway_a = h11+h00-2*h10_hat
    headway_b = h10_hat-h00
    capacity_hat=1/(headway_a*linkportionCAV*(1-linkportionCAV)*intensity+headway_a*linkportionCAV*linkportionCAV+2*headway_b*linkportionCAV+h00)
    capacity_hdv=1/(headway_a*linkportionCAV*linkportionCAV+2*headway_b*linkportionCAV+h00)
    capacity_toll=capacity_hat/capacity_hdv*capacitynew*tollportion
    capacity_untoll = capacity_hat / capacity_hdv * capacitynew*(1-tollportion)

    if capacity < 1e-3:
        return np.finfo(np.float32).max

    if optimal:
        if toll == 1:
            return fft * (1 + math.pow(((flowCAV * 1 + 1 * flowHDV) / capacity_toll), beta)) + tollfactor * length
        if toll == 2:
            return fft * (1 + math.pow(((flowCAV * 1 + 1 * flowHDV) / capacity_untoll), beta))
        else:
            return fft * (1 + alpha * math.pow(((flowHDV * 1 + 1 * flowCAV) / capacitynew), beta))

    if not optimal:
        if toll == 1:
            return fft * (1 + math.pow(((flowCAV * 1 + 1 * flowHDV) / capacity_toll), beta))*VOTHDV + tollfactor * length * toll_index
        if toll == 2:
            return fft * (1 + math.pow(((flowCAV * 1 + 1 * flowHDV) / capacity_untoll), beta))*VOTHDV
        else:
            return fft * (1 + alpha * math.pow(((flowHDV * 1 + 1 * flowCAV) / capacitynew), beta))*VOTHDV



def updateTravelTime(network: FlowTransportNetwork, optimal: bool = False, costCAV=BPRcostFunctionCAV,
                     costHDV=BPRcostFunctionHDV):
    """
    This method updates the travel time on the links with the current flow
    """
    for l in network.linkSet:
        network.linkSet[l].costCAV = costCAV(optimal,
                                             network.linkSet[l].fft,
                                             network.linkSet[l].alpha,
                                             network.linkSet[l].flowCAV,
                                             network.linkSet[l].capacity,
                                             network.linkSet[l].beta,
                                             network.linkSet[l].length,
                                             network.linkSet[l].flowHDV,
                                             network.linkSet[l].speedLimit,
                                             network.linkSet[l].toll
                                             )
        network.linkSet[l].costHDV = costHDV(optimal,
                                             network.linkSet[l].fft,
                                             network.linkSet[l].alpha,
                                             network.linkSet[l].flowCAV,
                                             network.linkSet[l].capacity,
                                             network.linkSet[l].beta,
                                             network.linkSet[l].length,
                                             network.linkSet[l].flowHDV,
                                             network.linkSet[l].speedLimit,
                                             network.linkSet[l].toll
                                             )
        network.linkSet[l].cost = network.linkSet[l].costHDV + network.linkSet[l].costCAV


def updateTravelTimepre(network: FlowTransportNetwork, optimal: bool = False, costCAV=BPRcostFunctionCAV,
                        costHDV=BPRcostFunctionHDV):
    """
    This method updates the travel time on the links with the current flow
    """
    for l in network.linkSet:
        network.linkSet[l].precostCAV = costCAV(optimal,
                                                network.linkSet[l].fft,
                                                network.linkSet[l].alpha,
                                                network.linkSet[l].preflowCAV,
                                                network.linkSet[l].capacity,
                                                network.linkSet[l].beta,
                                                network.linkSet[l].length,
                                                network.linkSet[l].preflowHDV,
                                                network.linkSet[l].speedLimit,
                                                network.linkSet[l].toll
                                                )
        network.linkSet[l].precostHDV = costHDV(optimal,
                                                network.linkSet[l].fft,
                                                network.linkSet[l].alpha,
                                                network.linkSet[l].preflowCAV,
                                                network.linkSet[l].capacity,
                                                network.linkSet[l].beta,
                                                network.linkSet[l].length,
                                                network.linkSet[l].preflowHDV,
                                                network.linkSet[l].speedLimit,
                                                network.linkSet[l].toll
                                                )


def findAlpha(x_barCAV, x_barHDV, network: FlowTransportNetwork, optimal: bool = False,
              costFunction=BPRcostFunctionCAV):
    """
    This uses unconstrained optimization to calculate the optimal step size required
    for Frank-Wolfe Algorithm
    """

    def df(alpha):
        alpha = max(0, min(1, alpha))
        sum_derivative = 0  # this line is the derivative of the objective function.
        for l in network.linkSet:
            tmpFlowCAV = alpha * x_barCAV[l] + (1 - alpha) * network.linkSet[l].flowCAV
            tmpCostCAV = costFunction(optimal,
                                      network.linkSet[l].fft,
                                      network.linkSet[l].alpha,
                                      tmpFlowCAV,
                                      network.linkSet[l].capacity,
                                      network.linkSet[l].beta,
                                      network.linkSet[l].length,
                                      network.linkSet[l].flowHDV,
                                      network.linkSet[l].speedLimit,
                                      network.linkSet[l].toll
                                      )

            sum_derivative = sum_derivative + (x_barCAV[l] - network.linkSet[l].flowCAV) * tmpCostCAV
        return sum_derivative

    sol = fsolve(df, np.array([0.01]))
    return max(0, min(1, sol[0]))


def findAlphaHDV(x_barCAV, x_barHDV, network: FlowTransportNetwork, optimal: bool = False,
                 costFunction=BPRcostFunctionHDV):
    """
    This uses unconstrained optimization to calculate the optimal step size required
    for Frank-Wolfe Algorithm
    """

    def df(alphaHDV):
        alphaHDV = max(0, min(1, alphaHDV))
        sum_derivativeHDV = 0  # this line is the derivative of the objective function.
        for l in network.linkSet:
            tmpFlowHDV = alphaHDV * x_barHDV[l] + (1 - alphaHDV) * network.linkSet[l].flowHDV
            tmpCostHDV = costFunction(optimal,
                                      network.linkSet[l].fft,
                                      network.linkSet[l].alpha,
                                      network.linkSet[l].flowCAV,
                                      network.linkSet[l].capacity,
                                      network.linkSet[l].beta,
                                      network.linkSet[l].length,
                                      tmpFlowHDV,
                                      network.linkSet[l].speedLimit,
                                      network.linkSet[l].toll
                                      )
            sum_derivativeHDV = sum_derivativeHDV + (x_barHDV[l] - network.linkSet[l].flowHDV) * tmpCostHDV
        return sum_derivativeHDV

    sol = fsolve(df, np.array([0.01]))
    return max(0, min(1, sol[0]))


def findAlphaCAV(x_barCAV, x_barHDV, network: FlowTransportNetwork, optimal: bool = False,
                    costFunction=BPRcostFunctionCAV):
    """
    This uses unconstrained optimization to calculate the optimal step size required
    for Frank-Wolfe Algorithm
    """

    def df(alpha):
        alpha = max(0, min(1, alpha))
        sum_derivative = 0  # this line is the derivative of the objective function.
        for l in network.linkSet:
            tmpFlowCAV = alpha * x_barCAV[l] + (1 - alpha) * network.linkSet[l].flowCAV
            tmpCostCAV = costFunction(optimal,
                                      network.linkSet[l].fft,
                                      network.linkSet[l].alpha,
                                      tmpFlowCAV,
                                      network.linkSet[l].capacity,
                                      network.linkSet[l].beta,
                                      network.linkSet[l].length,
                                      network.linkSet[l].flowHDV,
                                      network.linkSet[l].speedLimit,
                                      network.linkSet[l].toll
                                      )

            sum_derivative = sum_derivative + (x_barCAV[l] - network.linkSet[l].flowCAV) * tmpCostCAV
        return sum_derivative

    sol = fsolve(df, np.array([0.01]))
    return max(0, min(1, sol[0]))


def findAlphaPRE(x_barCAV, x_barHDV, network: FlowTransportNetwork, optimal: bool = False,
                 costFunction=BPRcostFunctionHDV, costFunctionCAV=BPRcostFunctionCAV):
    """
    This uses unconstrained optimization to calculate the optimal step size required
    for Frank-Wolfe Algorithm
    """

    def df(alpha):
        alpha = max(0, min(1, alpha))
        sum_derivative = 0  # this line is the derivative of the objective function.
        for l in network.linkSet:
            tmpFlowCAV = alpha * x_barCAV[l] + (1 - alpha) * network.linkSet[l].flowCAV
            tmpFlowHDV = alpha * x_barHDV[l] + (1 - alpha) * network.linkSet[l].flowHDV
            tmpCostHDV = costFunction(optimal,
                                      network.linkSet[l].fft,
                                      network.linkSet[l].alpha,
                                      tmpFlowCAV,
                                      network.linkSet[l].capacity,
                                      network.linkSet[l].beta,
                                      network.linkSet[l].length,
                                      tmpFlowHDV,
                                      network.linkSet[l].speedLimit,
                                      network.linkSet[l].toll
                                      )
            tmpCostCAV = costFunctionCAV(optimal,
                                         network.linkSet[l].fft,
                                         network.linkSet[l].alpha,
                                         tmpFlowCAV,
                                         network.linkSet[l].capacity,
                                         network.linkSet[l].beta,
                                         network.linkSet[l].length,
                                         tmpFlowHDV,
                                         network.linkSet[l].speedLimit,
                                         network.linkSet[l].toll
                                         )
            sum_derivative = sum_derivative + (x_barHDV[l] - network.linkSet[l].flowHDV) * (tmpCostHDV) + (
                    x_barCAV[l] - network.linkSet[l].flowCAV) * (tmpCostCAV)

        return sum_derivative

    sol = fsolve(df, np.array([0.005]))
    return max(0, min(1, sol[0]))


def tracePredsCAV(destCAV, network: FlowTransportNetwork):
    """
    This method traverses predecessor nodes in order to create a shortest path
    """
    prevNodeCAV = network.nodeSet[destCAV].predCAV
    spLinksCAV = []
    while prevNodeCAV is not None:
        spLinksCAV.append((prevNodeCAV, destCAV))
        destCAV = prevNodeCAV
        prevNodeCAV = network.nodeSet[destCAV].predCAV
        # print(spLinksCAV)
    return spLinksCAV


def tracePredsHDV(destHDV, network: FlowTransportNetwork):
    """
    This method traverses predecessor nodes in order to create a shortest path
    """
    prevNodeHDV = network.nodeSet[destHDV].predHDV
    spLinksHDV = []
    while prevNodeHDV is not None:
        spLinksHDV.append((prevNodeHDV, destHDV))
        destHDV = prevNodeHDV
        prevNodeHDV = network.nodeSet[destHDV].predHDV
    return spLinksHDV


def loadAONCAV(network: FlowTransportNetwork, computeXbar: bool = True):
    """
    This method produces auxiliary flows (CAV) for all or nothing loading.
    """
    x_barCAV = {l: 0.0 for l in network.linkSet}
    SPTTCAV = 0.0
    for r in network.originZones:
        DijkstraHeapCAV(r, network=network)
        for s in network.zoneSet[r].destList:
            '''define dem for multiclass demand'''
            demCAV = network.tripSet[r, s].demandCAV
            '''similarly applied to multiclass traffic distribution though I'm not sure what this is used for'''
            if demCAV <= 0:
                continue
            '''   ???   '''
            SPTTCAV = SPTTCAV + network.nodeSet[s].label8 * demCAV
            # print(SPTTCAV)
            if computeXbar and r != s:
                for spLinksCAV in tracePredsCAV(s, network):
                    x_barCAV[spLinksCAV] = x_barCAV[spLinksCAV] + demCAV

    return SPTTCAV, x_barCAV


def loadAONHDV(network: FlowTransportNetwork, computeXbar: bool = True):
    """
    This method produces auxiliary flows (HDV) for all or nothing loading.
    """
    x_barHDV = {l: 0.0 for l in network.linkSet}
    SPTTHDV = 0.0
    for r in network.originZones:
        DijkstraHeapHDV(r, network=network)
        for s in network.zoneSet[r].destList:
            '''define dem for multiclass demand'''
            demHDV = network.tripSet[r, s].demandHDV
            # print(demHDV)
            '''similarly applied to multiclass traffic distribution though I'm not sure what this is used for'''
            if demHDV <= 0:
                continue
            '''   ???   '''
            SPTTHDV = SPTTHDV + network.nodeSet[s].label7 * demHDV
            # print('SPTTHDV=', SPTTHDV)
            if computeXbar and r != s:
                for spLinksHDV in tracePredsHDV(s, network):
                    x_barHDV[spLinksHDV] = x_barHDV[spLinksHDV] + demHDV

    return SPTTHDV, x_barHDV


def readDemand(demand_df: pd.DataFrame, network: FlowTransportNetwork):
    for index, row in demand_df.iterrows():

        init_node = str(int(row["init_node"]))
        term_node = str(int(row["term_node"]))
        demand = row["demand"]
        demandCAV = demand * CAV_proportion
        demandHDV = demand - demandCAV

        network.tripSet[init_node, term_node] = Demand(init_node, term_node, demand, demandCAV, demandHDV)
        if init_node not in network.zoneSet:
            network.zoneSet[init_node] = Zone(init_node)
        if term_node not in network.zoneSet:
            network.zoneSet[term_node] = Zone(term_node)
        if term_node not in network.zoneSet[init_node].destList:
            network.zoneSet[init_node].destList.append(term_node)

    print(len(network.tripSet), "OD pairs")
    print(len(network.zoneSet), "OD zones")


def readNetwork(network_df: pd.DataFrame, network: FlowTransportNetwork):
    for index, row in network_df.iterrows():

        init_node = str(int(row["init_node"]))
        term_node = str(int(row["term_node"]))
        capacity = row["capacity"]
        length = row["length"]
        free_flow_time = row["free_flow_time"]
        b = row["b"]
        power = row["power"]
        speed = row["speed"]
        toll = row["toll"]
        link_type = row["link_type"]

        network.linkSet[init_node, term_node] = Link(init_node=init_node,
                                                     term_node=term_node,
                                                     capacity=capacity,
                                                     length=length,
                                                     fft=free_flow_time,
                                                     b=b,
                                                     power=power,
                                                     speed_limit=speed,
                                                     toll=toll,
                                                     linkType=link_type
                                                     )
        if init_node not in network.nodeSet:
            network.nodeSet[init_node] = Node(init_node)
        if term_node not in network.nodeSet:
            network.nodeSet[term_node] = Node(term_node)
        if term_node not in network.nodeSet[init_node].outLinks:
            network.nodeSet[init_node].outLinks.append(term_node)
        if init_node not in network.nodeSet[term_node].inLinks:
            network.nodeSet[term_node].inLinks.append(init_node)

    print(len(network.nodeSet), "nodes")
    print(len(network.linkSet), "links")


def get_TSTT(network: FlowTransportNetwork, costCAV=BPRcostFunctionCAV, costHDV=BPRcostFunctionHDV,
             use_max_capacity: bool = False):

    TSTTCAV = round(sum([network.linkSet[a].flowCAV * costCAV(optimal=False,
                                                              fft=network.linkSet[
                                                                  a].fft,
                                                              alpha=network.linkSet[
                                                                  a].alpha,
                                                              flowCAV=network.linkSet[
                                                                  a].flowCAV,
                                                              flowHDV=network.linkSet[
                                                                  a].flowHDV,
                                                              capacity=network.linkSet[
                                                                  a].max_capacity if use_max_capacity else
                                                              network.linkSet[
                                                                  a].capacity,
                                                              beta=network.linkSet[
                                                                  a].beta,
                                                              length=network.linkSet[
                                                                  a].length,
                                                              maxSpeed=network.linkSet[
                                                                  a].speedLimit,
                                                              toll=network.linkSet[
                                                                  a].toll
                                                              ) for a in
                         network.linkSet]), 9)

    TSTTHDV = round(sum([network.linkSet[a].flowHDV * costHDV(optimal=False,
                                                              fft=network.linkSet[
                                                                  a].fft,
                                                              alpha=network.linkSet[
                                                                  a].alpha,
                                                              flowHDV=network.linkSet[
                                                                  a].flowHDV,
                                                              flowCAV=network.linkSet[
                                                                  a].flowCAV,
                                                              capacity=network.linkSet[
                                                                  a].max_capacity if use_max_capacity else
                                                              network.linkSet[
                                                                  a].capacity,
                                                              beta=network.linkSet[
                                                                  a].beta,
                                                              length=network.linkSet[
                                                                  a].length,
                                                              maxSpeed=network.linkSet[
                                                                  a].speedLimit,
                                                              toll=network.linkSet[
                                                                  a].toll
                                                              ) for a in
                         network.linkSet]), 9)
    TSTT = TSTTCAV + TSTTHDV
    return TSTT, TSTTCAV, TSTTHDV


def assignment_loop(network: FlowTransportNetwork,
                    algorithm: str = "FW",
                    systemOptimal: bool = False,
                    accuracy: float = 0.01,
                    maxIter: int = 1000,
                    maxTime: int = 60,
                    verbose: bool = True):
    """
    For explaination of the algorithm see Chapter 7 of:
    https://sboyles.github.io/blubook.html
    PDF:
    https://sboyles.github.io/teaching/ce392c/book.pdf
    """
    network.reset_flow()

    iteration_number = 1
    gap = np.inf
    TSTT = np.inf
    assignmentStartTime = time.time()
    # Check if desired accuracy is reached
    while gap > accuracy:

        # Get x_bar through all-or-nothing assignment
        # _, x_bar = loadAON(network=network)
        _, x_barCAV = loadAONCAV(network=network)
        _, x_barHDV = loadAONHDV(network=network)

        if algorithm == "MSA" or iteration_number == 1:
            alphaCAV = (1 / iteration_number)
            alphaHDV = (1 / iteration_number)
            alpha = alphaCAV

            for l in network.linkSet:
                '''Initial flow SO (CAV)'''
                network.linkSet[l].flowCAV = alpha * x_barCAV[l] + (1 - alpha) * network.linkSet[l].flowCAV

        elif algorithm == "FW":
            # If using Frank-Wolfe determine the step size alpha by solving a nonlinear equation
            '''alphaCAV = findAlpha(x_barCAV,
                                 x_barHDV,
                                 network=network,
                                 optimal=systemOptimal,
                                 costFunction=BPRcostFunctionCAV)
            alphaHDV = findAlphaHDV(x_barCAV,
                                    x_barHDV,
                                    network=network,
                                    optimal=systemOptimal,
                                    costFunction=BPRcostFunctionHDV)

            alpha = min(alphaCAV, alphaHDV)
            if alpha == 0:
                alpha = max(alphaCAV, alphaHDV)'''
            alpha = findAlphaHDV(x_barCAV,
                                 x_barHDV,
                                 network=network,
                                 optimal=systemOptimal,
                                 costFunction=BPRcostFunctionHDV)
            # print(alpha)
        else:
            print("Terminating the program.....")
            print("The solution algorithm ", algorithm, " does not exist!")
            raise TypeError('Algorithm must be MSA or FW')

        # Apply flow improvement
        for l in network.linkSet:
            '''express the flow of CAV, HDV and total flow in the next iteration'''
            '''note that only the HDV flows are renewed, and the CAVs needs to have alphaCAV calculated below'''
            network.linkSet[l].preflowCAV = network.linkSet[l].flowCAV
            network.linkSet[l].preflowHDV = network.linkSet[l].flowHDV
            network.linkSet[l].flowHDV = alpha * x_barHDV[l] + (1 - alpha) * network.linkSet[l].flowHDV
            network.linkSet[l].flow = network.linkSet[l].flowCAV + network.linkSet[l].flowHDV

        '''INER-LOOP FOR SO(CAV) PLAYERS'''
        if algorithm == "FW":
            alphaSO = findAlphaCAV(x_barCAV,
                                 x_barHDV,
                                 network=network,
                                 optimal=systemOptimal,
                                 costFunction=BPRcostFunctionCAV)
            for l in network.linkSet:

                network.linkSet[l].flowCAV = alphaSO * x_barCAV[l] + (1 - alphaSO) * network.linkSet[l].flowCAV

        #print(alpha)
        #print(x_barCAV)
        # Compute the new travel time
        updateTravelTimepre(network=network,
                            optimal=systemOptimal,
                            costCAV=BPRcostFunctionCAV,
                            costHDV=BPRcostFunctionHDV)

        updateTravelTime(network=network,
                         optimal=systemOptimal,
                         costCAV=BPRcostFunctionCAV,
                         costHDV=BPRcostFunctionHDV)
        # Compute the relative gap
        SPTTCAV, _ = loadAONCAV(network=network, computeXbar=False)
        SPTTCAV = round(SPTTCAV, 12)
        SPTTHDV, _ = loadAONHDV(network=network, computeXbar=False)
        SPTTHDV = round(SPTTHDV, 12)
        TSTT = round(sum([network.linkSet[a].flowHDV * network.linkSet[a].costHDV for a in
                          network.linkSet] + [network.linkSet[a].flowCAV * network.linkSet[a].costCAV for a in
                                              network.linkSet]), 12)
        lanesgapHDV = math.pow(round(sum([network.linkSet[a].flowHDV - network.linkSet[a].preflowHDV for a in
                                          network.linkSet]), 12), 2)
        lanesgapCAV = math.pow(round(sum([network.linkSet[a].flowCAV - network.linkSet[a].preflowCAV for a in
                                          network.linkSet]), 12), 2)

        lanesflowCAV = round(sum([network.linkSet[a].flowCAV for a in network.linkSet]), 12)
        lanesflowHDV = round(sum([network.linkSet[a].flowHDV for a in network.linkSet]), 12)

        SPTT = SPTTHDV + SPTTCAV
        # print(TSTT, SPTT, SPTTCAV, SPTTHDV, "Max capacity", max([l.capacity for l in network.linkSet.values()]))
        if CAV_proportion ==1:
            gap=TSTT/SPTT-1
        elif lanesflowCAV == 0:
            gap = math.pow(lanesgapHDV, 0.5) / lanesflowHDV
        elif lanesflowHDV == 0:
            gap = math.pow(lanesgapCAV, 0.5) / lanesflowCAV
        else:
            gap = 0.5 * math.pow(lanesgapCAV, 0.5) / lanesflowCAV + 0.5 * math.pow(lanesgapHDV, 0.5) / lanesflowHDV


        if gap < 0:
            print("Error, gap is less than 0, this should not happen")
            print("TSTT", "SPTT", TSTT, SPTT)

            # Uncomment for debug

            # print("Capacities:", [l.capacity for l in network.linkSet.values()])
            print("FlowsHDV:", [l.flowHDV for l in network.linkSet.values()])
            print("FlowsCAV:", [l.flowCAV for l in network.linkSet.values()])

        # Compute the real total travel time (which in the case of system optimal rounting is different from the TSTT above)
        TSTT, TSTTCAV,TSTTHDV = get_TSTT(network=network, costCAV=BPRcostFunctionCAV, costHDV=BPRcostFunctionHDV)

        iteration_number += 1
        if iteration_number > maxIter:
            if verbose:
                print(
                    "The assignment did not converge to the desired gap and the max number of iterations has been reached")
                print("Assignment took", round(time.time() - assignmentStartTime, 5), "seconds")
                print("Current gap:", round(gap, 5))
            return TSTT
        if time.time() - assignmentStartTime > maxTime:
            if verbose:
                print("The assignment did not converge to the desired gap and the max time limit has been reached")
                print("Assignment did ", iteration_number, "iterations")
                print("Current gap:", round(gap, 5))
            return TSTT

    if verbose:
        print("Assignment converged in ", iteration_number, "iterations")
        print("Assignment took", round(time.time() - assignmentStartTime, 5), "seconds")
        print("Current gap:", round(gap, 12))
    return TSTT, iteration_number


def writeResults(network: FlowTransportNetwork, output_file: str, costCAV=BPRcostFunctionCAV,
                 costHDV=BPRcostFunctionHDV,
                 systemOptimal: bool = False, verbose: bool = True):
    outFile = open(output_file, "w")
    TSTT,TSTTCAV,TSTTHDV = get_TSTT(network=network, costCAV=BPRcostFunctionCAV, costHDV=BPRcostFunctionHDV)
    if verbose:
        print("\nTotal system travel time:", f'{TSTT} secs')
    tmpOut = "Total Travel Time:\t" + str(TSTT)
    outFile.write(tmpOut + "\n")
    tmpOut = "Cost function used:\t" + costCAV.__name__
    outFile.write(tmpOut + "\n")
    tmpOut = ["User equilibrium (UE) or system optimal (SO):\t"] + ["SO" if systemOptimal else "UE"]
    outFile.write("".join(tmpOut) + "\n\n")
    tmpOut = "init_node\tterm_node\tflowHDV\tflowCAV\ttravelTimeHDV\ttravelTimeCAV"
    outFile.write(tmpOut + "\n")
    for i in network.linkSet:
        tmpOut = str(network.linkSet[i].init_node) + "\t" + str(
            network.linkSet[i].term_node) + "\t" + str(
            network.linkSet[i].flowHDV) + "\t" + str(
            network.linkSet[i].flowCAV) + "\t" + str(costHDV(False,
                                                             network.linkSet[i].fft,
                                                             network.linkSet[i].alpha,
                                                             network.linkSet[i].flowCAV,
                                                             network.linkSet[i].max_capacity,
                                                             network.linkSet[i].beta,
                                                             network.linkSet[i].length,
                                                             network.linkSet[i].flowHDV,
                                                             network.linkSet[i].speedLimit,
                                                             network.linkSet[i].toll
                                                             )) + "\t" + str(costCAV(False,
                                                                                     network.linkSet[i].fft,
                                                                                     network.linkSet[i].alpha,
                                                                                     network.linkSet[i].flowCAV,
                                                                                     network.linkSet[i].max_capacity,
                                                                                     network.linkSet[i].beta,
                                                                                     network.linkSet[i].length,
                                                                                     network.linkSet[i].flowHDV,
                                                                                     network.linkSet[i].speedLimit,
                                                                                     network.linkSet[i].toll
                                                                                     ))
        outFile.write(tmpOut + "\n")
    outFile.close()
    return network


def load_network(net_file: str,
                 demand_file: str = None,
                 force_net_reprocess: bool = False,
                 verbose: bool = True
                 ) -> FlowTransportNetwork:
    readStart = time.time()

    if demand_file is None:
        demand_file = '_'.join(net_file.split("_")[:-1] + ["trips.tntp"])

    net_name = net_file.split("/")[-1].split("_")[0]

    if verbose:
        print(f"Loading network {net_name}...")

    net_df, demand_df = import_network(
        net_file,
        demand_file,
        force_reprocess=force_net_reprocess
    )

    network = FlowTransportNetwork()

    readDemand(demand_df, network=network)
    readNetwork(net_df, network=network)

    network.originZones = set([k[0] for k in network.tripSet])

    if verbose:
        print("Network", net_name, "loaded")
        print("Reading the network data took", round(time.time() - readStart, 2), "secs\n")

    return network


def computeAssingment(net_file: str,
                      demand_file: str = None,
                      algorithm: str = "MSA",  # FW or MSA
                      systemOptimal: bool = False,
                      accuracy: float = 0.005,
                      maxIter: int = 1000,
                      maxTime: int = 60,
                      results_file: str = None,
                      force_net_reprocess: bool = False,
                      verbose: bool = True
                      ) -> tuple[Any, FlowTransportNetwork, int]:
    """
    This is the main function to compute the user equilibrium UE (default) or system optimal (SO) traffic assignment
    All the networks present on https://github.com/bstabler/TransportationNetworks following the tntp format can be loaded


    :param net_file: Name of the network (net) file following the tntp format (see https://github.com/bstabler/TransportationNetworks)
    :param demand_file: Name of the demand (trips) file following the tntp format (see https://github.com/bstabler/TransportationNetworks), leave None to use dafault demand file
    :param algorithm:
           - "FW": Frank-Wolfe algorithm (see https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm)
           - "MSA": Method of successive averages
           For more information on how the algorithms work see https://sboyles.github.io/teaching/ce392c/book.pdf
    :param costFunction: Which cost function to use to compute travel time on edges, currently available functions are:
           - BPRcostFunction (see https://rdrr.io/rforge/travelr/man/bpr.function.html)
           - greenshieldsCostFunction (see Greenshields, B. D., et al. "A study of traffic capacity." Highway research board proceedings. Vol. 1935. National Research Council (USA), Highway Research Board, 1935.)
           - constantCostFunction
    :param systemOptimal: Wheather to compute the system optimal flows instead of the user equilibrium
    :param accuracy: Desired assignment precision gap
    :param maxIter: Maximum nuber of algorithm iterations
    :param maxTime: Maximum seconds allowed for the assignment
    :param results_file: Name of the desired file to write the results,
           by default the result file is saved with the same name as the input network with the suffix "_flow.tntp" in the same folder
    :param force_net_reprocess: True if the network files should be reprocessed from the tntp sources
    :param verbose: print useful info in standard output
    :return: Totoal system travel time
    """

    network = load_network(net_file=net_file, demand_file=demand_file, verbose=verbose,
                           force_net_reprocess=force_net_reprocess)

    if verbose:
        print("Computing assignment...")
    TSTT, iteration_number = assignment_loop(network=network, algorithm=algorithm, systemOptimal=systemOptimal,
                           accuracy=accuracy, maxIter=maxIter, maxTime=maxTime, verbose=verbose)
    TSTT,TSTTCAV,TSTTHDV= get_TSTT(network=network, costCAV=BPRcostFunctionCAV, costHDV=BPRcostFunctionHDV)

    if results_file is None:
        results_file = '_'.join(net_file.split("_")[:-1] + ["flow.tntp"])

    output = writeResults(network=network,
                          output_file=results_file,
                          systemOptimal=systemOptimal,
                          verbose=verbose)

    return TSTT, output,iteration_number,TSTTCAV,TSTTHDV





if __name__ == '__main__':
    # This is an example usage for calculating System Optimal and User Equilibrium with Frank-Wolfe
    names = locals()
    net_file1 = str(PathUtils.sioux_falls_lanes_net_file)


    tf = np.arange(0,0.65,0.09)
    #cp=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    cp=[0.1,0.15,0.25,0.35,0.5,0.65,0.8,0.9]


    TSTC=[]
    zindex = []
    VOTCAV = 0
    CAV_proportion = 0.5
    loopcount=0
    VOTHDV=0
    toll_index=1
    tollportion = 0.5
    linkflow26=[]
    linkflow43=[]
    linkflow67=[]
    linkflow65=[]
    intensity=1
    for i in tf:
        for o in cp:

            tollfactor = i
            tollportion = o


            total_system_travel_time_optimal, outputSO, iteration_numberSO,TSTTCAV,TSTTHDV = computeAssingment(net_file=net_file1,
                                                                                                           algorithm="FW",
                                                                                                           systemOptimal=True,
                                                                                                           verbose=True,
                                                                                                           accuracy=5e-5,
                                                                                                           maxIter=80000,
                                                                                                           maxTime=6000000)
            TSTC=np.append(TSTC,TSTTHDV)



            loopcount=loopcount+1
            print(loopcount)

fig = plt.figure()
ax = plt.axes(projection='3d')
xv,TOLLFACTOR=np.meshgrid(cp,tf)
Z=np.reshape(TSTC,(8,8))



ax.plot_surface(xv, TOLLFACTOR, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ax.set_title('Total Revenue');

ax.set_xlabel('Ml Portion')
ax.set_ylabel('Toll Rate')
ax.set_zlabel('Revenue');

plt.show()


'''
fig = plt.figure()

x = tf

y1=linkflow26
y2=linkflow43
y3=linkflow67
y4=linkflow65



plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)

plt.legend(['ML on Link26','ML on Link43','ML on Link67','ML on Link65'],loc='upper right')



#plt.plot(x1 , y2, ms=4,marker='*', color='red')
#plt.legend(['TSTT_UE','TSTT_MULTI'],loc='upper right')
plt.title('HDV Users on ML')
plt.xlabel('Toll Rates')
plt.ylabel('HDV Flow')

plt.show()
'''

'''
linkflow26 = np.append(linkflow26, names.get('extHDV' + str(1)))
linkflow43 = np.append(linkflow43, names.get('extHDV' + str(5)))
linkflow67 = np.append(linkflow67, names.get('extHDV' + str(9)))
linkflow65 = np.append(linkflow65, names.get('extHDV' + str(13)))
'''
'''
count = 0
for i in outputSO.linkSet:
    count = count + 1
    names['extHDV' + str(count)] = float(str(outputSO.linkSet[i].flowHDV))
    names['extCAV' + str(count)] = float(str(outputSO.linkSet[i].flowCAV))
    names['extcostHDV' + str(count)] = float(str(outputSO.linkSet[i].costHDV))
    names['extcostCAV' + str(count)] = float(str(outputSO.linkSet[i].costCAV))
    if count >= 93:
        break

linkflow26 = np.append(linkflow26, names.get('extHDV' + str(3)))
linkflow43 = np.append(linkflow43, names.get('extHDV' + str(7)))
linkflow67 = np.append(linkflow67, names.get('extHDV' + str(11)))
linkflow65 = np.append(linkflow65, names.get('extHDV' + str(15)))
'''


'''
fig = plt.figure()
ax = plt.axes(projection='3d')
xv,TOLLFACTOR=np.meshgrid(cp,tf)
Z=np.reshape(TSTC,(9,9))



ax.plot_surface(xv, TOLLFACTOR, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ax.set_title('Toll Revenue');

ax.set_xlabel('ML Portion')
ax.set_ylabel('Toll Rate')
ax.set_zlabel('Revenue');

plt.show()
'''

'''count = 0
for i in outputSO.linkSet:
    count = count + 1
    names['extHDV' + str(count)] = float(str(outputSO.linkSet[i].flowHDV))
    names['extCAV' + str(count)] = float(str(outputSO.linkSet[i].flowCAV))
    names['extcostHDV' + str(count)] = float(str(outputSO.linkSet[i].costHDV))
    names['extcostCAV' + str(count)] = float(str(outputSO.linkSet[i].costCAV))
    if count >= 23:
        break

linkflow1 = np.append(linkflow1, names.get('extHDV' + str(8)))
linkflow2 = np.append(linkflow2, names.get('extHDV' + str(14)))'''

'''    for i in tmp:
        tollfactor = 0.1
        tollportion = 0.3
        CAV_proportion = 0.5
        intensity=i
        total_system_travel_time_optimal, outputSO, iteration_numberSO  = computeAssingment(net_file=net_file1,
                                                                       algorithm="FW",
                                                                       systemOptimal=True,
                                                                       verbose=True,
                                                                       accuracy=2e-4,
                                                                       maxIter=8000,
                                                                       maxTime=6000000)
        TSTTSO3 = np.append(TSTTSO3, total_system_travel_time_optimal)

        tollfactor = 0
        tollportion = 0.3
        CAV_proportion = 0.5
        intensity=i
        total_system_travel_time_optimal, outputSO, iteration_numberSO  = computeAssingment(net_file=net_file1,
                                                                       algorithm="FW",
                                                                       systemOptimal=True,
                                                                       verbose=True,
                                                                       accuracy=2e-4,
                                                                       maxIter=8000,
                                                                       maxTime=6000000)
        TSTTSO4 = np.append(TSTTSO4, total_system_travel_time_optimal)

    for i in tmp:
        tollfactor = 0.1
        tollportion = 0.4
        CAV_proportion = 0.5
        intensity=i
        total_system_travel_time_optimal, outputSO, iteration_numberSO  = computeAssingment(net_file=net_file1,
                                                                       algorithm="FW",
                                                                       systemOptimal=True,
                                                                       verbose=True,
                                                                       accuracy=2e-4,
                                                                       maxIter=8000,
                                                                       maxTime=6000000)
        TSTTSO5 = np.append(TSTTSO5, total_system_travel_time_optimal)

        tollfactor = 0
        tollportion = 0.4
        CAV_proportion = 0.5
        intensity=i
        total_system_travel_time_optimal, outputSO, iteration_numberSO  = computeAssingment(net_file=net_file1,
                                                                       algorithm="FW",
                                                                       systemOptimal=True,
                                                                       verbose=True,
                                                                       accuracy=2e-4,
                                                                       maxIter=8000,
                                                                       maxTime=6000000)
        TSTTSO6 = np.append(TSTTSO6, total_system_travel_time_optimal)


    for i in tmp:
        tollfactor = 0.1
        tollportion = 0.5
        CAV_proportion = 0.5
        intensity=i
        total_system_travel_time_optimal, outputSO, iteration_numberSO  = computeAssingment(net_file=net_file1,
                                                                       algorithm="FW",
                                                                       systemOptimal=True,
                                                                       verbose=True,
                                                                       accuracy=2e-4,
                                                                       maxIter=8000,
                                                                       maxTime=6000000)
        TSTTSO7 = np.append(TSTTSO7, total_system_travel_time_optimal)

        tollfactor = 0
        tollportion = 0.5
        CAV_proportion = 0.5
        intensity=i
        total_system_travel_time_optimal, outputSO, iteration_numberSO  = computeAssingment(net_file=net_file1,
                                                                       algorithm="FW",
                                                                       systemOptimal=True,
                                                                       verbose=True,
                                                                       accuracy=2e-4,
                                                                       maxIter=8000,
                                                                       maxTime=6000000)
        TSTTSO8 = np.append(TSTTSO8, total_system_travel_time_optimal)'''

'''for i in range(1, 24):
    names['linkflowHDV' + str(i)] = []
    names['linkflowCAV' + str(i)] = []
    names['linkcostCAV' + str(i)] = []
    names['linkcostHDV' + str(i)] = []
    names['TSTTSO' + str(i)] = []'''
'''TSTTgap1=TSTTSO2-TSTTSO1
TSTTgap2=TSTTSO4-TSTTSO3
TSTTgap3=TSTTSO6-TSTTSO5
TSTTgap4=TSTTSO8-TSTTSO7

print(TSTTgap1)
print(TSTTgap2)
print(tmp)

fig = plt.figure()

x = tmp

y1=TSTTgap1
y2=TSTTgap2
y3=TSTTgap3
y4=TSTTgap4



plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)

plt.legend(['Ea=0.2','Ea=0.3','Ea=0.4','Ea=0.5'],loc='upper right')



#plt.plot(x1 , y2, ms=4,marker='*', color='red')
#plt.legend(['TSTT_UE','TSTT_MULTI'],loc='upper right')
plt.title('TSTT Difference')
plt.xlabel('CAV PROP')
plt.ylabel('TSTT Diff')

plt.show()'''

'''count = 0
for i in outputSO.linkSet:
    count = count + 1
    names['extHDV' + str(count)] = float(str(outputSO.linkSet[i].flowHDV))
    names['extCAV' + str(count)] = float(str(outputSO.linkSet[i].flowCAV))
    names['extcostHDV' + str(count)] = float(str(outputSO.linkSet[i].costHDV))
    names['extcostCAV' + str(count)] = float(str(outputSO.linkSet[i].costCAV))
    if count >= 23:
        break'''



'''TSTTSO=[TSTTSO1,TSTTSO2,TSTTSO3,TSTTSO4]
fig1, ax1 = plt.subplots()
ax1.set_title('TSTT Distribution due to Platooning Intensity with Different ML Portions')
ax1.set_xlabel('Ea Cases')
ax1.set_ylabel('TSTT')
labels=['Ea=0.2','Ea=0.3','Ea=0.4','Ea=0.5']
ax1.boxplot(TSTTSO,labels=labels,showfliers=False)

plt.show()
'''



'''        total_system_travel_time_equilibrium, outputUE, iteration_numberUE = computeAssingment(net_file=net_file,
                                                                           algorithm="FW",
                                                                           systemOptimal=False,
                                                                           verbose=True,
                                                                           accuracy=1e-4,
                                                                           maxIter=5000,
                                                                           maxTime=6000000)
        TSTTUE = np.append(TSTTUE, total_system_travel_time_equilibrium)
        TSTTSO = np.append(TSTTSO, total_system_travel_time_optimal)
        TSTTGAP=TSTTUE-TSTTSO'''
        #iteration_number_listFW = np.append(iteration_number_listFW, iteration_numberFW)
        #iteration_number_listMSA = np.append(iteration_number_listMSA, iteration_numberMSA)
    # print("UE - SO = ", total_system_travel_time_equilibrium - total_system_travel_time_optimal)


'''        count = 0
        for i in outputUE.linkSet:
            count = count + 1
            names['extHDV'+str(count)] = float(str(outputUE.linkSet[i].flowHDV))
            names['extCAV'+str(count)] = float(str(outputUE.linkSet[i].flowCAV))
            names['extcostHDV' + str(count)] = float(str(outputUE.linkSet[i].costHDV))
            names['extcostCAV' + str(count)] = float(str(outputUE.linkSet[i].costCAV))
            if count >=19:
                break

        for i in [5]:
            names['linkflowHDV' + str(i)] = np.append(names.get('linkflowHDV' + str(i)), names.get('extHDV' + str(i)))
            names['linkflowCAV' + str(i)] = np.append(names.get('linkflowCAV' + str(i)), names.get('extCAV' + str(i)))'''





'''        for i in [1,5,7,9,11]:
            ODCOST_SUB_1 = ODCOST_SUB_1 + names.get('extcostHDV' + str(i)) + names.get('extcostCAV' + str(i))


        for i in [1,5,7,10,16]:
            ODCOST_SUB_2 = ODCOST_SUB_2 + names.get('extcostHDV' + str(i)) + names.get('extcostCAV' + str(i))

        for i in [3,5,7,9,11]:
            ODCOST_SUB_3 = ODCOST_SUB_3 + names.get('extcostHDV' + str(i)) + names.get('extcostCAV' + str(i))
        for i in [3,5,7,10,16]:
            ODCOST_SUB_4 = ODCOST_SUB_4 + names.get('extcostHDV' + str(i)) + names.get('extcostCAV' + str(i))
        ODCOST_1 = np.append(ODCOST_1, ODCOST_SUB_1)
        ODCOST_2 = np.append(ODCOST_2, ODCOST_SUB_2)
        ODCOST_3 = np.append(ODCOST_3, ODCOST_SUB_3)
        ODCOST_4 = np.append(ODCOST_4, ODCOST_SUB_4)

        for i in [1]:
            zvalue1=zvalue1+names.get('extcostHDV' + str(i)) + names.get('extcostCAV' + str(i))

        zvalue=zvalue1+names.get('extcostHDV' + str(4))'''




'''        for i in [5]:
            names['linkflowHDV' + str(i)] = np.append(names.get('linkflowHDV' + str(i)), names.get('extHDV' + str(i)))
            names['linkflowCAV' + str(i)] = np.append(names.get('linkflowCAV' + str(i)), names.get('extCAV' + str(i)))
            print(names.get('linkflowHDV' + str(5)))'''



'''        TSTTLIST=np.append(TSTTLIST,total_system_travel_time_equilibrium)

        zindex=np.append(zindex,zvalue)'''






'''fig = plt.figure()

x1 = tmp

y1 = TSTTGAP'''


'''    y6 = names.get('linkflowCAV' + str(1))
    y7 = names.get('linkflowCAV' + str(2))
    y8 = names.get('linkflowCAV' + str(3))
    y9 = names.get('linkflowCAV' + str(4))
    y10 = names.get('linkflowCAV' + str(5))
    y11 = zindex'''

'''
plt.plot(x, y2)
    plt.plot(x, y3)
    plt.plot(x, y4,  marker='*', ms=5)p
    plt.plot(x, y5,  marker='o', ms=5)
    plt.plot(x, y6, linestyle='dotted')
    plt.plot(x, y7, linestyle='dotted')
    plt.plot(x, y8, linestyle='dotted')
    plt.plot(x, y9, linestyle='dotted')
    plt.plot(x, y10, linestyle='dotted')
    plt.legend(['Link1HDV','Link2/3HDV','Link4HDV','Link5HDV','Link1CAV','Link2CAV','Link3CAV','Link4CAV','Link5CAV'],loc='upper right')'''

'''plt.plot(x1 , y1,ms=4, color='orange')

#plt.plot(x1 , y2, ms=4,marker='*', color='red')
#plt.legend(['TSTT_UE','TSTT_MULTI'],loc='upper right')
plt.title('TSTT Difference')
plt.xlabel('CAV PROP')
plt.ylabel('TSTT Diff')

plt.show()'''

