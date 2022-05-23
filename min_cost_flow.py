#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 15:17:59 2022

@author: yanbing_wang
"""
import networkx as nx
import queue
from collections import deque 
import matplotlib.pyplot as plt
from i24_database_api.db_reader import DBReader
from i24_database_api.db_writer import DBWriter
import os
from i24_configparse.parse import parse_cfg
from utils.stitcher_module import min_nll_cost
from utils.data_structures import Fragment

# TODO:
    # 1. check if the answers agree with nx.edmond_karp
    # 2. add more intelligent enter/exiting cost based on the direction and where they are relative to the road
    
    
    
def read_to_queue(parameters):
    '''
    construct MOT graph from fragment list based on the specified loss function
    '''
    # connect to database
    raw = DBReader(host=parameters.default_host, port=parameters.default_port, username=parameters.readonly_user,   
                   password=parameters.default_password,
                   database_name=parameters.db_name, collection_name=parameters.raw_collection)
    print("connected to raw collection")
    gt = DBReader(host=parameters.default_host, port=parameters.default_port, username=parameters.readonly_user,   
                    password=parameters.default_password,
                    database_name=parameters.db_name, collection_name=parameters.gt_collection)
    print("connected to gt collection")
    stitched = DBWriter(host=parameters.default_host, port=parameters.default_port, 
            username=parameters.default_username, password=parameters.default_password,
            database_name=parameters.db_name, collection_name=parameters.stitched_collection,
            server_id=1, process_name=1, process_id=1, session_config_id=1, schema_file=None)
    stitched.collection.drop()
    
    # stitched_reader = DBReader(host=parameters.default_host, port=parameters.default_port, 
    #                            username=parameters.readonly_user, password=parameters.default_password,
    #                            database_name=parameters.db_name, collection_name=parameters.stitched_collection)
    print("connected to stitched collection")
    
    # specify ground truth ids and the corresponding fragment ids
    gt_ids = [7]
    fragment_ids = []
    gt_res = gt.read_query(query_filter = {"ID": {"$in": gt_ids}},
                            limit = 0)
    
    for gt_doc in gt_res:
        fragment_ids.extend(gt_doc["fragment_ids"])
    
    raw_res = raw.read_query(query_filter = {"_id": {"$in": fragment_ids}},
                              query_sort = [("last_timestamp", "ASC")])
    # raw_res = raw.read_query(query_filter = {"$and":[ {"last_timestamp": {"$gt": 545}}, 
    #                                                   {"last_timestamp": {"$lt": 580}},
    #                                                   {"_id": {"$in": fragment_ids}}]},
                               # query_sort = [("last_timestamp", "ASC")])
    
    # write fragments to queue
    fragment_queue = queue.Queue()
    # fragment_set = set()
    for doc in raw_res:
        fragment_queue.put(doc)
        # fragment_set.add(doc["_id"])
        
    fragment_size = fragment_queue.qsize()
    print("Queue size: ", fragment_size)

    return fragment_queue
    

class MOT_Graph:
    
    def __init__(self, inclusion = -0.1, threshold = 10, parameters = None):
        self.inclusion = inclusion
        self.threshold = threshold
        self.pos = None
        self.collapse_pos = None
        self.parameters = parameters
        
       
    def construct_graph(self):
        G = nx.DiGraph()
        G.add_edge("1-post", "2-pre", weight = 2, flipped=False)
        G.add_edge("1-post", "4-pre", weight = 3, flipped=False)      
        G.add_edge("1-post", "5-pre", weight = 6, flipped=False)      
        G.add_edge("3-post", "4-pre", weight = 4, flipped=False)
        G.add_edge("3-post", "2-pre", weight = 5, flipped=False)
        G.add_edge("4-post", "2-pre", weight = 9, flipped=False)
        G.add_edge("4-post", "5-pre", weight = 1, flipped=False)
        
     
        edge_list = list(G.edges())
        for (u,v) in edge_list: # make "low-cost" edges negative to incentivize stitching
            if G[u][v]["weight"] > self.threshold:
                G.remove_edge(u,v)
            else:
                G[u][v]["weight"] -= self.threshold
        
        # extract unique nodes numbers
        nodes = {s[0] for s in G.nodes}
        n = len(nodes)

        for i in range(1,n+1):
            # add inclusion edges
            G.add_edge(str(i)+"-pre", str(i)+"-post", weight = self.inclusion, flipped=False)
            # add source node and sink node
            G.add_edge("s", str(i)+"-pre", weight = 0, flipped=False)
            G.add_edge(str(i)+"-post", "t", weight = 0, flipped=False)
        
        return G
    
       
    def construct_graph_from_fragments(self, fragment_queue):
        '''
        Fragments are ordered by last_timestamp in queue
        nodes: id = fragment_id
        '''
        TIME_WIN = self.parameters.time_win
        VARX = self.parameters.varx
        VARY = self.parameters.vary

        
        G = nx.DiGraph()
        fragment_dict = {}
        edge_list = []
        
        while not fragment_queue.empty():
            new_fgmt = Fragment(fragment_queue.get())
            new_fgmt.compute_stats()
            fragment_dict[new_fgmt.ID] = new_fgmt 
            for fgmt_id, fgmt in fragment_dict.items():
                cost = min_nll_cost(fgmt, new_fgmt, TIME_WIN, VARX, VARY)
                if cost < self.threshold and cost > -999:
                    edge_list.append(((fgmt.ID, new_fgmt.ID),cost))
         
            
        # add transition edges
        for e,c in edge_list:
            if e[0] != e[1]:
                G.add_edge(str(e[0]) + "-post", str(e[1]) + "-pre", weight = c-self.threshold, flipped=False)    
            
        
        # add observation edges
        for i in fragment_dict:
            G.add_edge(str(i)+"-pre", str(i)+"-post", weight = self.inclusion, flipped=False)
            # add source node and sink node
            G.add_edge("s", str(i)+"-pre", weight = 0, flipped=False)
            G.add_edge(str(i)+"-post", "t", weight = 0, flipped=False)
            
        
        self.G = G
        self.fragment_dict = fragment_dict
        self.edge_list = edge_list
                
                
                

    def shortest_path(self, G, s):
        '''
        Single source shortest path from s
        # Dynamic programming cannot be used on residual graph because it is not a DAG
        # Use Bellman-Ford algorithm to handle negative edges
        '''

        if s not in G:
            print("Source not in graph")
            return
        
        inf = 10e6
        q = queue.Queue()
        q.put(s)
        d = {n: inf for n in G.nodes()} # keep track of the shortest distance from the source
        d[s] = 0
        p = {n: -1 for n in G.nodes()} # keep track of parent (path)
        while not q.empty():
            u = q.get()
            for v in list(G.adj[u]):
                if  d[v] > d[u] + G[u][v]["weight"]:
                    d[v] = d[u] + G[u][v]["weight"]
                    p[v] = u
                    q.put(v) # for the next round of updates
        
        return d, p
    
    def min_cost_flow(self, start, end):
        '''
        Edmond Karp algorithm for max flow
        self.G is modified each iteration
        '''
        
        tot_flow = 0
        tot_cost = 0
        inf = 10e6
        while tot_flow < inf:
            
            # FIND THE SHORTEST PATH FROM S TO T
            d, p = self.shortest_path(self.G, "s")
            if d["t"] >= 0:
                break
            
            # print()
            # FLIP THAT PATH 
            n = "t"
            path = deque()
            path_cost = 0
            while n != "s":
                cost = self.G[p[n]][n]["weight"]
                flipped = self.G[p[n]][n]["flipped"]
                path_cost += cost
                self.G.remove_edge(p[n], n)
                self.G.add_edge(n,p[n], weight = -cost, flipped = not flipped)
                # print("flipped {}-{}".format(p[n],n))
                path.appendleft(n)
                n = p[n]
                
            tot_flow += 1
            tot_cost += path_cost # total cost so far
            
            # print("path: ", path)
            # print("path cost: ",path_cost)
            # plt.figure()
            # self.draw_graph(self.G, collapse = False)
            # plt.title(str(path))
            # print("current shortest path: ", path)
           
        print("total flow: {}".format(tot_flow))
        print("total cost: {}".format(tot_cost))
        # self.find_all_post_paths(self.G, "t", "s")
        # print("all paths: ", self.all_paths)
        
        # plt.figure()
        # self.draw_graph(self.G, collapse = False)
        # plt.title("final graph")
        return tot_flow, tot_cost
    
    
    def dfs_pre_order_flipped(self, G, node, destination):
        # DFS traversal
        # save paths in post-order
        # only traverse on edges with "flipped" set to True
        self.visited.add(node)
        self.pre_order.append(node) 
        if node == destination:
            self.all_paths.append(list(self.pre_order))
        else:
            for n in G.adj[node]:
                if n not in self.visited and G[node][n]["flipped"]:
                    self.dfs_pre_order_flipped(G, n, destination)
            
        self.pre_order.pop()
        try:
            self.visited.remove(node)
        except:
            pass
        
    def dfs_pre_order(self, G, node, destination):
        # DFS traversal
        # save paths in post-order
        self.visited.add(node)
        self.pre_order.append(node) 
        if node == destination:
            self.all_paths.append(list(self.pre_order))
        else:
            for n in G.adj[node]:
                if n not in self.visited:
                    self.dfs_pre_order(G, n, destination)
            
        self.pre_order.pop()
        try:
            self.visited.remove(node)
        except:
            pass

    def dfs_post_order(self, G, node, destination):
        self.visited.add(node)
        self.post_order.appendleft(node)
        
        if node == destination:
            self.all_paths.append(list(self.post_order))
        else:
            for i in G.adj[node]:
                if i not in self.visited :
                    self.dfs_post_order(G, i, destination)
        
        self.post_order.popleft()
        self.visited.remove(node)
        
    def dfs_post_order_flipped(self, G, node, destination):
        
        self.visited.add(node)
        self.post_order.appendleft(node)
        
        if node == destination:
            self.all_paths.append(list(self.post_order))
        else:
            for i in G.adj[node]:
                if i not in self.visited and G[node][i]["flipped"]:
                    self.dfs_post_order_flipped(G, i, destination)
        
        self.post_order.popleft()
        self.visited.remove(node)
   

    def find_all_pre_paths(self, G, node, destination):
        self.visited = set()
        self.all_paths = []
        self.pre_order = []
        self.post_order = []
        self.dfs_pre_order_flipped(G, node, destination)
        # print(self.all_paths)
        
    def find_all_post_paths(self, G, node, destination):
        self.visited = set()
        self.all_paths = []
        self.pre_order = []
        self.post_order = deque()
        self.dfs_post_order_flipped(G, node, destination)
        # print(self.all_paths)
      
        
    def print_st_paths(self):
        '''
        print self.all_paths but make it prettier
        '''
        for path in self.all_paths:
            p = []
            i = 0
            # for i, node in enumerate(path[:-1]):
            while i < len(path)-1:
                if path[i][-3:] == "pre" and path[i+1][-3:] == "ost":
                    p.append(path[i][:-4])
                    i += 2
                elif path[i] == "s":
                    i += 1
                else:
                    p.append(path[i])
                    i += 1
            print(p)
                
    def draw_graph(self, G, collapse = True):
        '''
        Collapse pre and post nodes into a single node when drawing
        '''
        # k controls the distance between the nodes and varies between 0 and 1
        # iterations is the number of times simulated annealing is run
        # default = 50
        if not collapse:
            if not self.pos:
                self.pos=nx.spring_layout(G, k=0.2, iterations=100)
            for u,v in G.edges():
                if G[u][v]["flipped"]: G[u][v]['color']='r'
                else: G[u][v]['color']='k'
            colors = nx.get_edge_attributes(G,'color').values()
            nx.draw(G, self.pos, edge_color = colors, node_color= "w", with_labels = True, node_size = 500)
            labels = nx.get_edge_attributes(G,'weight')
            labels = {key: labels[key] for key in labels if abs(labels[key]) > 0.2 }
            nx.draw_networkx_edge_labels(G,self.pos,edge_labels=labels)
        else:
            n = int((len(G.nodes())-2)/2) # count pre-post as one node
            for i in range(1, n+1):
                G = nx.contracted_nodes(G, str(i)+"-pre", str(i)+"-post", self_loops=False, copy=True)
            if not self.collapse_pos:
                self.collapse_pos=nx.spring_layout(G, k=0.2, iterations=100)

            for u,v in G.edges():
                if G[u][v]["flipped"]: G[u][v]['color']='r'
                else: G[u][v]['color']='k'
            colors = nx.get_edge_attributes(G,'color').values()
            nx.draw(G, self.collapse_pos, edge_color = colors, node_color= "y", with_labels = True, node_size = 500)
            labels = nx.get_edge_attributes(G,'weight')
            labels = {key: labels[key] for key in labels if abs(labels[key]) > 0.2  }
            nx.draw_networkx_edge_labels(G,self.collapse_pos,edge_labels=labels)
        
        
    
if __name__ == '__main__':
    
    # get parameters
    cwd = os.getcwd()
    cfg = "./config"
    config_path = os.path.join(cwd,cfg)
    os.environ["user_config_directory"] = config_path
    parameters = parse_cfg("TEST", cfg_name = "test_param.config")
    
    fragment_queue = read_to_queue(parameters)
    m = MOT_Graph(inclusion = -0.1, threshold = 10, parameters = parameters)
    m.construct_graph_from_fragments(fragment_queue)
    
    
    # m.draw_graph(m.G, collapse = True)
    flow, cost = m.min_cost_flow("s", "t")
    m.find_all_post_paths(m.G, "t", "s")
    # plt.figure()
    # m.draw_graph(m.G, collapse = True)
    # plt.title("final graph")
    # print(m.all_paths)
    m.print_st_paths()
    
    