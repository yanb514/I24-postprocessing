import heapq
import numpy as np
import networkx as nx
import queue
from collections import deque
from utils.utils_stitcher_cost import min_nll_cost, nll, nll_modified, cost_1
   

    
class Fragment():
    # Constructor to create a new fragment
    def __init__(self, traj_doc):
        '''
        just a simple object that keeps all the important information from a trajectory document for tracking
        - id, timestamp, x_position, y_position, length, width, last_timsetamp, first_timestamp, direction
        '''

        # delete the unnucessary fields in traj_doc
        field_names = ["_id", "ID","timestamp","x_position","y_position","direction","last_timestamp", "first_timestamp", "length","width","height"]
        attr_names = ["id","ID","t","x","y","dir","last_timestamp","first_timestamp","length","width","height"]
        
        for i in range(len(field_names)): # set as many attributes as possible
            try: 
                if field_names[i] in {"_id", "ID"}: # cast bson ObjectId type to str
                    setattr(self, attr_names[i], str(traj_doc[field_names[i]]))
                else:
                    setattr(self, attr_names[i], traj_doc[field_names[i]])
            except: pass
        

        try: # cast to np array
            setattr(self, "x", np.array(self.x))#*0.3048)
            setattr(self, "y", np.array(self.y))#*0.3048)
            setattr(self, "t", np.array(self.t))

        except:
            pass
            
        
        
        
    def __repr__(self):
        try:
            return 'Fragment({!r})'.format(self.ID)
        except:
            return 'Fragment({!r})'.format(self.id)

        



class MOTGraphSingle:
    '''
    same as MOT_Graph except that every fragment is represented as a single node. this is equivalent to say that the inclusion cost for each fragment is 0, or the false positive rate is 0
    a fragment must be included in the trajectories
    '''
    def __init__(self, attr = "ID", parameters = None):
        self.parameters = parameters
        self.G = nx.DiGraph()
        self.G.add_nodes_from(["s","t"])
        self.G.nodes["s"]["subpath"] = []
        self.G.nodes["t"]["subpath"] = []
        self.all_paths = []
        self.attr = attr
        # self.fragment_dict = {}
        self.in_graph_deque = deque() # keep track of fragments that are currently in graph, ordered by last_timestamp
                        
                                
    def add_node(self, fragment):
        '''
        add one node i in G
        add edge t->i, mark the edge as match = True
        update distance from t
        add all incident edges from i to other possible nodes, mark edges as match = False
        '''
        
        TIME_WIN = self.parameters.time_win
        VARX = self.parameters.varx
        VARY = self.parameters.vary
        
        new_id = getattr(fragment, self.attr)
        self.G.add_edge("t", new_id, weight=0, match=True)
        self.G.nodes[new_id]["subpath"] = [new_id]
        self.G.nodes[new_id]["last_timestamp"] = fragment.last_timestamp

        for fgmt in reversed(self.in_graph_deque):
            # cost = min_nll_cost(fgmt, fragment, TIME_WIN, VARX, VARY)
            # cost = nll(fgmt, fragment, TIME_WIN, VARX, VARY)
            cost = cost_1(fgmt, fragment, TIME_WIN, VARX, VARY)
            # print(getattr(fgmt, self.attr), getattr(fragment, self.attr), cost)
            
            if cost <= 0:  # new edge points from new_id to existing nodes, with postive cost
                fgmt_id = getattr(fgmt, self.attr)
                self.G.add_edge(new_id, fgmt_id, weight = -cost, match = False)
        
        # add Fragment pointer to the dictionary
        self.in_graph_deque.append(fragment)
        # self.fragment_dict[new_id] = fragment

        # check for time-out fragments in deque and compress paths
        while self.in_graph_deque[0].last_timestamp < fragment.first_timestamp - TIME_WIN:
            fgmt = self.in_graph_deque.popleft()
            fgmt_id = getattr(fgmt, self.attr)
            for v,_,data in self.G.in_edges(fgmt_id, data = True):
                if data["match"] and v != "t":
                    # compress fgmt and v -> roll up subpath 
                    # TODO: need to check the order
                    self.G.nodes[v]["subpath"].extend(self.G.nodes[fgmt_id]["subpath"])
                    self.G.remove_node(fgmt_id)
                    break
        
        
    def clean_graph(self, path):
        '''
        remove all nodes in path from G and in_graph_deque
        '''
        for node in path:
            try:
                self.G.remove_node(node)
            except:
                pass
        
    
            
    def find_legal_neighbors(self, node):
        '''
        find ``neighbors`` of node x in G such that 
        cost(x, u) - cost(u,v) > 0, and (x,u) is unmatched, and (u,v) is matched i.e., positive delta if x steals u from v
        the idea is similar to alternating path in Hungarian algorithm
        '''
        nei = []
        for u in self.G.adj[node]:
            if not self.G[node][u]["match"]:
                cost_p = self.G[node][u]["weight"]
                for v,_ ,data in self.G.in_edges(u, data=True):
                    if data["match"]:
                        cost_m = self.G[v][u]["weight"]
                        if cost_p - cost_m > 0:
                            nei.append([u,v,cost_p - cost_m])
                        
        # print("legal nei for {} is {}".format(node, nei))
        return nei

            
            
        
    def find_alternating_path(self, root):
        '''
        construct an alternative matching tree (Hungarian tree) from root, alternate between unmatched edge and matched edge
        terminate until a node cannot change the longest distance of its outgoing neighbors
        TODO: simultaneously build tree and keep track of the longest distance path from root to leaf, output that path
        '''
        q = queue.Queue()
        q.put((root, [root], 0)) # keep track of current node, path from root, cost delta of the path
        best_dist = -1
        explored = set()
        best_path = None
        
        while not q.empty():
            x, path_to_x, dist_x = q.get()
            explored.add(x)
            nei = self.find_legal_neighbors(x)
            if not nei:
                if dist_x > best_dist:
                    best_dist = dist_x
                    best_path = path_to_x
            for u, v, delta in nei:
                if v == "t":
                    if dist_x + delta > best_dist:
                        best_dist = dist_x + delta
                        best_path = path_to_x + [u, v]
                if u not in explored:
                    q.put((v, path_to_x + [u, v], dist_x + delta))
                   
        # print("alt path for {} is {}".format(root, best_path))           
        return best_path, best_dist
    
    
    def augment_path(self, node):
        '''
        calculate an alternating path by adding node to G (assume node is already properly added to G)
        reverse that path in G (switch match bool)
        '''
        
        alt_path, _ = self.find_alternating_path(node)
        # print("alt path for {} is {}".format(node, alt_path))
        forward = True
        for i in range(len(alt_path)-1):
            if forward:
                self.G[alt_path[i]][alt_path[i+1]]["match"] = True
            else:
                self.G[alt_path[i+1]][alt_path[i]]["match"] = False
            forward = not forward

        
    def get_next_match(self, node):
        for _, i, data in self.G.out_edges(node, data=True):
            if data["match"]:
                return i
        return None   
    
    
    def get_all_traj(self):
        '''
        traverse G along matched edges
        '''
        self.all_paths = []
        
        def dfs(node, path):
            if not node: # at the leaf
                self.all_paths.append(list(path))
                
                return list(path)
            path = path + self.G.nodes[node]["subpath"]
            next = self.get_next_match(node)
            return dfs(next, path)
            
        tails =  self.G.adj["t"]
        for tail in tails:
            if self.G["t"][tail]["match"]:
                one_path = dfs(tail, [])
                # self.clean_graph([i for sublist in self.all_paths for i in sublist])
                
        return self.all_paths
            
        
    
    def pop_path(self, time_thresh):
        '''
        examine tail and pop if timeout (last_timestamp < time_thresh)
        remove the paths from G
        return paths
        '''
        all_paths = []
        
        def dfs(node, path):
            if not node: # at the leaf
                all_paths.append(list(path))
                
                return list(path)
            path = path + self.G.nodes[node]["subpath"]
            next = self.get_next_match(node)
            return dfs(next, path)
            
        tails =  self.G.adj["t"]
        for tail in tails:
            if tail in self.G.nodes and self.G["t"][tail]["match"] and self.G.nodes[tail]["last_timestamp"] < time_thresh:
                one_path = dfs(tail, [])
                # print("*** tail: ", tail, one_path)
                # self.clean_graph(one_path)
                
        return all_paths
        
        
if __name__ == '__main__':
    import os
    from i24_configparse import parse_cfg
    
    # get parameters
    cwd = os.getcwd()
    cfg = "../config"
    config_path = os.path.join(cwd,cfg)
    os.environ["user_config_directory"] = config_path
    os.environ["my_config_section"] = "DEBUG"
    parameters = parse_cfg("my_config_section", cfg_name = "test_param.config")
    
    
    m = MOTGraphSingle()
    
    # ex1
    # m.G.add_edge("g", "a", weight=2, match = True)
    # m.G.add_edge("g", "b", weight=1, match = False)
    # m.G.add_edge("g", "d", weight=2, match = True)
    # m.G.add_edge("e", "d", weight=2.5, match = False)
    # m.G.add_edge("e", "b", weight=3, match = True)
    # m.G.add_edge("i", "b", weight=5, match = False)
    # m.G.add_edge("t", "i", weight=0, match = True)
    # m.G.add_edge("t", "g", weight=0, match = True)
    # m.G.add_edge("t", "e", weight=0, match = True)
    
    
    # ex2
    # m.G.add_edge("t", "a", weight=0, match = True)
    # m.G.add_edge("t", "b", weight=0, match = True)
    # m.G.add_edge("t", "c", weight=0, match = True)
    # m.G.add_edge("c", "a", weight=2, match = False)
    # m.G.add_edge("c", "b", weight=3.5, match = False)
    # m.augment_path("c")
    # print("c: ", m.get_all_traj())
    
    # m.G.add_edge("d", "b", weight=3, match = False)
    # m.G.add_edge("d", "a", weight=0.2, match = False)
    # m.G.add_edge("t", "d", weight=0, match = True)
    # m.augment_path("d")
    # print("d: ", m.get_all_traj())
    
    # m.G.add_edge("e", "a", weight=5, match = False)
    # m.G.add_edge("t", "e", weight=0, match = True)
    # m.augment_path("e")
    # print("e: ", m.get_all_traj())
    
    
    # ex3
    m.G.add_edge("t", "a", weight=0, match = True)
    m.G.add_edge("t", "b", weight=0, match = True)
    m.G.add_edge("t", "c", weight=0, match = True)
    m.G.add_edge("c", "a", weight=6, match = False)
    m.G.add_edge("c", "b", weight=1, match = False)
    m.augment_path("c")
    print("c: ", m.get_all_traj())
    
    m.G.add_edge("d", "b", weight=3, match = False)
    m.G.add_edge("d", "a", weight=7, match = False)
    m.G.add_edge("t", "d", weight=0, match = True)
    m.augment_path("d")
    print("d: ", m.get_all_traj())
    
    # alt_path, delta = m.find_alternating_path("c")
    # print(alt_path, delta)

    
    
    
    
    
    
    
    
    
    
    