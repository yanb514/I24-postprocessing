import numpy as np

import torch
from collections import defaultdict, deque, OrderedDict
import heapq
from data_structures import DoublyLinkedList, UndirectedGraph,Fragment
import time
import sys


    
def dummy_stitcher(direction, fragment_queue, stitched_trajectory_queue, parameters):
    '''
    directly put fragments into stitched_queue
    '''
 
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("stitcher_"+direction)
    stitcher_logger.info("** min_cost_flow_online_alt_path starts", extra = None)
    setattr(stitcher_logger, "_default_logger_extra",  {})

    sig_hdlr = SignalHandler()

    GET_TIMEOUT = parameters["stitcher_timeout"]
    while sig_hdlr.run:
        try:
            try:
                fgmt = fragment_queue.get(timeout = GET_TIMEOUT)
                # stitcher_logger.debug("get fragment id: {}".format(raw_fgmt["_id"]))
                # fgmt = Fragment(raw_fgmt)
                stitched_trajectory_queue.put([fgmt])
                
            except queue.Empty: # queue is empty
                stitcher_logger.info("Stitcher timed out after {} sec.".format(GET_TIMEOUT))
 

        except Exception as e: 
            if sig_hdlr.run:
                raise e
                # stitcher_logger.error("Unexpected exception: {}".format(e))
            else:
                stitcher_logger.warning("SIGINT detected. Exception:{}".format(e))
            break
            
        
    stitcher_logger.info("Exit stitcher")
        
    return   
 

class MOT_Graph:
    
    def __init__(self, attr = "ID", parameters = None):
        self.parameters = parameters
        self.G = nx.DiGraph()
        self.G.add_nodes_from(["s","t"])
        self.all_paths = []
        self.attr = attr
        
        
       
    def collapse_paths(self):
        '''
        Collapse all paths from s to t into single nodes, with the tail ID (the nodes that connect to t)
        '''
        # find all paths from "s" to "t"
        self.find_all_post_paths( "t", "s")
        for path in all_paths:
            if len(path) > 3:
                head = path[1]
                tail = path[-2]
        return



    def add_node(self, fragment, fragment_dict):
        '''
        fragment: a document
        add fragment to existing graph (self.G) with possible connections
        look for id-fragment pair in fragment_dict
        note that nodes in self.G are a subset of fragment_dict, but the node ids needs preprocessing
        '''
        TIME_WIN = self.parameters.time_win
        VARX = self.parameters.varx
        VARY = self.parameters.vary
        # THRESHOLD = self.parameters.thresh
        INCLUSION = self.parameters.inclusion
        
        edge_list = []

        for fgmt_id, fgmt in fragment_dict.items():
            # cost = min_nll_cost(fgmt, fragment, TIME_WIN, VARX, VARY)
            cost = nll(fgmt, fragment, TIME_WIN, VARX, VARY)
            # print(fgmt.ID, fragment.ID, cost)
            
            if cost <= 0: # Natually, if cost > 0, it's preferable to break into different trajectories
            # if cost < THRESHOLD: 
                edge_list.append(((fgmt_id, getattr(fragment, self.attr)),cost))
            
        # add transition edges
        for e,c in edge_list:
            if e[0] != e[1]:
                self.G.add_edge(str(e[0]) + "-post", str(e[1]) + "-pre", weight = c, flipped=False)    
        
        # add observation edges
        ID = getattr(fragment, self.attr)
        self.G.add_edge(str(ID)+"-pre", str(ID)+"-post", weight = INCLUSION, flipped=False)
        
        # add source node and sink node
        self.G.add_edge("s", str(ID)+"-pre", weight = 0, flipped=False)
        self.G.add_edge(str(ID)+"-post", "t", weight = 0, flipped=False)
        
        
        return 
                
                

    def construct_graph_from_fragments(self, fragment_queue):
        '''
        batch construct a graph from fragment_queue
        '''
        TIME_WIN = self.parameters.time_win
        VARX = self.parameters.varx
        VARY = self.parameters.vary
        THRESHOLD = self.parameters.thresh
        INCLUSION = self.parameters.inclusion
    
        
        G = nx.DiGraph()
        fragment_dict = {}
        edge_list = []
        
        while not fragment_queue.empty():
            new_fgmt = Fragment(fragment_queue.get())
            new_fgmt.compute_stats()

            for fgmt_id, fgmt in fragment_dict.items():
                # if fgmt_id == "10800011.0" and new_fgmt.ID == "10800012.0":
                #     print('here')
                cost = nll(fgmt, new_fgmt, TIME_WIN, VARX, VARY)
                # print("construct_graph ", ((getattr(fgmt, self.attr), getattr(new_fgmt, self.attr)),cost))
                if cost < 0:
                    edge_list.append(((getattr(fgmt, self.attr), getattr(new_fgmt, self.attr)),cost))
                    
            fragment_dict[getattr(new_fgmt, self.attr)] = new_fgmt 
            
        # add transition edges
        for e,c in edge_list:
            if e[0] != e[1]:
                G.add_edge(str(e[0]) + "-post", str(e[1]) + "-pre", weight = c, flipped=False)    
            
        
        # add observation edges
        for i in fragment_dict:
            G.add_edge(str(i)+"-pre", str(i)+"-post", weight = INCLUSION, flipped=False)
            # add source node and sink node
            G.add_edge("s", str(i)+"-pre", weight = 0, flipped=False)
            G.add_edge(str(i)+"-post", "t", weight = 0, flipped=False)
            
        
        self.G = G
        self.fragment_dict = fragment_dict
        self.edge_list = edge_list
                
    
    def shortest_path(self, s):
        '''
        Single source shortest path from s
        # Dynamic programming cannot be used on residual graph because it is not a DAG
        # Use Bellman-Ford algorithm to handle negative edges
        '''

        if s not in self.G:
            print("Source not in graph")
            return
        
        inf = 10e6
        q = queue.Queue()
        q.put(s)
        d = {n: inf for n in self.G.nodes()} # keep track of the shortest distance from the source
        d[s] = 0
        p = {n: -1 for n in self.G.nodes()} # keep track of parent (path)
        while not q.empty():
            u = q.get()
            for v in list(self.G.adj[u]):
                if  d[v] > d[u] + self.G[u][v]["weight"]:
                    d[v] = d[u] + self.G[u][v]["weight"]
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
            d, p = self.shortest_path("s")
            if d["t"] > 0:
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
            
        return tot_flow, tot_cost
    
    
    def flip_edge_along_path(self, path, flipped = None):
        '''
        flip all the edges along a given path, where edge cost is negated, and flipped is set to the given value
        if flipped = None, simply negate the exising flipped flag
        '''
        for i in range(len(path)-1):
            u,v = path[i], path[i+1]
            cost = self.G[u][v]["weight"]
            bool = self.G[u][v]["flipped"]
            self.G.remove_edge(u,v)
            self.G.add_edge(v,u , weight = -cost, flipped = not bool if not flipped else flipped)
        return
            
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
        
    def dfs_post_order_flipped(self, G, node, destination, cost):
        self.visited.add(node)
        self.post_order.appendleft(node)
        
        if node == destination:
            self.all_paths.append((list(self.post_order), cost))
        else:
            for i in G.adj[node]:
                if i not in self.visited and G[node][i]["flipped"]:
                    cost += G[node][i]["weight"]
                    self.dfs_post_order_flipped(G, i, destination, cost)
        
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
        self.dfs_post_order_flipped(G, node, destination, 0)
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
            
    def pretty_path(self, path):
        '''
        get rid of "-pre, -post" in paths and remove duplicate ids
        '''
        p = []
        i = 0
        prev_id = ""
        # for i, node in enumerate(path[:-1]):
        while i < len(path)-1:
            if path[i] not in {"s", "t"}:
                id = path[i].partition("-")[0]
                if id != prev_id:
                    p.append(id)
                    prev_id = id
            i += 1
        return p
                
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
            
            



class FragmentOld(Node):
    # Constructor to create a new fragment
    def __init__(self, traj_doc):
        '''
        traj_doc is a record from raw_trajectories database collection
        '''
        
        self.suc = [] # tail matches to [(cost, Fragment_obj)] - minheap
        self.pre = [] # head matches to [(cost, Fragment_obj)] - minheap
        # self.conflicts_with = set() # keep track of conflicts - bi-directional
        # self.ready = False # if tail is ready to be matched
    
        self.child = None # its immediate successor
        self.root = self # the first fragment in its path
        self.parent = None # its immediate predecessor
        self.tail_matched = False # flip to true when its tail matches to another fragment's head
        self.head_matched = False
        
        # only one of the following is true when first entering the process
        self.curr = False # in current view of time window?
        self.past = False # in left time window and tail is ready to be matched?
        self.gone = False # left past view because already matched or no match exists, ready to be popped out
         
        # time_series = ["timestamp", "x_position", "y_position"]
        if traj_doc: 
            # delete the unnucessary fields in traj_doc
            field_names = ["_id", "ID","timestamp","x_position","y_position","direction","last_timestamp","last_timestamp", "first_timestamp"]
            attr_names = ["id","ID","t","x","y","dir","tail_time","last_timestamp","first_timestamp"]
            
            try:
                unwanted = set(traj_doc.keys()) - set(field_names)
                for unwanted_key in unwanted: 
                    try: del traj_doc[unwanted_key]
                    except: pass
            except:
                pass
            
            for i in range(len(field_names)): # set as many attributes as possible
                try: 
                    if field_names[i] in {"_id", "ID"}: # cast bson ObjectId type to str
                        setattr(self, attr_names[i], str(traj_doc[field_names[i]]))
                    else:
                        setattr(self, attr_names[i], traj_doc[field_names[i]])
                except: pass
            super().__init__(None)
            

            try:
                setattr(self, "x", np.array(self.x))#*0.3048)
                setattr(self, "y", np.array(self.y))#*0.3048)
                setattr(self, "t", np.array(self.t))
            except:
                pass
            
        else:
            super().__init__(traj_doc)
        
        
        
    def __repr__(self):
        try:
            return 'Fragment({!r})'.format(self.ID)
        except:
            return 'Fragment({!r})'.format(self.id)

        

    
    def compute_stats(self):
        '''
        compute statistics for matching cost
        based on linear vehicle motion (least squares fit constant velocity)
        WARNING: may not be precise with floating timestamps. use scipy.stats.linregress() instead
        '''
        t,x,y = self.t, self.x, self.y
        ct = np.nanmean(t)
        if len(t)<2:
            v = 30 * self.dir # assume 30m/s
            b = x-v*ct # recalculate y-intercept
            fitx = np.array([v,b[0]])
            fity = np.array([0,y[0]])
        else:
            xx = np.vstack([t,np.ones(len(t))]).T # N x 2
            fitx = np.linalg.lstsq(xx,x, rcond=None)[0]
            fity = np.linalg.lstsq(xx,y, rcond=None)[0]
        self.fitx = fitx
        self.fity = fity
        return
    

    # add successor to fragment with matching cost
    def add_suc(self, cost, fragment):
        heapq.heappush(self.suc, (cost, fragment))
    
    # add predecessor
    def add_pre(self, cost, fragment):
        heapq.heappush(self.pre, (cost, fragment))
       
    def peek_first_suc(self):
        # return the best successor of self if exists
        # otherwise return None
        # heappop empty fragments from self.suc
        while self.suc and self.suc[0][1].head_matched: # get the first "unmatched" suc
            _, suc = heapq.heappop(self.suc)
        try: suc = self.suc[0][1] # self.suc[0] is None
        except: suc = None
        return suc
    
    def peek_first_pre(self):
        # return the best predecessor of self if exists
        # otherwise return None
        # heappop empty fragments from self.suc
        # while self.pre and self.pre[0][1].id is None: # get the first non-empty fragment
        while self.pre and self.pre[0][1].tail_matched: # get the first "unmatched" pre
            _, pre = heapq.heappop(self.pre)
        try: pre = self.pre[0][1]
        except: pre = None
        return pre
    
    # union conflicted neighbors, set head's pre to [], delete self
    @classmethod
    def match_tail_head(cls, u,v):
        v.pre = [] # v's head cannot be matched again
        u.suc = [] # u's tail cannot be matched again
        u.tail_matched = True
        v.head_matched = True
        return

        
# A class to represent a disjoint set
class PathCache(SortedDLL):
    '''
    This class combines a union-find data structure and an LRU data structure
    Purpose:
        - keep track of root (as the first-appeared fragments) and their successors in paths
        - output paths to stitched_trajectories database if time out

    '''
    def __init__(self, attr_name = "id"):
        super().__init__(attr_name)
        self.path = {} # keep pointers for all Fragments
        self.attr = attr_name
        
    def make_set(self, docs):
        for doc in docs:
            self.add_node(doc)
            
    def add_node(self, node):
        if not isinstance(node, Fragment):
            node = Fragment(node) # create a new node
        # self.cache[node.id] = node
        self.append(node)
        self.path[str(getattr(node, self.attr))] = node

    def get_fragment(self, id):
        return self.path[id]

    
    # Find the root of the set in which element `k` belongs
    def find(self, node):
    
        if not node.parent: # if node is the root
            return node
        # delete node from cache, because node is not the root
        try:
            self.delete(getattr(node, self.attr))
        except:
            pass
        # path compression
        node.root = node.parent.root 
        return self.find(node.parent)

        
    # Perform Union of two subsets
    def union(self, id1, id2):
        # id2 comes after id1
        if id1 == id2:
            return
        # find the root of the sets in which Nodes `id1` and `id2` belong
        node1, node2 = self.path[id1], self.path[id2]
        root1 = self.find(node1)
        root2 = self.find(node2)
        
        # if `id1` and `id2` are present in the same set, only move to the end of cache
        if root1 == root2:
            self.update(getattr(root1, self.attr), max(root1.tail_time, node2.tail_time), attr_name = "tail_time")
            return
        
        # compress path: update parent and child pointers for node1 and node2
        # they should be on the same path from the shared root
        # by matching logic, node1 has no child
        # Follow "merging two sorted linked lists in-place" from leetcode
        p1,p2 = node1, node2
        if node2.child:
            node2_is_leaf = False
            head = node2.child
        else:
            node2_is_leaf = True
            head = Fragment(None) # create dummy 
            setattr(head, self.attr, -1)

        while p1 and p2:
            if p1.last_timestamp < p2.last_timestamp:
                head.parent = p2
                p2.child = head
                p2 = p2.parent
            else:
                head.parent = p1
                p1.child = head
                p1 = p1.parent
            head = head.parent
         
        
        head.parent = p1 or p2
        head.parent.child = head
        
        if node2_is_leaf:
            node2.child = None
                
        # update roots along the path
        new_root1 = self.find(node1)
        node1.root = new_root1
        new_root2 = self.find(node2)
        node2.root = new_root2

        # update tail_time for all nodes along the path
        # self.path_down_update(new_root1) # TODO probably unnecessary
        self.update(getattr(new_root1, self.attr), max(new_root1.tail_time, node2.tail_time), attr_name = "tail_time")

        
    def print_sets(self):
        
        print([getattr(self.find(val), self.attr) for key,val in self.path.items()])
        return
          

    def get_all_paths(self):
        '''
        get all the paths from roots, whatever remains in self.cache.keys are roots!
        DFS
        '''
        all_paths = [] # nested lists
         
        # for DLL cache
        node = self.first_node()
        while node != self.sentinel:
            path = self.path_down(node, self.attr)
            all_paths.append(path)
            node = node.next
        return all_paths
    
                
    def get_all_roots(self, attr_name=None):
        head = self.sentinel.next
        roots = []
        while head != self.sentinel:
            roots.append(head)
            head = head.next
        if attr_name:
            roots = [getattr(head, attr_name) for head in roots]
        return roots
    
    def print_cache(self):
        roots = self.get_all_roots()
        for root in roots:
            print("Root {}: tail_time is {}".format(getattr(root, self.attr), root.tail_time))
        
    def print_attr(self, attr_name):
        for node in self.path.values():
            try:
                print("Node {}: {}: {}".format(getattr(root, self.attr), attr_name, getattr(node, attr_name)))
            except:
                print("Node {} has no {}".format(getattr(root, self.attr), attr_name))
                
    def pop_first_path(self):
        '''
        pop the first node (root) from cache if cache is not empty
        delete all nodes along the path in self.path
        return the path: a list of ids
        '''
        # DLL cache
        first_node = self.first_node()
        if first_node: # if cache is not empty
            path = self.path_down(first_node)
            for p in path:
                try:
                    fragment = self.path.pop(p)
                    fragment.gone = True
                except:
                    pass
            self.delete(first_node)
            return path
                
    def path_up(self, node):
        path = []
        def _dfs(node, path):
            if node:
                path.append(getattr(node, self.attr)) 
                _dfs(node.parent, path)
        _dfs(node, path)
        return path
    
    def path_down(self, node):
        path = []
        def _dfs(node, path):
            if node:
                path.append(getattr(node, self.attr)) 
                _dfs(node.child, path)
        _dfs(node, path)
        return path  
     
    def path_down_update(self, node, attr_name = "tail_time"):
        def _dfs(node):
            if node: # if node is not leaf
                _dfs(node.child)
                try: # if node.child exists
                    setattr(node, attr_name, max(getattr(node.child, attr_name), getattr(node, attr_name))) 
                except:
                    pass
        _dfs(node)
        return        
        
 
    
 
def min_cost_flow_batch(direction, fragment_queue, stitched_trajectory_queue, parameters):
    '''
    solve min cost flow problem on a given graph using successive shortest path 
    - derived from Edmonds-Karp for max flow
    '''
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("stitcher_"+direction)
    stitcher_logger.info("** min_cost_flow_batch starts. fragment_queue size: {}".format(fragment_queue.qsize()),extra = None)

    # Make a database connection for writing
    schema_path = os.path.join(os.environ["user_config_directory"],parameters.stitched_schema_path)
    dbw = DBWriter(host=parameters.default_host, port=parameters.default_port, 
                username=parameters.default_username, password=parameters.default_password,
                database_name=parameters.db_name, collection_name = parameters.stitched_collection, 
                server_id=1, process_name=1, process_id=1, session_config_id=1, max_idle_time_ms = None,
                schema_file=schema_path)
    
    # Get parameters
    ATTR_NAME = parameters.fragment_attr_name
    
    # Initialize some data structures
    m = MOT_Graph(ATTR_NAME, parameters)
    m.construct_graph_from_fragments(fragment_queue)
    # all_paths = []
    
    
            
    # count = 0
    m.min_cost_flow("s", "t")
    
    # Collapse paths
    m.find_all_post_paths(m.G, "t", "s") 
    
    for path, cost in m.all_paths:
        path = m.pretty_path(path)
        # all_paths.append(path)
        # print(path)
        # print("** write to db: root {}, path length: {}".format(path[0], len(path)))
        stitched_trajectory_queue.put(path, timeout = parameters.stitched_trajectory_queue_put_timeout)
        # dbw.write_one_trajectory(thread = True, fragment_ids = path)
        
    # return all_paths
    return
    
        



# @catch_critical(errors = (Exception))
def min_cost_flow_online_neg_cycle(direction, fragment_queue, stitched_trajectory_queue, parameters):
    '''
    run MCF on the entire (growing) graph as a new fragment is added in
    this online version is an approximation of batch MCF
    '''
    
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("stitcher_"+direction)
    stitcher_logger.info("** min_cost_flow_online_neg_cycle starts. fragment_queue size: {}".format(fragment_queue.qsize()),extra = None)

    # Make a database connection for writing
    schema_path = os.path.join(os.environ["user_config_directory"],parameters.stitched_schema_path)
    dbw = DBWriter(host=parameters.default_host, port=parameters.default_port, 
               username=parameters.default_username, password=parameters.default_password,
               database_name=parameters.db_name, collection_name = parameters.stitched_collection, 
               server_id=1, process_name=1, process_id=1, session_config_id=1, max_idle_time_ms = None,
               schema_file=schema_path)
        
    
    
    ATTR_NAME = parameters.fragment_attr_name
    m = MOT_Graph(ATTR_NAME, parameters)
    cache = {}

    IDLE_TIME = parameters.idle_time

    
    while True:
        
        # stitcher_logger.info("fragment qsize: {}".format(fragment_queue.qsize()))
        
        # t1 = time.time()
        # cum_time.append(t1-t0)
        # cum_mem.append(len(cache))
        
        try:
            fgmt = Fragment(fragment_queue.get(timeout = 2))
            # fgmt.compute_stats()
            m.add_node(fgmt, cache)
            fgmt_id = getattr(fgmt, ATTR_NAME)
            cache[fgmt_id] = fgmt
            left = fgmt.first_timestamp
            # stitcher_logger.info("fgmt_id: {}, first timestamp: {:.2f}".format(fgmt_id, left))
            
        except:
            # process the remaining in G
            m.find_all_post_paths(m.G, "t", "s")
            all_paths = m.all_paths
            for path,_ in all_paths:
                path = m.pretty_path(path)
                stitched_trajectory_queue.put(path)
                for p in path: # should not have repeats in nodes_to_remove
                    m.G.remove_node(p+"-pre")
                    m.G.remove_node(p+"-post")
                    cache.pop(p)
                stitcher_logger.info("** no new fgmt, write to queue. path length: {}, head id: {}, graph size: {}".format(len(path), path[0], len(cache)))
                # print("final flush head tail: ", path[0], path[-1])
            break
        
        # Finding all pivots (predecessor nodes of new_fgmt-pre such that the path cost pivot-new_fgmt-t is negative)
        pivot_heap = []
        for pred, data in m.G.pred[fgmt_id + "-pre"].items():
            cost_new_path = data["weight"] + parameters.inclusion
            if cost_new_path < 0: # favorable to attach fgmt after pred
                heapq.heappush(pivot_heap, (cost_new_path, pred))
           

        # check the cost of old path from t->pivot along "flipped" edges
        while pivot_heap:
            cost_new_path, pred = heapq.heappop(pivot_heap) 
            if pred == "s": # create a new trajectory
                m.flip_edge_along_path([pred, fgmt_id+"-pre", fgmt_id+"-post", "t"], flipped = True)
                break

            m.find_all_post_paths(m.G, "t", pred)
            old_path, cost_old_path = m.all_paths[0] # should have one path only
            
            # the new path that starts from pivot and includes the new fragment is better than the path starting from pivot and not including the new fgmt
            # therefore, update the new path
            
            if cost_new_path < -cost_old_path:
                # flip edges in path from pivot -> fgmt -> t
                m.flip_edge_along_path([pred, fgmt_id+"-pre", fgmt_id+"-post", "t"], flipped = True)     
                
                succ = old_path[1]
                m.flip_edge_along_path([succ, pred], flipped = False)# back to the original state
                
                # if sucessor of pivot in old path is not t, connect sucessor to s
                if succ != "t":
                    m.flip_edge_along_path(["s", succ], flipped = True)
                    
                break # no need to check further in heap
        
        
        # look at all neighbors of "t", which are the tail nodes
        # if the tail is time-out, then pop the entire path
        m.find_all_post_paths(m.G, "t", "s")
        # print(m.all_paths)
        
        nodes_to_remove = []
        for node in m.G.adj["t"]: # for each tail node
            if m.G["t"][node]["flipped"]:
                tail_id = node.partition("-")[0]
                if cache[tail_id].last_timestamp + IDLE_TIME < left:
                    m.find_all_post_paths(m.G, node, "s")
                    path, cost = m.all_paths[0]
                    path = m.pretty_path(path)
                    # print("remove: ", path)
                    stitcher_logger.info("** write to queue. path length: {}, head id: {}, graph size: {}".format(len(path), path[0], len(cache)))
                    stitched_trajectory_queue.put(path)
                    # dbw.write_one_trajectory(thread = True, fragment_ids = path)
                    # remove path from G and cache
                    nodes_to_remove.extend(path)

                        
        for p in nodes_to_remove: # should not have repeats in nodes_to_remove
            m.G.remove_node(p+"-pre")
            m.G.remove_node(p+"-post")
            cache.pop(p)
            
        # stitcher_logger.info("nodes to remove: {}, cache size: {}".format(len(nodes_to_remove), len(cache)))
                
        
    # return cum_time, cum_mem
    return







def min_cost_flow_online_slow(direction, fragment_queue, stitched_trajectory_queue, parameters):
    '''
    run MCF on the entire (growing) graph as a new fragment is added in
    print out the paths, just to see how paths change
    how does the new fgmt modify the previous MCF solution?
        1. the new fgmt creates a new trajectory (Y)
        2. the new fgmt addes to the tail of an existing trajectory (Y)
        3. the new fgmt breaks an existing trajectory (in theory should exist, but not observed)
        4 ...?
    '''
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("stitcher_"+direction)

    # Make a database connection for writing
    schema_path = os.path.join(os.environ["user_config_directory"],parameters.stitched_schema_path)
    dbw = DBWriter(host=parameters.default_host, port=parameters.default_port, 
               username=parameters.default_username, password=parameters.default_password,
               database_name=parameters.db_name, collection_name = parameters.stitched_collection, 
               server_id=1, process_name=1, process_id=1, session_config_id=1, max_idle_time_ms = None,
               schema_file=schema_path)
        
    stitcher_logger.info("** min_cost_flow_online_neg_cycle starts. fragment_queue size: {}".format(fragment_queue.qsize()),extra = None)
    
    ATTR_NAME = parameters.fragment_attr_name
    m = MOT_Graph(ATTR_NAME, parameters)
    cache = {}
    
    while True:
        try:
            fgmt = Fragment(fragment_queue.get(block=True))
        except:
            all_paths = m.all_paths
            for path,_ in all_paths:
                path = m.pretty_path(path)
                stitched_trajectory_queue.put(path)
            break
        # fgmt.compute_stats()
        m.add_node(fgmt, cache)
        fgmt_id = getattr(fgmt, ATTR_NAME)
        cache[fgmt_id] = fgmt
        print("new fgmt: ", fgmt_id)
        
        # run MCF
        m.min_cost_flow("s", "t")
        # m.find_all_post_paths(m.G, "t", "s") 
        # print(m.all_paths)
        
        # flip edges back
        edge_list = list(m.G.edges)
        for u,v in edge_list:
            if m.G.edges[u,v]["flipped"]:
                cost = m.G.edges[u,v]["weight"]
                m.G.remove_edge(u,v)
                m.G.add_edge(v,u, weight = -cost, flipped = False)
                
                
        # check if the tail of any paths is timed out. if so remove the entire path from the graph
        # TODO
    return

    

def dummy_stitcher(old_q, new_q):
    # Initiate a logger
    stitcher_logger = log_writer.logger
    stitcher_logger.set_name("dummy")
    stitcher_logger.info("** min_cost_flow_online_alt_path starts", extra = None)

    # Signal handling    
    signal.signal(signal.SIGINT, signal.SIG_IGN)    
    signal.signal(signal.SIGPIPE,signal.SIG_DFL) # reset SIGPIPE so that no BrokePipeError when SIGINT is received
    
    while True:
        try:
            x = old_q.get(timeout = 3)
        except:
            stitcher_logger.info("old_q is empty, exit")
            sys.exit(2)
        
        time.sleep(0.1)
        
        new_q.put([x["_id"]])
        stitcher_logger.info("old_q size: {}, new_q size:{}".format(old_q.qsize(),new_q.qsize()))
        
        
    stitcher_logger.info("Exiting dummy stitcher while loop")
    sys.exit(2)
        
        
        
        
        
loss = torch.nn.GaussianNLLLoss()   
   
def _compute_stats(track):
    t,x,y = track['t'],track['x'],track['y']
    ct = np.nanmean(t)
    if len(t)<2:
        v = np.sign(x[-1]-x[0]) # assume 1/-1 m/frame = 30m/s
        b = x-v*ct # recalculate y-intercept
        fitx = np.array([v,b[0]])
        fity = np.array([0,y[0]])
    else:
        xx = np.vstack([t,np.ones(len(t))]).T # N x 2
        fitx = np.linalg.lstsq(xx,x, rcond=None)[0]
        fity = np.linalg.lstsq(xx,y, rcond=None)[0]
    track['t'] = t
    track['x'] = x
    track['y'] = y
    track['fitx'] = fitx
    track['fity'] = fity
    return track
   

            
def stitch_objects_tsmn_online_2(o, THRESHOLD_MAX=3, VARX=0.03, VARY=0.03, time_out = 500):
        '''
        Dan's online version
        
        can potentially "under" stitch if interior fragments have higher matching cost
        THRESHOLD_MAX: aobve which pairs should never be matched
        online version of stitch_objects_tsmn_ll
        track: dict with key: id, t, x, y
            {"id": 20,
             "t": [frame1, frame2,...],
             "x":[x1,x2,...], 
             "y":[y1,y2...],
             "fitx": [vx, bx], least square fit
             "fity": [vy, by]}
        tracks come incrementally as soon as they end
        '''
        # define cost
        def _getCost(track1, track2):
            '''
            track1 always ends before track2 ends
            999: mark as conflict
            -1: invalid
            '''

            if track2["t"][0] < track1['t'][-1]: # if track2 starts before track1 ends
                return 999
            if track2['t'][0] - track1['t'][-1] > time_out: # if track2 starts TIMEOUT after track1 ends
                return -1
            
            # predict from track1 forward to time of track2
            xx = np.vstack([track2['t'],np.ones(len(track2['t']))]).T # N x 2
            targetx = np.matmul(xx, track1['fitx'])
            targety = np.matmul(xx, track1['fity'])
            pt1 = track1['t'][-1]
            varx = (track2['t']-pt1) * VARX 
            vary = (track2['t']-pt1) * VARY
            input = torch.transpose(torch.tensor([track2['x'],track2['y']]),0,1)
            target = torch.transpose(torch.tensor([targetx, targety]),0,1)
            var = torch.transpose(torch.tensor([varx,vary]),0,1)
            nll1 = loss(input,target,var).item()
            
            # predict from track2 backward to time of track1 
            xx = np.vstack([track1['t'],np.ones(len(track1['t']))]).T # N x 2
            targetx = np.matmul(xx, track2['fitx'])
            targety = np.matmul(xx, track2['fity'])
            pt1 = track2['t'][-1]
            varx = (track1['t']-pt1) * VARX 
            vary = (track1['t']-pt1) * VARY
            input = torch.transpose(torch.tensor([track1['x'],track1['y']]),0,1)
            target = torch.transpose(torch.tensor([targetx, targety]),0,1)
            var = torch.transpose(torch.tensor([varx,vary]),0,1)
            nll2 = loss(input,target,np.abs(var)).item()
            return min(nll1, nll2)
            # return nll1
        
        def _first(s):
            '''Return the first element from an ordered collection
               or an arbitrary element from an unordered collection.
               Raise StopIteration if the collection is empty.
            '''
            return next(iter(s.values()))
        
        df = o.df
        # sort tracks by start/end time - not for real deployment
        
        groups = {k: v for k, v in df.groupby("ID")}
        ids = list(groups.keys())
        ordered_tracks = deque() # list of dictionaries
        all_tracks = {}
        S = []
        E = []
        for id, car in groups.items():
            t = car["Frame #"].values
            x = (car.bbr_x.values + car.bbl_x.values)/2
            y = (car.bbr_y.values + car.bbl_y.values)/2
            notnan = ~np.isnan(x)
            t,x,y = t[notnan], x[notnan],y[notnan]
            if len(t)>1: # ignore empty or only has 1 frame
                S.append([t[0], id])
                E.append([t[-1], id])
                track = {"id":id, "t": t, "x": x, "y": y} 
                # ordered_tracks.append(track)
                all_tracks[id] = track

            
        heapq.heapify(S) # min heap (frame, id)
        heapq.heapify(E)
        EE = E.copy()
        while EE:
            e, id = heapq.heappop(EE)
            ordered_tracks.append(all_tracks[id])
            
        # Initialize
        X = UndirectedGraph() # exclusion graph
        TAIL = defaultdict(list) # id: [(cost, head)]
        HEAD = defaultdict(list) # id: [(cost, tail)]
        curr_tracks = deque() # tracks in view. list of tracks. should be sorted by end_time
        path = {} # oldid: newid. to store matching assignment
        past_tracks = DoublyLinkedList() # set of ids indicate end of track ready to be matched
        TAIL_MATCHED = set()
        HEAD_MATCHED = set()
        matched = 0 # count matched pairs
        
        running_tracks = OrderedDict() # tracks that start but not end at e 
        
        start = time.time()
        for i,track in enumerate(ordered_tracks):
            # print("\n")
            # print('Adding new track {}/{}'.format(i, len(ordered_tracks)))
            # print("Out of view: {}".format(past_tracks.size))

            curr_id = track['id'] # last_track = track['id']
            path[curr_id] = curr_id
            right = track['t'][-1] # right pointer: current time
            
            # get tracks that started but not end - used to define the window left pointer
            while S and S[0][0] < right: # append all the tracks that already starts
                started_time, started_id = heapq.heappop(S)
                running_tracks[started_id] = started_time
                
            # compute track statistics
            track = _compute_stats(track)
            
            try: 
                left = max(0,_first(running_tracks) - time_out)
            except: left = 0
            # print("window size :", right-left)
            # remove out of sight tracks 
            while curr_tracks and curr_tracks[0]['t'][-1] < left:           
                past_tracks.append(curr_tracks.popleft()['id'])
            
            # compute score from every track in curr to track, update Cost
            for curr_track in curr_tracks:
                cost = _getCost(curr_track, track)
                if cost > THRESHOLD_MAX:
                    X._addEdge(curr_track['id'], track['id'])
                elif cost > 0:
                    heapq.heappush(TAIL[curr_track['id']], (cost, track['id']))
                    heapq.heappush(HEAD[track['id']], (cost, curr_track['id']))
            
            # print("TAIL {}, HEAD {}".format(len(TAIL), len(HEAD)))
            # start matching from the first ready tail
            tail_node = past_tracks.head
            if not tail_node:  # no ready tail available: keep waiting
                curr_tracks.append(track)        
                running_tracks.pop(curr_id) # remove tracks that ended
                continue # go to the next track in ordered_tracks

            while tail_node is not None:
                tail = tail_node.data # tail is ready (time-wise)
                
                # remove already matched
                while TAIL[tail] and TAIL[tail][0][1] in HEAD_MATCHED:
                    heapq.heappop(TAIL[tail]) 
                if not TAIL[tail]: # if tail does not have candidate match
                    TAIL.pop(tail)
                    tail_node = tail_node.next # go to the next ready tail
                    continue
                _, head = TAIL[tail][0] # best head for tail
                while HEAD[head] and HEAD[head][0][1] in TAIL_MATCHED:
                    heapq.heappop(HEAD[head]) 
                if not HEAD[head]:
                    HEAD.pop(head)
                    tail_node = tail_node.next
                    continue
                else: _, tail2 = HEAD[head][0]

                # tail and head agrees with each other
                if tail==tail2:
                    if head in X[tail]: # conflicts
                        HEAD.pop(head)
                        TAIL.pop(tail)
                    else: # match tail and head
                        # print("matching {} & {}".format(tail, head))
                        path[head] = path[tail]
                        X._union(head, tail)
                        HEAD.pop(head)
                        TAIL.pop(tail)
                        HEAD_MATCHED.add(head)
                        TAIL_MATCHED.add(tail)
                        matched += 1
                        past_tracks.delete_element(tail)
                        X._remove(tail)
                    
                # match or not, process the next ready tail  
                tail_node = tail_node.next
                
                    
            curr_tracks.append(track)        
            running_tracks.pop(curr_id) # remove tracks that ended
            # print("matched:", matched)
            
        # delete IDs that are empty
        # print("\n")
        # print("{} Ready: ".format(past_tracks.printList()))
        # print("{} Processsed: ".format(len(processed)))
        print("{} pairs matched".format(matched))
        # print("Deleting {} empty tracks".format(len(empty_id)))
        # df = df.groupby("ID").filter(lambda x: (x["ID"].iloc[0] not in empty_id))
        end = time.time()
        print('run time online stitching:', end-start)
        # for debugging only
        o.path = path
        # o.C = C
        o.X = X
        o.groupList = ids
        o.past_tracks = past_tracks.convert_to_set()
        o.TAIL = TAIL
        o.HEAD = HEAD
        # replace IDs
        newids = [v for _,v in path.items()]
        m = dict(zip(path.keys(), newids))
        df = df.replace({'ID': m})
        df = df.sort_values(by=['Frame #','ID']).reset_index(drop=True)         
        
        print("Before DA: {} unique IDs".format(len(ids))) 
        print("After DA: {} unique IDs".format(df.groupby("ID").ngroups))
        print("True: {} unique IDs".format(len([id for id in ids if id<1000])))
      
        o.df = df
        return o


# define cost
def _getCost(track1, track2, time_out, VARX, VARY):
    '''
    track1 always ends before track2 ends
    999: mark as conflict
    -1: invalid
    '''

    if track2.t[0] < track1.t[-1]: # if track2 starts before track1 ends
        return 999
    if track2.t[0] - track1.t[-1] > time_out: # if track2 starts TIMEOUT after track1 ends
        return -1
    
    # predict from track1 forward to time of track2
    xx = np.vstack([track2.t,np.ones(len(track2.t))]).T # N x 2
    targetx = np.matmul(xx, track1.fitx)
    targety = np.matmul(xx, track1.fity)
    pt1 = track1.t[-1]
    varx = (track2.t-pt1) * VARX 
    vary = (track2.t-pt1) * VARY
    input = torch.transpose(torch.tensor([track2.x,track2.y]),0,1)
    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
    var = torch.transpose(torch.tensor([varx,vary]),0,1)
    nll1 = loss(input,target,var).item()
    
    # predict from track2 backward to time of track1 
    xx = np.vstack([track1.t,np.ones(len(track1.t))]).T # N x 2
    targetx = np.matmul(xx, track2.fitx)
    targety = np.matmul(xx, track2.fity)
    pt1 = track2.t[-1]
    varx = (track1.t-pt1) * VARX 
    vary = (track1.t-pt1) * VARY
    input = torch.transpose(torch.tensor([track1.x,track1.y]),0,1)
    target = torch.transpose(torch.tensor([targetx, targety]),0,1)
    var = torch.transpose(torch.tensor([varx,vary]),0,1)
    nll2 = loss(input,target,np.abs(var)).item()
    return min(nll1, nll2)
    # return nll1
    
def _first(s):
        '''Return the first element from an ordered collection
           or an arbitrary element from an unordered collection.
           Raise StopIteration if the collection is empty.
        '''
        return next(iter(s.values()))
        
def stitch_objects_tsmn_online_3(o, THRESHOLD_MAX=3, VARX=0.03, VARY=0.03, time_out = 500):
        '''
        build on ver2 - with object-oriented data structure
        '''
        
        df = o.df
        # sort tracks by start/end time - not for real deployment
        
        groups = {k: v for k, v in df.groupby("ID")}
        ids = list(groups.keys())
        ordered_tracks = deque() # list of dictionaries
        all_tracks = {}
        S = []
        E = []
        for id, car in groups.items():
            t = car["Frame #"].values
            x = (car.bbr_x.values + car.bbl_x.values)/2
            y = (car.bbr_y.values + car.bbl_y.values)/2
            notnan = ~np.isnan(x)
            t,x,y = t[notnan], x[notnan],y[notnan]
            if len(t)>1: # ignore empty or only has 1 frame
                S.append([t[0], id])
                E.append([t[-1], id])
                track = Fragment(id, t,x,y)
                # ordered_tracks.append(track)
                all_tracks[id] = track

        heapq.heapify(S) # min heap (frame, id)
        heapq.heapify(E)

        while E:
            e, id = heapq.heappop(E)
            ordered_tracks.append(all_tracks[id])
        del all_tracks
            

        # Initialize
        # X = UndirectedGraph() # exclusion graph
        running_tracks = OrderedDict() # tracks that start but not end at e 
        curr_tracks = deque() # tracks in view. list of tracks. should be sorted by end_time
        past_tracks = OrderedDict() # set of ids indicate end of track ready to be matched

        path = {} # oldid: newid. to store matching assignment
        matched = 0 # count matched pairs 
        
        start = time.time()
        for i,track in enumerate(ordered_tracks):
            # print("\n")
            # print('Adding new track {}/{},{}'.format(i, len(ordered_tracks),track.id))
            # print("Past tracks: {}".format(len(past_tracks)))
            # print("Curr tracks: {}".format(len(curr_tracks)))
            # print("running tracks: {}".format(len(running_tracks)))
            # print("path bytes: {}".format(sys.getsizeof(path)))
            curr_id = track.id # last_track = track['id']
            path[curr_id] = curr_id
            right = track.t[-1] # right pointer: current time
            
            # get tracks that started but not end - used to define the window left pointer
            while S and S[0][0] < right: # append all the tracks that already starts
                started_time, started_id = heapq.heappop(S)
                running_tracks[started_id] = started_time
                
            # compute track statistics
            track._computeStats()
            
            try: 
                left = max(0,_first(running_tracks) - time_out)
            except: left = 0
            
            # print("window size :", right-left)
            # remove out of sight tracks 
            while curr_tracks and curr_tracks[0].t[-1] < left: 
                past_track = curr_tracks.popleft()
                past_tracks[past_track.id] = past_track
            # print("Curr_tracks ", [i.id for i in curr_tracks])
            # print("past_tracks ", past_tracks.keys())
            # compute score from every track in curr to track, update Cost
            for curr_track in curr_tracks:
                cost = _getCost(curr_track, track, time_out, VARX, VARY)
                if cost > THRESHOLD_MAX:
                    curr_track._addConflict(track)
                elif cost > 0:
                    curr_track._addSuc(cost, track)
                    track._addPre(cost, curr_track)
                          
            prev_size = 0
            curr_size = len(past_tracks)
            while curr_size > 0 and curr_size != prev_size:
                prev_size = len(past_tracks)
                remove_keys = set()
                # ready = _first(past_tracks) # a fragment object
                for ready_id, ready in past_tracks.items():
                    best_head = ready._getFirstSuc()
                    if not best_head or not best_head.pre: # if ready has no match or best head already matched to other tracks# go to the next ready
                        # past_tracks.pop(ready.id)
                        remove_keys.add(ready.id)
                    
                    else:
                         try: best_tail = best_head._getFirstPre()
                         except: best_tail = None
                         if best_head and best_tail and best_tail.id == ready.id and best_tail.id not in ready.conflicts_with:
                            # print("** match tail of {} to head of {}".format(best_tail.id, best_head.id))
                            path[best_head.id] = path[best_tail.id]
                            remove_keys.add(ready.id)
                            Fragment._matchTailHead(best_tail, best_head)
                            matched += 1

                [past_tracks.pop(key) for key in remove_keys]
                curr_size = len(past_tracks)
                
                
                    
            curr_tracks.append(track)        
            running_tracks.pop(track.id) # remove tracks that ended
            # print("matched:", matched)
            
        # delete IDs that are empty
        # print("\n")
        # print("{} Ready: ".format(past_tracks.printList()))
        # print("{} Processsed: ".format(len(processed)))
        print("{} pairs matched".format(matched))
        # print("Deleting {} empty tracks".format(len(empty_id)))
        # df = df.groupby("ID").filter(lambda x: (x["ID"].iloc[0] not in empty_id))
        end = time.time()
        print('run time online stitching:', end-start)
        # for debugging only
        o.path = path
        # o.C = C
        # o.X = X
        o.groupList = ids
        o.past_tracks = past_tracks.keys()
        # replace IDs
        newids = [v for _,v in path.items()]
        m = dict(zip(path.keys(), newids))
        df = df.replace({'ID': m})
        df = df.sort_values(by=['Frame #','ID']).reset_index(drop=True)         
        
        print("Before DA: {} unique IDs".format(len(ids))) 
        print("After DA: {} unique IDs".format(df.groupby("ID").ngroups))
        print("True: {} unique IDs".format(len([id for id in ids if id<1000])))
      
        o.df = df
        return o