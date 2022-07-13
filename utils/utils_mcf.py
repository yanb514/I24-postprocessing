import heapq
import numpy as np
import networkx as nx
import queue
from collections import deque
from utils.utils_stitcher import min_nll_cost, nll, nll_modified, cost_1

class Node:
    '''
    A generic Node object to use in linked list
    '''
    def __init__(self, data):
        '''
        :param data: a dictionary of {feature: values}
        '''
        # customized features
        if data:
            for key, val in data.items():
                setattr(self, key, val)
                
        # required features
        self.next = None
        self.prev = None
        
    def __repr__(self):

        try:
            return 'Node({!r})'.format(self.ID)
        except:
            try:
                return 'Node({!r})'.format(self.id)
            except:
                return 'Sentinel Node'

    
# Create a sorted doubly linked list class
class SortedDLL:
    '''
    Sorted dll by a specified node's attribute
    original code (unsorted) from  http://projectpython.net/chapter17/
    Hierarchy:
        SortedDll()
            - Node()
    
    '''
    def __init__(self, attr = "id"):
      self.sentinel = Node(None)
      self.sentinel.next = self.sentinel
      self.sentinel.prev = self.sentinel
      self.cache = {} # key: fragment_id, val: Node address with that fragment in it, keep the address of pointers
      self.attr = attr
      
    def count(self):
        return len(self.cache)
    
    # Return a reference to the first node in the list, if there is one.
    # If the list is empty, return None.
    def first_node(self):
        if self.sentinel.next == self.sentinel:
            return None
        else:
            return self.sentinel.next
        
    # Insert a new node with data after node pivot.
    def insert_after(self, pivot, node):
        if isinstance(node, dict):
            node = Node(node)

        self.cache[getattr(node, self.attr)] = node 
        # Fix up the links in the new node.
        node.prev = pivot
        node.next = pivot.next
        # The new node follows x.
        pivot.next = node
        # And it's the previous node of its next node.
        node.next.prev = node
        
        
    # Insert a new node with data after node pivot.
    def insert_before(self, pivot, node):
        if isinstance(node, dict):
            node = Node(node)
        self.cache[getattr(node, self.attr)] = node 
        # Fix up the links in the new node.
        node.next = pivot
        node.prev = pivot.prev
        # The new node suceeds pivot.
        pivot.prev = node
        # And it's the previous node of its next node.
        node.prev.next = node
            
    # Insert a new node at the end of the list.
    def append(self, node):
        if not isinstance(node, Node):
            node = Node(node) 
        last_node = self.sentinel.prev
        self.insert_after(last_node, node)
        self.swim_up(node) # insert to the correct order
    
    # Delete node x from the list.
    def delete(self, node):
        if not isinstance(node, Node):
            try:
                node = self.cache[node]
            except:
                # no id's available in cache
                return None
        # Splice out node x by making its next and previous
        # reference each other.
        node.prev.next = node.next
        node.next.prev = node.prev
        self.cache.pop(getattr(node, self.attr))
        return node
      
    # def find(self, data):
    #     return self.cache[data.id]  
    # def get_node(self, id):
    #     return self.cache[id] # KeyError if not in cache
    
    def update(self, key, attr_val, attr_name = "tail_time"):
        # update the position where data lives in DLL
        # move up if data.tail_time less than its prev
        # move down if data.tail_time is larger than its next
        if key not in self.cache:
            print("key doesn't exists in update() SortedDll")
            return
        
        node = self.cache[key]
        setattr(node, attr_name, attr_val)
        
        # check if needs to move down
        if node == self.sentinel.next: # if node is head
            self.swim_down(node)
            
        elif node == self.sentinel.prev: # if node is tail
            self.swim_up(node)
            
        elif getattr(node, attr_name) >= getattr(node.next, attr_name): # node is in-between head and tail, compare
            self.swim_down(node)
        
        elif getattr(node, attr_name) < getattr(node.prev, attr_name):
            self.swim_up(node)
            
        return
        
    def get_attr(self, attr_name="tail_time"):
        arr = []
        head = self.sentinel.next
        if attr_name == "self":
           while head != self.sentinel:
               arr.append(head)
               head = head.next 
        else:
            while head != self.sentinel:
                arr.append(getattr(head, attr_name))
                head = head.next
        return arr
    
    def swim_down(self, node, attr_name="tail_time"):
        pointer = node
        val = getattr(node, attr_name)
        
        while pointer.next != self.sentinel and val >= getattr(pointer.next, attr_name):
            pointer = pointer.next
        if pointer == node:
            return # node is already at the correct position
        
        # insert node right after pointer
        # delete node and insert node after pointer
        node = self.delete(node)
        self.insert_after(pointer, node)
        return
    
    def swim_up(self, node, attr_name="tail_time"):
        pointer = node
        val = getattr(node, attr_name)
        
        while pointer.prev != self.sentinel and val < getattr(pointer.prev, attr_name):
            pointer = pointer.prev
        if pointer == node:
            return # node is already at the correct position
        
        # move node right before pointer
        node = self.delete(node)
        self.insert_before(pointer, node)
        return
    
    # Return the string representation of a circular, doubly linked
    # list with a sentinel, just as if it were a Python list.
    def __repr__(self):
        return self.print_list()
    
    def print_list(self):
        s = "["
        x = self.sentinel.next
        while x != self.sentinel:  # look at each node in the list
            try:
                s += getattr(x, self.attr)
            except:
                s += "x"
            if x.next != self.sentinel:
                s += ", "   # if not the last node, add the comma and space
            x = x.next
        s += "]"
        return s      
        
    
    
    
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

    
    
    
    
    
    
    
    
    
    
    