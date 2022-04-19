import heapq
from collections import defaultdict,OrderedDict
import numpy as np
#from time import sleep



class LRUCache:
    def __init__(self, Capacity):
        self.size = Capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache: return -1
        val = self.cache[key]
        self.cache.move_to_end(key)
        return val

    def put(self, key, val):
        if key in self.cache: del self.cache[key]
        self.cache[key] = val
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)

 
class Fragment:
    # Constructor to create a new fragment
    def __init__(self, doc = None):
        '''
        doc is a record from raw_trajectories database collection
        '''
        self.suc = [] # tail matches to [(cost, Fragment_obj)] - minheap
        self.pre = [] # head matches to [(cost, Fragment_obj)] - minheap
        self.conflicts_with = set() # keep track of conflicts - bi-directional
        self.ready = False # if tail is ready to be matched
    
        self.child = None # for printing path from root 
        self.root = self # 
        self.parent = None
        self.tail_matched = False # flip to true when its tail matches to another fragment's head
            
        if doc:   
            field_names = ["_id","timestamp","x_position","y_position","direction","last_timestamp","last_timestamp" ]
            attr_names = ["id","t","x","y","dir","last_timestamp","last_modified_timestamp"]
            for i in range(len(field_names)): # set as many attributes as possible
                try:
                    setattr(self, attr_names[i], doc[field_names[i]])
                except:
                    pass
            
    def __repr__(self):
        return 'Fragment({!r})'.format(self.id)
    
    def compute_stats(self):
        '''
        compute statistics for matching cost
        based on linear vehicle motion (least squares fit constant velocity)
        '''
        t,x,y = self.t, self.x, self.y
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
        self.fitx = fitx
        self.fity = fity
        return

    # add successor to fragment with matching cost
    def add_suc(self, cost, fragment):
        heapq.heappush(self.suc, (cost, fragment))
    
    # add predecessor
    def add_pre(self, cost, fragment):
        heapq.heappush(self.pre, (cost, fragment))
       
    def get_first_suc(self):
        # return the best successor of self if exists
        # otherwise return None
        # heappop empty fragments from self.suc
        while self.suc and self.suc[0][1].id is None: # get the first non-empty fragment
            _, suc = heapq.heappop(self.suc)
        try: suc = self.suc[0][1] # self.suc[0] is None
        except: suc = None
        return suc
    
    def get_first_pre(self):
        # return the best successor of self if exists
        # otherwise return None
        # heappop empty fragments from self.suc
        while self.pre and self.pre[0][1].id is None: # get the first non-empty fragment
            _, pre = heapq.heappop(self.pre)
        try: pre = self.pre[0][1]
        except: pre = None
        return pre
    
    # clear the reference once out of sight or matched
    # TODO: call a destructor? - a Fragment object will be permanently deleted if the path it belongs to is written to the stitched_trajectories database
    def delete(self):
        self.ready = False # if tail is ready to be matched
        self.id = None
        self.t = None
        self.x = None
        self.y = None
        self.suc = None # tail matches to [(cost, Fragment_obj)] - minheap
        self.pre = None # head matches to [(cost, Fragment_obj)] - minheap
        self.conflicts_with = None
        # del self.id, self.t, self.x
       
    # add bi-directional conflicts
    def add_conflict(self, fragment):
        if fragment.id: # only add if fragment is valid (not out of view)
            self.conflicts_with.add(fragment)
            fragment.conflicts_with.add(self)
        return
    
    # union conflicted neighbors, set head's pre to [], delete self
    @classmethod
    def match_tail_head(cls, u,v):
        # by the call, u = v._getFirstPre and u._getFirstSuc = v
        # match u's tail to v's head
        # 1. add u's conflicts to v -> contagious conflicts!
        nei_u = u.conflicts_with
        nei_v = v.conflicts_with     
        u_v = nei_u-nei_v
        for w in u_v:
            v.add_conflict(w)      
            
        # 2. remove all u's succ from u -> remove all from v's pre
        # 4/13/22 modified from v.pre = None to heapq.heappop(v.pre), because v's head can still be matched to others
        #TODO: not tested
        heapq.heappop(v.pre)
        
        # 3. delete u 
        # 4/16/22 modifed from u.delete() to u.tail_matched = True. Reason: only delete when path_cache.popFirstPath(), to prevent data loss
        # TODO: not tested
        u.delete()
        u.tail_matched = True
        return

        
# A class to represent a disjoint set
class PathCache:
    '''
    This class combines a union-find data structure and an LRU data structure
    Purpose:
        - keep track of root (as the first-appeared fragments) and their successors in paths
        - output paths to stitched_trajectories database if time out
    # TODO:
        - how to utilize cache? what to keep track of?
        - stress test
        - can it replace / integrate with Fragment object?
        - parent = None to initialize
    '''
    
    def __init__(self):
        # store a LRU cache (key: root_id, val: last_timestamp in assignment)
        # why OderedDict(): support "least recently used" cache.
        # items at the front of the cache are less recently used, and therefore is more likely to be outputted
        # why keep last_timestamp? to compare with current_time and idle_time to detetermine if output is ready
        # why not heap? cannot query and remove items in constant time
        self.cache = OrderedDict()
        self.path = {} # just a collection for all the Nodes' memory locations(key: id, val: Node object)
    
    def make_set(self, docs):
        for doc in docs:
            self.add_node(doc)
            
    def add_node(self, doc):
        node = Fragment(doc) # create a new node
        self.cache[node.id] = node
        self.path[node.id] = node

    # Find the root of the set in which element `k` belongs
    def find(self, node):
    
        if not node.parent: # if node is the root
            return node
        # delete node from cache, because node is not the root
        try:
            self.cache.pop(node.id)
        except:
            pass
        # path compression
#        if node.parent.root.last_timestamp <= node.parent.last_timestamp:
        node.root = node.parent.root 
        
        node.last_modified_timestamp = max(node.last_modified_timestamp, node.root.last_modified_timestamp)
        return self.find(node.parent)

        
    # Perform Union of two subsets
    def union(self, id1, id2):
#        print("Union {} {}".format(id1, id2))
        # assumes id2 comes after id1
        
        # find the root of the sets in which Nodes `id1` and `id2` belong
        node1, node2 = self.path[id1], self.path[id2]
        root1 = self.find(node1)
        root2 = self.find(node2)
        
        # if `id1` and `id2` are present in the same set, only move to the end of cache
        if root1 == root2:
            self.cache[root1.id].last_modified_timestamp = max(self.cache[root1.id].last_modified_timestamp, node2.last_timestamp)
            self.cache.move_to_end(root1.id)
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
            head = Fragment() # create dummy 
            head.id = -1

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
        root1 = self.find(node1)
        node1.root = root1
        root2 = self.find(node2)
        node2.root = root2

        # update LRU cache
        self.cache[root2.id].last_modified_timestamp = max(self.cache[root2.id].last_modified_timestamp, node2.last_timestamp)
        self.cache.move_to_end(root2.id)


    def print_sets(self):
        
        print([self.find(val).id for key,val in self.path.items()])
        return
          

    def get_all_paths(self):
        '''
        get all the paths from roots, whatever remains in self.cache.keys are roots!
        DFS
        '''
        all_paths = [] # nested lists
        
        for node in self.cache.values():
            # print(node.last_timestamp)
            path = self.path_down(node)
            all_paths.append(path)
            
        return all_paths
    
                
    def get_all_roots(self):
        return self.cache.values()
    
    def print_cache(self):
        roots = self.get_all_roots()
        for root in roots:
            print("Root {}: last_modified is {}".format(root.id, root.last_modified))
        
    def print_attr(self, attr_name):
        for node in self.path.values():
            try:
                print("Node {}: {}: {}".format(node.id, attr_name, getattr(node, attr_name)))
            except:
                print("Node {} has no {}".format(node.id, attr_name))
                
    def pop_first_path(self):
        '''
        pop the first node (root) from cache if cache is not empty
        delete all nodes along the path in self.path
        return the path: a list of ids
        '''
        try:
            root_id, root_node = self.cache.popitem()
            path = self.path_down(root_node)
            for p in path:
                self.path.pop(p)
            return path
        except StopIteration:
            raise Exception
        
    def path_up(self, node):
        path = []
        def _dfs(node, path):
            if node:
                path.append(node.id) 
                _dfs(node.parent, path)
        _dfs(node, path)
        return path
    
    def path_down(self, node):
        path = []
        def _dfs(node, path):
            if node:
                path.append(node.id) 
                _dfs(node.child, path)
        _dfs(node, path)
        return path       
        
        
    
 

if __name__ == '__main__':
 
    # universe of items
    # create synthetic docs
    docs = []
    ids = list("abcdef")
    last_timestamps = np.arange(len(ids))
    
    for i in range(len(ids)):
        docs.append({
            "_id": ids[i],
            "last_timestamp": last_timestamps[i]})
            
    
 
    # initialize `DisjointSet` class
    pc = PathCache()
 
    # create a singleton set for each element of the universe
    pc.makeSet(docs)
    
    pc.union("a", "f")  
    # print(pc.cache)     
#    print(pc._getAllPaths())
    # print(pc.cache.keys())
#    pc.printParents()
    
    pc.union("b", "e") 
#    pc.printRoots()
#    print(pc._getAllPaths())
    # print(pc.cache.keys())
    
    pc.union("e", "f")
#    pc._printRoots()
#    print(pc._getAllPaths())
    # print(pc.cache.keys())
    
    pc.union("c", "d") 
#    pc._printRoots()
#    print(pc._getAllPaths())
#    pc._printRoots()

    pc.union("d", "f") 
    # print(pc.cache.keys())

    # print(pc.cache)
    # pc._printSets()
    # print(pc.path)
    
    pc.printAttr("root")
    all_paths = pc.getAllPaths()
    
    print(all_paths)
#    print(pc.popFirstPath())
#    print(pc._popFirstPath())
    