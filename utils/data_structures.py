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
    def __init__(self, doc):
        '''
        doc is a record from raw_trajectories database collection
        '''
        self.id = doc["_id"]
        self.t = np.array(doc["timestamp"])
        self.x = np.array(doc["x_position"])
        self.y = np.array(doc["y_position"])
        self.dir = np.array(doc["direction"])
        self.suc = [] # tail matches to [(cost, Fragment_obj)] - minheap
        self.pre = [] # head matches to [(cost, Fragment_obj)] - minheap
        self.conflicts_with = set() # keep track of conflicts - bi-directional
        self.ready = False # if tail is ready to be matched
    
    def _computeStats(self):
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
    def _addSuc(self, cost, fragment):
        heapq.heappush(self.suc, (cost, fragment))
    
    # add predecessor
    def _addPre(self, cost, fragment):
        heapq.heappush(self.pre, (cost, fragment))
       
    def _getFirstSuc(self):
        # return the best successor of self if exists
        # otherwise return None
        # heappop empty fragments from self.suc
        while self.suc and self.suc[0][1].id is None: # get the first non-empty fragment
            _, suc = heapq.heappop(self.suc)
        try: suc = self.suc[0][1] # self.suc[0] is None
        except: suc = None
        return suc
    
    def _getFirstPre(self):
        # return the best successor of self if exists
        # otherwise return None
        # heappop empty fragments from self.suc
        while self.pre and self.pre[0][1].id is None: # get the first non-empty fragment
            _, pre = heapq.heappop(self.pre)
        try: pre = self.pre[0][1]
        except: pre = None
        return pre
    
    # clear the reference once out of sight or matched
    def _delete(self):
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
    def _addConflict(self, fragment):
        if fragment.id: # only add if fragment is valid (not out of view)
            self.conflicts_with.add(fragment)
            fragment.conflicts_with.add(self)
        return
    
    # union conflicted neighbors, set head's pre to [], delete self
    @classmethod
    def _matchTailHead(cls, u,v):
        # by the call, u = v._getFirstPre and u._getFirstSuc = v
        # match u's tail to v's head
        # 1. add u's conflicts to v -> contagious conflicts!
        nei_u = u.conflicts_with
        nei_v = v.conflicts_with     
        u_v = nei_u-nei_v
        for w in u_v:
            v._addConflict(w)      
            
        # 2. remove all u's succ from u -> remove all from v's pre
        # 4/13/22 modified from v.pre = None to heapq.heappop(v.pre), because v's head can still be matched to others
        # v.pre = None 
        #TODO: not tested
        heapq.heappop(v.pre)
        
        # 3. delete u 
        #TODO: completely delete u from memorys
        u._delete()
        return




# A node in DisjointSet1
class Node:

    # Constructor to create a new node
    def __init__(self, doc=None):
        if not doc: # for dummy node
            self.parent = None
            self.root = self
            self.child = None
        else:
            self.id = doc["_id"]
            self.parent = None
            self.last_timestamp = doc["last_timestamp"] # for sorting children
            self.last_modified = doc["last_timestamp"] # for cache
            # TODO: for now children is a list, when printing, sort the list in the end
            # could optimize this
            self.child = None # for printing path from root 
            self.root = self

    def __repr__(self):
        return 'Node({!r})'.format(self.id)

        
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
        - parent = NOne to initialize
    '''
    
    def __init__(self):
        # store a LRU cache (key: root_id, val: last_timestamp in assignment)
        # why OderedDict(): support "least recently used" cache.
        # items at the front of the cache are less recently used, and therefore is more likely to be outputted
        # why keep last_timestamp? to compare with current_time and idle_time to detetermine if output is ready
        # why not heap? cannot query and remove items in constant time
        self.cache = OrderedDict()
        self.path = {} # just a collection for all the Nodes' memory locations(key: id, val: Node object)
    
    def _makeSet(self, docs):
        for doc in docs:
            self._addNode(doc)
            
    def _addNode(self, doc):
        node = Node(doc) # create a new node
        self.cache[node.id] = node
        self.path[node.id] = node

    # Find the root of the set in which element `k` belongs
    def _find(self, node):
        # # if `node` is not the root
        # if node.root != node:
        #     # dfs path compression
        #     node.root = self._find(node.root)
        # return node.root
    
        if not node.parent: # if node is the root
            return node
        # path compression
        node.root = node.parent.root
        return self._find(node.parent)

        
    # Perform Union of two subsets
    def _union(self, id1, id2):
        print("Union {} {}".format(id1, id2))
        # assumes id2 comes after id1
        
        # find the root of the sets in which Nodes `id1` and `id2` belong
        node1, node2 = self.path[id1], self.path[id2]
        root1 = self._find(node1)
        root2 = self._find(node2)
        
        # if `id1` and `id2` are present in the same set, only move to the end of cache
        if root1 == root2:
            self.cache[root1.id].last_modified = max(self.cache[root1.id].last_modified, node2.last_timestamp)
            self.cache.move_to_end(root1.id)
            return
        
        # compress path: update parent and child pointers for node1 and node2
        # they should be on the same path from the shared root
        # by matching logic, node1 has no child
        
        p1,p2 = node1, node2
        if node2.child:
            node2_is_leaf = False
            head = node2.child
        else:
            node2_is_leaf = True
            head = Node() # create dummy 

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
            
        if node2_is_leaf:
            node2.child = None
                

        # TODO update root along the path, by now they should have a shared root
        # root2.parent = root1
        
        # # update LRU cache
        # self.cache[root1.id].last_modified = max(self.cache[root1.id].last_modified, node2.last_timestamp)
        # self.cache.move_to_end(root1.id)
        # # delete root2 from cache
        try:
            self.cache.pop(node2.id)
        except:
            pass


    def _printSets(self):
        
        print([self._find(val).id for key,val in self.path.items()])
        return

    def _dfs(self, node, path):      
        if node:
            path.append(node.id) 
            self._dfs(node.child, path)
            # for child in node.children:
            #     self._dfs(child, path)
 
    # def _getPath(self, root):
    #     '''
    #     get all the paths from roots, whatever remains in self.cache.keys are roots!
    #     DFS
    #     '''
    #     # TODO: use try catch, test this
    #     if root.id not in self.cache:
    #         print("root not in cache")
    #         returnall_paths = [] # nested lists

    #     path = []
    #     self._dfs(root, path)
            
    #     return path 
          

    def _getAllPaths(self):
        '''
        get all the paths from roots, whatever remains in self.cache.keys are roots!
        DFS
        # TODO: sort paths by last_timestamp
        '''
        all_paths = [] # nested lists
        
        for node in self.cache.values():
            # print(node.last_timestamp)
            path = []
            self._dfs(node, path)
            all_paths.append(path)
            
        return all_paths
    
    def _printParents(self):
        for node in self.path.values():
            try:
                print("Node {}: parent: {}".format(node.id, node.parent.id))
            except:
                print("Node {} has no parent".format(node.id))
    
    def _printChildren(self):
        for node in self.path.values():
            try:
                print("Node {}: parent: {}".format(node.id, node.child.id))
            except:
                print("Node {} has no child".format(node.id))
                
    def _outputPath(self, curr_time, idle_time):
        # TODO
        return
        
        
        
    
 

if __name__ == '__main__':
 
    # universe of items
    # create synthetic docs
    docs = []
    ids = list("abcdefghi")
    last_timestamps = np.arange(len(ids))
    
    for i in range(len(ids)):
        docs.append({
            "_id": ids[i],
            "last_timestamp": last_timestamps[i]})
            
    
 
    # initialize `DisjointSet` class
    ds = PathCache()
 
    # create a singleton set for each element of the universe
    ds._makeSet(docs)
    
    ds._union("d", "i")  
    # print(ds.cache)     
    # print(ds._getAllPaths())
    # print(ds.cache.keys())
    ds._printParents()
    
    ds._union("a", "g") 
    # print(ds._getAllPaths())
    # print(ds.cache.keys())
    
    ds._union("c", "g")
    # print(ds._getAllPaths())
    # print(ds.cache.keys())
    
    ds._union("b", "e") 
    # print(ds.cache.keys())

    # print(ds.cache)
    # ds._printSets()
    # print(ds.path)
    
    ds._printParents()
    ds._printChildren()
    all_paths = ds._getAllPaths()
    
    print(all_paths)
    