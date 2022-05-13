import heapq
import numpy as np

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
    def __init__(self):
      self.sentinel = Node(None)
      self.sentinel.next = self.sentinel
      self.sentinel.prev = self.sentinel
      self.cache = {} # key: fragment_id, val: Node address with that fragment in it, keep the address of pointers

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

        self.cache[node.id] = node 
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
        self.cache[node.id] = node 
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
        self.cache.pop(node.id)
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
        return self.print_list("id")
    
    def print_list(self, attr_name="id"):
        s = "["
        x = self.sentinel.next
        while x != self.sentinel:  # look at each node in the list
            try:
                s += getattr(x, attr_name)
            except:
                s += "x"
            if x.next != self.sentinel:
                s += ", "   # if not the last node, add the comma and space
            x = x.next
        s += "]"
        return s      
        
    
    
    
class Fragment(Node):
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
                    if field_names[i] == "_id": # cast bson ObjectId type to str
                        setattr(self, attr_names[i], str(traj_doc[field_names[i]]))
                    else:
                        setattr(self, attr_names[i], traj_doc[field_names[i]])
                except: pass
            super().__init__(None)
            

            try:
                setattr(self, "x", np.array(self.x)*0.3048)
                setattr(self, "y", np.array(self.y)*0.3048)
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
        
    # def __del__(self):
    #     '''
    #     TODO: destroy this object and all reference to it at once?
    #     '''
    #     pass

    
    def compute_stats(self):
        '''
        compute statistics for matching cost
        based on linear vehicle motion (least squares fit constant velocity)
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
    # TODO:
        - how to utilize cache? what to keep track of?
        - stress test
        - can it replace / integrate with Fragment object?
        - parent = None to initialize
    '''
    def __init__(self):
        super().__init__()
        self.path = {} # keep pointers for all Fragments
        
    def make_set(self, docs):
        for doc in docs:
            self.add_node(doc)
            
    def add_node(self, node):
        if not isinstance(node, Fragment):
            node = Fragment(node) # create a new node
        # self.cache[node.id] = node
        self.append(node)
        # try:
        #     self.path[node.ID] = node
        # except:
        self.path[node.id] = node

    def get_fragment(self, id):
        return self.path[id]

    
    # Find the root of the set in which element `k` belongs
    def find(self, node):
    
        if not node.parent: # if node is the root
            return node
        # delete node from cache, because node is not the root
        try:
            self.delete(node.id)
        except:
            pass
        # path compression
        node.root = node.parent.root 
        return self.find(node.parent)

        
    # Perform Union of two subsets
    def union(self, id1, id2):
        # id2 comes after id1
        
        # find the root of the sets in which Nodes `id1` and `id2` belong
        node1, node2 = self.path[id1], self.path[id2]
        root1 = self.find(node1)
        root2 = self.find(node2)
        
        # if `id1` and `id2` are present in the same set, only move to the end of cache
        if root1 == root2:
            self.update(root1.id, max(root1.tail_time, node2.tail_time), attr_name = "tail_time")
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
        new_root1 = self.find(node1)
        node1.root = new_root1
        new_root2 = self.find(node2)
        node2.root = new_root2

        # update tail_time for all nodes along the path
        # self.path_down_update(new_root1) # TODO probably unnecessary
        self.update(new_root1.id, max(new_root1.tail_time, node2.tail_time), attr_name = "tail_time")

        
    def print_sets(self):
        
        print([self.find(val).id for key,val in self.path.items()])
        return
          

    def get_all_paths(self, attr_name="id"):
        '''
        get all the paths from roots, whatever remains in self.cache.keys are roots!
        DFS
        '''
        all_paths = [] # nested lists
         
        # for DLL cache
        node = self.first_node()
        while node != self.sentinel:
            path = self.path_down(node, attr_name)
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
            print("Root {}: tail_time is {}".format(root.id, root.tail_time))
        
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
                path.append(node.id) 
                _dfs(node.parent, path)
        _dfs(node, path)
        return path
    
    def path_down(self, node, attr_name="id"):
        path = []
        def _dfs(node, path):
            if node:
                path.append(getattr(node, attr_name)) 
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
    pc.make_set(docs)
    
    pc.union("a", "f")  
    
    pc.union("b", "e") 
    pc.union("e", "f")  
    pc.union("c", "d") 

    
    
    
   
#    print(pc.popFirstPath())
#    print(pc._popFirstPath())
    for root in pc.get_all_roots():
        path = pc.path_down_update(pc.path[root.id])
    pc.print_attr("tail_time")
    pc.print_attr("root")
    all_paths = pc.get_all_paths(attr_name = "id")
    print(all_paths)
    