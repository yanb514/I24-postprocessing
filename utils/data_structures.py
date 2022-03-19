import heapq
from collections import defaultdict,OrderedDict
import numpy as np

# A linked list node
class Node:

    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

# Class to create a Doubly Linked List
class DoublyLinkedList:

    # Constructor for empty Doubly Linked List
    def __init__(self):
        self.head = None
        self.size = 0

    # Given a reference to the head of a list and an
    # integer, inserts a new node on the front of list
    def push(self, new_data):

        # 1. Allocates node
        # 2. Put the data in it
        new_node = Node(new_data)

        # 3. Make next of new node as head and
        # previous as None (already None)
        new_node.next = self.head

        # 4. change prev of head node to new_node
        if self.head is not None:
            self.head.prev = new_node

        # 5. move the head to point to the new node
        self.head = new_node
        self.size += 1
        
    # Given a node as prev_node, insert a new node after
    # the given node
    def insertAfter(self, prev_node, new_data):

        # 1. Check if the given prev_node is None
        if prev_node is None:
            print("the given previous node cannot be NULL")
            return

        # 2. allocate new node
        # 3. put in the data
        new_node = Node(new_data)

        # 4. Make net of new node as next of prev node
        new_node.next = prev_node.next

        # 5. Make prev_node as previous of new_node
        prev_node.next = new_node

        # 6. Make prev_node ass previous of new_node
        new_node.prev = prev_node

        # 7. Change previous of new_nodes's next node
        if new_node.next:
            new_node.next.prev = new_node

        self.size+=1
        
    # Given a reference to the head of DLL and integer,
    # appends a new node at the end
    def append(self, new_data):

        # 1. Allocates node
        # 2. Put in the data
        new_node = Node(new_data)

        # 3. This new node is going to be the last node,
        # so make next of it as None
        # (It already is initialized as None)

        # 4. If the Linked List is empty, then make the
        # new node as head
        if self.head is None:
            self.head = new_node
            self.size += 1
            return

        # 5. Else traverse till the last node
        last = self.head
        while last.next:
            last = last.next

        # 6. Change the next of last node
        last.next = new_node

        # 7. Make last node as previous of new node
        new_node.prev = last
        self.size += 1
        return

    def delete_element(self, x):
        # TODO: make it more efficient if input is a node pointer
        if self.head is None:
            # print("The list has no element to delete")
            return 
        if self.head.next is None:
            if self.head.data == x:
                self.head = None
                self.size = 0
            # else:
            #     print("Item not found")
            return 

        if self.head.data == x:
            self.head = self.head.next
            self.head.prev = None
            self.size -= 1
            return

        n = self.head
        while n.next is not None:
            if n.data == x:
                break
            n = n.next
        if n.next is not None:
            n.prev.next = n.next
            n.next.prev = n.prev
            self.size -= 1
        else:
            if n.data == x:
                n.prev.next = None
                self.size -= 1
            # else:
            #     print("Element not found")
        
    # This function prints contents of linked list
    # starting from the given node
    def printList(self):

        node = self.head
        temp = []
        while node:
            temp.append(node.data)
            # print(" {}".format(node.data))
            # last = node
            node = node.next
        print(temp)

    def convert_to_set(self):

        node = self.head
        s = set()
        while node:
            s.add(node.data)
            # print(" {}".format(node.data))
            # last = node
            node = node.next
        return s



class UndirectedGraph(defaultdict):
    def __init__(self):
        super().__init__(set)
        
    # add two-way edge
    def _addEdge(self,u,v):
        self[u].add(v)
        self[v].add(u)
        
    # remove node u and all edges to/from it
    def _remove(self, u):
        neighbors = self[u]
        for v in neighbors:
            self[v].remove(u)
        self.pop(u)
        
    # make u and v's neighbors the union of their neighbors
    def _union(self, u, v):
        nei_u = self[u]
        nei_v = self[v]
        
        u_v = nei_u-nei_v
        for w in u_v:
            self._addEdge(w,v)
            
        v_u = nei_v-nei_u
        for w in v_u:
            self._addEdge(w,u)
            
    def _printGraph(self):
        for key,val in self.items():
            print(key, val)
 
class Fragment:
    # Constructor to create a new fragment
    def __init__(self, id,t,x,y):
        self.ready = False # if tail is ready to be matched
        self.id = id
        self.t = t
        self.x = x
        self.y = y
        self.suc = [] # tail matches to [(cost, Fragment_obj)] - minheap
        self.pre = [] # head matches to [(cost, Fragment_obj)] - minheap
        self.conflicts_with = set() # keep track of conflicts - bi-directional
    
    def _computeStats(self):
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
        # match u's tail to v's head
        # 1. add u's conflicts to v -> contagious conflicts!
        nei_u = u.conflicts_with
        nei_v = v.conflicts_with     
        u_v = nei_u-nei_v
        for w in u_v:
            v._addConflict(w)      
        # v_u = nei_v-nei_u
        # for w in v_u:
        #     u._addEdge(u)
            
        # 2. remove all u's succ from u -> remove all from v's pre
        v.pre = None
        # 3. delete u
        u._delete()
        return
    
    # remove node u and all edges to/from it
    def _remove(self, u):
        neighbors = self[u]
        for v in neighbors:
            self[v].remove(u)
        self.pop(u)


        
if __name__ == "__main__":
    ug = UndirectedGraph()
    ug._addEdge(1,2)
    ug._addEdge(3,4)
    ug._addEdge(4,5)
    
    # ug._printGraph()
    ug._union(2,4)
    ug._printGraph()
    
    ug._remove(4)
    ug._printGraph()