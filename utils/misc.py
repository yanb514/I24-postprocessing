from i24_logger.log_writer import catch_critical
import numpy as np
from sklearn import linear_model




@catch_critical(errors = (Exception))
def add_filter(traj, raw, residual_threshold_x, residual_threshold_y, 
               conf_threshold, remain_threshold):
    '''
    add a filter to trajectories based on
    - RANSAC fit on x and
    - bad detection confidence
    get total mask (both lowconf and outlier)
    apply ransac again on y-axis
    save fitx, fity and tot_mask
    and save filter field to raw collection
    '''
    filter = True
    try:
        filter.pop("filter")
    except:
        pass
    t = np.array(traj["timestamp"])
    x = np.array(traj["x_position"])
    y = np.array(traj["y_position"])
    conf = np.array(traj["detection_confidence"])
        
    # length = len(t)
    
    # get confidence mask
    lowconf_mask = np.array(conf < conf_threshold)
    highconf_mask = np.logical_not(lowconf_mask)
    num_highconf = np.count_nonzero(highconf_mask)
    if num_highconf < 4:
        traj["filter"] = []
    
    else:
        # fit x only on highconf
        ransacx = linear_model.RANSACRegressor(residual_threshold=residual_threshold_x)
        X = t.reshape(1, -1).T
        ransacx.fit(X[highconf_mask], x[highconf_mask])
        fitx = [ransacx.estimator_.coef_[0], ransacx.estimator_.intercept_]
        
        # --- with ransac outlier mask
        # inlier_mask = ransacx.inlier_mask_ # True if inlier
        # outlier_mask = np.logical_not(inlier_mask) # True if outlier
        # total mask (filtered by both outlier and by low confidence)
        # mask1 = np.arange(length)[lowconf_mask] # all the bad indices
        # mask2 = np.arange(length)[highconf_mask][outlier_mask]
        
        # bad_idx = np.concatenate((mask1, mask2))
        # remain = length-len(bad_idx)
        
        # # print("bad rate: {}".format(bad_ratio))
        # if remain < remain_threshold:
        #     filter = []
      
        # # fit y only on mask
        # ransacy = linear_model.RANSACRegressor(residual_threshold=residual_threshold_y)
        # ransacy.fit(X[highconf_mask][inlier_mask], y[highconf_mask][inlier_mask])
        # fity = [ransacy.estimator_.coef_[0], ransacy.estimator_.intercept_]
        
        # # save to raw collection
        # if filter:
        #     filter = length*[1]
        #     for i in bad_idx:         
        #         filter[i]=0
    
        
        # ----- no ransac outlier masks
        ransacy = linear_model.RANSACRegressor(residual_threshold=residual_threshold_y)
        ransacy.fit(X[highconf_mask], y[highconf_mask])
        fity = [ransacy.estimator_.coef_[0], ransacy.estimator_.intercept_]
        filter = 1*highconf_mask

        # save filter to database- non-blocking
        # _id = traj["_id"]
        # thread = threading.Thread(target=thread_update_one, args=(raw, _id, filter, fitx, fity,))
        # thread.start()
    
        # update traj document
        traj["filter"] = list(filter)
        traj["fitx"] = list(fitx)
        traj["fity"] = list(fity)
    
    return traj



@catch_critical(errors = (Exception))
def calc_fit(traj, residual_threshold_x, residual_threshold_y):
    '''
    add a filter to trajectories based on
    - RANSAC fit on x and
    - bad detection confidence
    get total mask (both lowconf and outlier)
    apply ransac again on y-axis
    save fitx, fity and tot_mask
    and save filter field to raw collection
    '''
    
    t = np.array(traj["timestamp"])
    x = np.array(traj["x_position"])
    y = np.array(traj["y_position"])
        
    ransacx = linear_model.RANSACRegressor(residual_threshold=residual_threshold_x)
    X = t.reshape(1, -1).T
    ransacx.fit(X, x)
    fitx = [ransacx.estimator_.coef_[0], ransacx.estimator_.intercept_]
      
    ransacy = linear_model.RANSACRegressor(residual_threshold=residual_threshold_y)
    ransacy.fit(X, y)
    fity = [ransacy.estimator_.coef_[0], ransacy.estimator_.intercept_]
    
    # update traj document
    traj["fitx"] = list(fitx)
    traj["fity"] = list(fity)

    return traj

def calc_fit_select(t,x,y,residual_threshold_x, residual_threshold_y):
    '''
    same as calc_fit, but only on given t,x,y
    '''
    ransacx = linear_model.RANSACRegressor(residual_threshold=residual_threshold_x)
    X = t.reshape(1, -1).T
    ransacx.fit(X, x)
    fitx = [ransacx.estimator_.coef_[0], ransacx.estimator_.intercept_]
      
    ransacy = linear_model.RANSACRegressor(residual_threshold=residual_threshold_y)
    ransacy.fit(X, y)
    fity = [ransacy.estimator_.coef_[0], ransacy.estimator_.intercept_]
    
    return fitx, fity


def find_overlap_idx(x, y):
    '''
    x,y are timestamp arrays
    y ends before x
    find the intervals for x and y overlap, i.e.,
    x[s1: e1] overlaps with y[s2, e2]
    '''
    s1,s2=0,0
    # find starting pointers
    while s1 < len(x) and s2 < len(y):
        if abs(x[s1] - y[s2]) < 1e-3:
            break
        elif x[s1] < y[s2]:
            s1 += 1
        else:
            s2 += 1
    # find ending poitners
    e1, e2 = len(x)-1, len(y)-1
    while e1 >0 and e2 >0:
        if abs(x[e1] - y[e2]) < 1e-3:
            break
        if x[e1] < y[e2]:
            e2 -= 1
        else:
            e1 -= 1
            
    return s1, e1, s2, e2


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
        
    
    
    

      