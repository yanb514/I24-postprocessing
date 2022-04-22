#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:57:02 2022

@author: wangy79
"""
from utils.data_structures import PathCache
import numpy as np
import unittest


class T(unittest.TestCase):

    # creating test data
    docs = []
    ids = list("abcdefghij")
    last_timestamps = np.arange(len(ids))
    
    for i in range(len(ids)):
        docs.append({
            "_id": ids[i],
            "tail_time": last_timestamps[i],
            "last_timestamp": last_timestamps[i]})
    
    # initialize `DisjointSet` class
    pc = PathCache() 
    # create a singleton set for each element of the universe
    pc.make_set(docs)

    pc.union("a", "c") 
    pc.union("c", "f") 
    pc.union("h", "j") 
    pc.union("d", "j") 
    pc.union("b", "i")
    pc.union("e", "i")

    # store ground truth values
    real_roots = list("abadbagdbd")
    paths = [list("acf"), list("g"),list("bei"),list("dhj")]
    roots = [l[0] for l in paths]


    
    def test_all_roots(self):
        for i, node in enumerate(self.pc.path.values()):
            self.assertEqual(node.root.id, self.real_roots[i], "node {}'s root is incorrect".format(node.id))
            
            
    def test_final_paths(self):
        # number of roots
        self.assertEqual(self.pc.count(), len(self.roots), "Should be {} roots".format(len(self.roots)))
        # number of paths
        paths = self.pc.get_all_paths()
        self.assertEqual(len(paths), len(self.roots), "Should be {} roots".format(len(self.roots)))

        for i, path in enumerate(paths):
            self.assertEqual(path, self.paths[i], "{}th path is incorrect".format(i))

    def test_path_order(self):
        roots = self.pc.get_all_roots(attr_name = "id")
        self.assertEqual(list(roots), self.roots, "Path order incorrect")
        
    def test_pop_path(self):
        self.pc.pop_first_path()
        self.assertEqual(self.pc.count(), len(self.roots)-1, "first path incorrect")
        paths = self.pc.get_all_paths()
        self.assertEqual(paths, self.paths[1:], "first path incorrect")
        self.assertEqual(len(self.pc.path), len(self.docs)-len(self.paths[0]), "left over fragments not correct")
        



    
if __name__ == '__main__':
    unittest.main()

     
    

        

 
