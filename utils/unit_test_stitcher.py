#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:57:02 2022

@author: wangy79
"""
from data_structures import PathCache
import numpy as np
import unittest


class TestPathCache(unittest.TestCase):

    # creating test data
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
    pc._makeSet(docs)

    pc._union("a", "f") 
#    pc._union("b", "e") 
#    pc._union("e", "f") 
#    pc._union("c", "d") 
#    pc._union("d", "f") 

    # store ground truth values
    real_roots = list("abcdea")
    num_paths = 5
    paths = [list("b"),list("c"),list("d"),list("e"),list("af")]
    roots = list("bcdea")

    
    def test_all_roots(self):
        for i, node in enumerate(self.pc.path.values()):
            self.assertEqual(node.root.id, self.real_roots[i], "node {}'s root is incorrect".format(node.id))
            
    def test_final_paths(self):
        # number of roots
        self.assertEqual(len(self.pc._getAllRoots()), self.num_paths, "Should be {} roots".format(self.num_paths))
        # number of paths
        paths = self.pc._getAllPaths()
        self.assertEqual(len(paths), self.num_paths, "Should be {} roots".format(self.num_paths))

        for i, path in enumerate(paths):
            self.assertEqual(path, self.paths[i], "{}th path is incorrect".format(i))

    def test_path_order(self):
        roots = self.pc.cache.keys()
        self.assertEqual(list(roots), self.roots, "Path order incorrect")
        
    def test_modified_time(self):
        pass


    
    
if __name__ == '__main__':
    
    unittest.main()
     
    

        

 
