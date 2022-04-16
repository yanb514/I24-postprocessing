#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:57:02 2022

@author: wangy79
"""
from data_structures import PathCache
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
            "last_timestamp": last_timestamps[i]})
    
    # initialize `DisjointSet` class
    pc = PathCache() 
    # create a singleton set for each element of the universe
    pc._makeSet(docs)

    pc._union("a", "c") 
    pc._union("c", "f") 
    pc._union("h", "j") 
    pc._union("d", "j") 
    pc._union("b", "i")
    pc._union("e", "i")

    # store ground truth values
    real_roots = list("abadbagdbd")
    paths = [list("g"), list("acf"),list("dhj"),list("bei")]
    roots = [l[0] for l in paths]


    
    def test_all_roots(self):
        for i, node in enumerate(self.pc.path.values()):
            self.assertEqual(node.root.id, self.real_roots[i], "node {}'s root is incorrect".format(node.id))
            
    def test_final_paths(self):
        # number of roots
        self.assertEqual(len(self.pc._getAllRoots()), len(self.roots), "Should be {} roots".format(len(self.roots)))
        # number of paths
        paths = self.pc._getAllPaths()
        print(paths)
        print(self.paths)
        self.assertEqual(len(paths), len(self.roots), "Should be {} roots".format(len(self.roots)))

        for i, path in enumerate(paths):
            self.assertEqual(path, self.paths[i], "{}th path is incorrect".format(i))

    def test_path_order(self):
        roots = self.pc.cache.keys()
        self.assertEqual(list(roots), self.roots, "Path order incorrect")
        
    def test_modified_time(self):
        pass


    
    
if __name__ == '__main__':
    
    unittest.main()
     
    

        

 
