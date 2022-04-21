#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:57:02 2022

@author: wangy79
"""
from data_structures import PathCache, SortedDLL, Fragment
import numpy as np
import unittest


class T(unittest.TestCase):

    # creating test data
    docs = []
    ids = list("abcde")
    last_timestamps = np.arange(len(ids))
    
    for i in range(len(ids)):
        docs.append({
            "id": ids[i],
            "tail_time": last_timestamps[i]})
    
    # initialize
    dll = SortedDLL()
    # insert some nodes
    for doc in docs:
        dll.append(doc)
       
    true_count = len(docs)

    
    def test_append(self):
        # [a,b,c,d,e]
        # [0,1,2,3,4]
        first_node = self.dll.first_node() 
        self.assertEqual(first_node.id, "a")
        self.assertEqual(self.dll.count(), self.true_count)
        
    def test_update(self):
        self.dll.update("a", 2, "tail_time") 
        # [b,c,a,d,e]
        # [1,2,2,3,4]
        self.assertEqual(self.dll.get_attr("id"), list("bcade"), "DLL is not in the correct order after update")
        self.assertEqual(self.dll.get_attr("tail_time"), [1,2,2,3,4], "DLL is not in the correct order after update")
        
        self.dll.update("d", 5, "tail_time") 
        # [b,c,a,e,d]
        # [1,2,2,4,5]
        self.assertEqual(self.dll.get_attr("id"), list("bcaed"), "DLL is not in the correct order after update")
        self.assertEqual(self.dll.get_attr("tail_time"), [1,2,2,4,5], "DLL is not in the correct order after update")
        
        self.dll.update("b", 1, "tail_time") 
        # [b,c,a,e,d]
        # [1,2,2,4,5]
        self.assertEqual(self.dll.get_attr("id"), list("bcaed"), "DLL is not in the correct order after update")
        self.assertEqual(self.dll.get_attr("tail_time"), [1,2,2,4,5], "DLL is not in the correct order after update")
        
        
        self.dll.update("d", 3, "tail_time") 
        # [b,c,a,d,e]
        # [1,2,2,3,4]
        self.assertEqual(self.dll.get_attr("id"), list("bcade"), "DLL is not in the correct order after update")
        self.assertEqual(self.dll.get_attr("tail_time"), [1,2,2,3,4], "DLL is not in the correct order after update")
        
        self.dll.update("e", 5, "tail_time") 
        # [b,c,a,d,e]
        # [1,2,2,3,5]
        self.assertEqual(self.dll.get_attr("id"), list("bcade"), "DLL is not in the correct order after update")
        self.assertEqual(self.dll.get_attr("tail_time"), [1,2,2,3,5], "DLL is not in the correct order after update")
        
        self.dll.update("b", 0.5, "tail_time") 
        # [b,c,a,d,e]
        # [0.5,2,2,3,5]
        self.assertEqual(self.dll.get_attr("id"), list("bcade"), "DLL is not in the correct order after update")
        self.assertEqual(self.dll.get_attr("tail_time"), [0.5,2,2,3,5], "DLL is not in the correct order after update")
        
        self.dll.update("a", 1, "tail_time") 
        # [b,a,c,d,e]
        # [0.5,1,2,3,5]
        self.assertEqual(self.dll.get_attr("id"), list("bacde"), "DLL is not in the correct order after update")
        self.assertEqual(self.dll.get_attr("tail_time"), [0.5,1,2,3,5], "DLL is not in the correct order after update")
      
    # def test_delete(self):
        self.dll.delete("a") 
        # [b,c,d,e]
        # [0.5,2,3,5]
        self.assertEqual(self.dll.get_attr("id"), list("bcde"), "DLL is not in the correct order after update")
        self.assertEqual(self.dll.get_attr("tail_time"), [0.5,2,3,5], "DLL is not in the correct order after update")
        




    
if __name__ == '__main__':
    unittest.main()

     
    

        

 
