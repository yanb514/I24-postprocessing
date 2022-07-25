            



            
        
    def delete_collection(self, collection_list):
        '''
        delete (reset) collections in list
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dbw = DBWriter(self.config, collection_name = "none", schema_file=None)
            all_collections = dbw.db.list_collection_names()
            
            for col in collection_list:
                if col not in all_collections:
                    print(f"{col} not in collection list")
                # dbw.reset_collection() # This line throws OperationFailure, not sure how to fix it
                else:
                    dbw.db[col].drop()
                    if col not in dbw.db.list_collection_names():
                        print("Collection {} is successfully deleted".format(col))
                    
        
    def get_random(self, collection_name):
        '''
        Return a random document from a collection
        '''
        dbr = DBReader(self.config, collection_name=collection_name)
        import random
        doc = dbr.collection.find()[random.randrange(dbr.count())]
        return doc
    
        
    
    def plot_fragments(self, traj_ids):
        '''
        Plot fragments with the reconciled trajectory (if col2 is specified)

        Parameters
        ----------
        fragment_list : list of ObjectID fragment _ids
        rec_id: ObjectID (optional)
        '''
        
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        
        
        for f_id in traj_ids:
            f = self.dbr_v.find_one("_id", f_id)
            axs[0].scatter(f["timestamp"], f["x_position"], s=0.5, label=f_id)
            axs[1].scatter(f["timestamp"], f["y_position"], s=0.5, label=f_id)
            axs[2].scatter(f["x_position"], f["y_position"], s=0.5, label=f_id)

        axs[0].set_title("time v x")
        axs[1].set_title("time v y")
        axs[2].set_title("x v y")
            
        axs[0].legend()
        
    def fragment_length_dist(self):
        '''
        Get the distribution for the #fragments that matched to one trajectory
        '''
        if self.col2 is None:
            print("Collection 2 must be specified")
            return
        pipeline = [{'$project':{ 'count': { '$size':'$fragment_ids'} } }, 
                       { '$group' : {'_id':'$count', 'count':{'$sum':1} } },
                       { '$sort'  : {'count': -1 } } ]
        cur = self.col2.collection.aggregate(pipeline)
        pprint.pprint(list(cur))

    
    def evaluate_old(self):
        '''
        1. filtered fragments (not in reconciled fragment_ids)
        2. lengths of those fragments (should be <3) TODO: Some long documents are not matched
        3. 
        '''
        
        # find all unmatched fragments
        res = self.dbr_v.collection.aggregate([
               {
                  '$lookup':
                     {
                       'from': self.col2_name,
                       'localField': "_id",
                       'foreignField': "fragment_ids",
                        'pipeline': [
                            { '$project': { 'count': 1}  }, # dummy
                        ],
                       'as': "matched"
                     }
                },
                {
                 '$match': {
                     "matched": { '$eq': [] } # select all unmatched fragments
                   }
                }
            ] )
        self.res = list(res)
        print("{} out of {} fragments are not in reconciled ".format(len(self.res), self.col1.count()))
        
        # Get the length distribution of these unmatched fragments
        import math
        f_ids = [d['_id'] for d in self.res] # get all the ids
        pipeline = [{'$project':{ 'length': { '$size':'$timestamp'} } },
                    { '$match': { "_id": {'$in': f_ids } } },
                       { '$group' : {'_id':'$length', 'count':{'$sum':1}} },
                       { '$sort'  : {'count': -1 } } ]
        cur = self.dbr_v.collection.aggregate(pipeline)
        dict = {d["_id"]: math.log(d["count"]) for d in cur}
        plt.bar(dict.keys(), dict.values(), width = 2, color='g')
        plt.xlabel("Lengths of documents")
        plt.ylabel("Count (log-scale)")
        plt.title("Unmatched fragments length distribution in log")
        
        # pprint.pprint(list(cur))
        
    
        