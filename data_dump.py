from i24_database_api import DBClient
import os
import json
from bson.json_util import dumps
import sys



def decimal_range(start, stop, increment):
    while start < stop: # and not math.isclose(start, stop): Py>3.5
        yield start
        start += increment
        

def main(database_name="", collection_name="", chunk_size = 1800):
    
    with open(os.environ["USER_CONFIG_DIRECTORY"]+"/db_param.json") as f:
        db_param = json.load(f)
     
    # chunk_size = 100 # 1800=30min
    dbc = DBClient(**db_param, database_name=database_name, collection_name = collection_name)
    
    start = dbc.collection.find_one(sort=[("first_timestamp", 1)])["first_timestamp"]
    end = dbc.collection.find_one(sort=[("last_timestamp", -1)])["first_timestamp"]
    # end = start+200
    
    
    # specify query - iterative ranges
    idx = 0
    for s in decimal_range(start, end, chunk_size):
        print("In progress (approx) {:.1f} %".format((s-start)/(end-start)*100))
        cursor = dbc.collection.find({"first_timestamp": {"$gte": s, "$lt": s+chunk_size}}).sort("first_timestamp",1)
        
        with open('{}_{:02d}.json'.format(collection_name, idx), 'w') as file: 
            json.dump(json.loads(dumps(cursor)), file)
            
        idx += 1
        
    print("Completed writing {} json files".format(idx))
        
        
        
        
        
        
        
if __name__ == '__main__':
    
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
        
    else:
        print("invalid input")