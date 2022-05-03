from db_writer import DBWriter
import db_parameters
from threading import Thread
import time
import queue
import random
import math

#%% with multiple threads in queue

dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)
 
# dbw.client.maxPoolSize = 200                         
col = dbw.db["test_collection"]
col.drop()  
     
def insert_to_col(collection):
    while True:
        document = q.get(True)
        collection.insert_one(document)
        q.task_done() # an indicator that this thread/worker is ready for the next task
          
q = queue.Queue()
num_workers = 100
num_tasks = 100

while True: 
    # simulate a CPU-intensive task for 1 sec
    t0 = time.time()
    startTime = time.time()
    while time.time() - startTime < 1:
        math.factorial(100) 
        
    t1 = time.time()
    for i in range(num_workers):
         t = Thread(target=insert_to_col, args=(col, ))
         t.daemon = True 
         t.start()

    for i in range(num_tasks):
        q.put({"key": i, 'rand': [random.random() for j in range(9000)],
               'message': 'a message '*100})

    # q.join()       # block until all tasks are done
    t2 = time.time()
    
    print("Final docs: {}, other proc: {:.2f}, insertion time: {:.2f}".format(col.count_documents({}), t1-t0, t2-t1))
    
    
#%% insert_one at a time (no thread)

dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)
 
# dbw.client.maxPoolSize = 200                         
col = dbw.db["test_collection"]
col.drop()  
num_tasks = 100

while True: 
    # simulate a CPU-intensive task for 1 sec
    t0 = time.time()
    startTime = time.time()
    while time.time() - startTime < 1:
        math.factorial(100) 
        
    t1 = time.time()

    for i in range(num_tasks):
        col.insert_one({"key": i, 'rand': [random.random() for j in range(40)],
               'message': 'a message '*100})
    t2 = time.time()
    
    print("Final docs: {}, other proc: {:.2f}, insertion time: {:.2f}".format(col.count_documents({}), t1-t0, t2-t1))
   
#%% one thread worker
dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)
                        
col = dbw.db["test_collection"]
col.drop()  
        
def thread_insert(collection, document):
    collection.insert_one(document) 
    
num_tasks = 100

while True: 
    # simulate a CPU-intensive task for 1 sec
    t0 = time.time()
    startTime = time.time()
    while time.time() - startTime < 3:
        math.factorial(100) 
    
    t1 = time.time()
    for i in range(num_tasks):
        doc = {"key": i, 'rand': [random.random() for j in range(40)],
               'message': 'a message '*100}
        t = Thread(target=thread_insert, args=(col, doc))
        t.daemon = True
        t.start()

    # q.join()       # block until all tasks are done
    t2 = time.time()
    
    print("Final docs: {}, other proc: {:.2f}, time to start all threads: {:.2f}".format(col.count_documents({}), t1-t0, t2-t1))


#%% Test multithread write on dbw

dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)

collection_name = "test_collection"
dbw.db[collection_name].drop()
num_tasks = 100

while True:
    
    startTime = time.time()
    
    while time.time() - startTime < 2:
        math.factorial(100)
    
    t1 = time.time()
    for i in range(num_tasks):
        dbw.write_trajectory(thread=True, collection_name = collection_name,
                        timestamp = [random.random() for j in range(9000)],
                        flags = "very normal stuff")
        
    t2 = time.time()
    print("Final docs: {}, time to start all threads: {:.2f}, each iter:{:.2f}".format(dbw.db[collection_name].count_documents({}),t2-t1, t2-startTime))
    
    
    
# %% compare multhread with insert_many()

# %% insert time vs. doc length and numbers
    
def foo(a,b):
    print(a,b)
args = (1,2)
foo(*args)