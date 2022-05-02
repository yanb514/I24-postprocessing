from db_writer import DBWriter
import db_parameters
import time


# %% Test Motor connection and write Done
dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)


# Test motor asynic write
def restart_col_motor():
    col = dbw.motor_db["test_collection"]
    col.drop()
    col = dbw.motor_db["test_collection"]
    return col
    
def restart_col_pymongo():
    col = dbw.db["test_collection"]
    col.drop()
    col = dbw.db["test_collection"]
    return col
    

async def one_iteration(doc):
    await col.insert_one(doc)
  
async def other_stuff():
    await asyncio.sleep(0.001)

def other_stuff_sync():
    time.sleep(0.001)

# plain vanilla insert_one
def f0(N, col):
    t1 = time.time()

    for i in range(N):
        # other_stuff_sync()
        col.insert_one({str(i): i})
    count = col.count_documents({})
    t2 = time.time()
    print("Final count: %d" % count)
    print("f0 bulk write takes {:.2f} sec".format(t2-t1))
    return t2-t1

# async insert_one
async def f1(N, col):
    t1 = time.time()
    for i in range(N):
        # await other_stuff()
        await one_iteration({str(i): i})
    count = await col.count_documents({})
    t2 = time.time()
    print("Final count: %d" % count)
    print("f1 bulk write takes {:.2f} sec".format(t2-t1))
    # await restart_col_motor()
    return t2-t1

# parallel insert_one
async def f2(N, col):
    t1 = time.time()
    coros = []
    for i in range(N):
        # await other_stuff()
        coros.append(one_iteration({str(i): i}))
    await asyncio.gather(*coros)   # all coroutines in parallel
    count = await col.count_documents({})
    t2 = time.time()
    # await restart_col_motor()
    print("Final count: %d" % count)
    print("f2 bulk write takes {:.2f} sec".format(t2-t1))
    return t2-t1
  
# %% Benchmark insert time
Ns = [10,100,200,300,400,500,600,700,800,900,1000] 
t0_arr = []
for N in Ns:
    col = restart_col_pymongo()
    t = f0(N, col)
    t0_arr.append(t)
    
#%% Test async insert
col = restart_col_motor()
asyncio.create_task(f2(10000, col))

    
# try:
#     asyncio.run(f1(N)) # if event loop is already running (e.g., Spyder IDE)
# except RuntimeError:
#     asyncio.create_task(f1(N))
# loop = asyncio.get_event_loop()
# asyncio.run(f2(N))
# loop.create_task(f1(N))

# try:
#     asyncio.run(f2(N)) # if event loop is already running (e.g., Spyder IDE)
# except RuntimeError:
#     asyncio.create_task(f2(N))

#%% Plot insert time
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(Ns, t0_arr, label = "sync insert_one")
t1_arr = [0.33, 1.97, 4.03, 5.78, 7.47, 9.47, 12.34, 14.09, 15.96, 18.82,19.94]
plt.scatter(Ns, t1_arr, label = "async insert_one")
t2_arr = [0.23, 0.37, 0.29, 0.58, 0.50, 0.69, 0.45, 0.82, 0.80,0.86,0.77]
plt.scatter(Ns, t2_arr, label = "parallel insert_one")
plt.xlabel("num documents")
plt.ylabel("total insert time (sec)")
plt.legend()

# %% Test Pymongo bulk write
col = dbw.db["test_collection"]
col.drop()
col = dbw.db["test_collection"]

t1 = time.time()
# for i in range(N):
#     col.insert_one({'i': i})
col.insert_many(({'i': i} for i in range(N)))
count = col.count_documents({})
t2 = time.time()
print("Final count: %d" % count)
print("Pymongo bulk write takes {:.2f} sec".format(t2-t1))


# %% pymongo insert multithread
from pymongo import MongoClient
from threading import Thread
import time

THREAD_COUNT = 10000

# Derive from Threading.thread to create a specialised insert thread
class DataInsertThread(Thread):
    database        = None
    threadNumber    = None

    def __init__(self, database_in, threadNumber):
        self.database       = database_in
        self.threadNumber   = threadNumber
        Thread.__init__(self)

    def run(self):
        self.database["test_collection"].insert_one({"key":1})


# Get a MongoClient instance
dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)
dbw.client.maxPoolSize = 400
dbw.client.waitQueueTimeoutMS = 200
dbw.client.waitQueueMultiple = 500


# Get the database object                            
databaseObject  = dbw.db
insertThreads   = []
col = dbw.db["test_collection"]
col.drop()
 
t1 = time.time()
# Create insert threads
for threadNum in range(THREAD_COUNT):
    insertThread   = DataInsertThread(databaseObject, threadNum)
    insertThreads.append(insertThread)

    # Start the insert thread
    insertThread.start()

# Wait till all the insert threads are complete
# for insertThread in insertThreads:
#     insertThread.join()
    
t2 = time.time()
t_multithread = t2-t1

col = dbw.db["test_collection"]

time.sleep(5)

print("Final docs: {}, time: {}".format(col.count_documents({}), t2-t1))

#%% not inherit from thread class
col.drop()

def insert_to_col(collection, document):
    # while True:
    collection.insert_one(document)

t1 = time.time()
threads = []
for threadNum in range(THREAD_COUNT):
    thread = Thread(target=insert_to_col, args=(col, {"key": threadNum}))
    thread.setDaemon(True)
    threads.append(thread)
    thread.start()
    
# for thread in threads:
#     thread.join() 
# while True:
#     pass
t2 = time.time()
time.sleep(5)
print("Final docs: {}, time: {}".format(col.count_documents({}), t2-t1))

#%%
# example from https://docs.python.org/2/library/queue.html
import queue
from threading import Thread
import time
import random

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
num_workers = 1
num_tasks = 1000



while True: # non-stop write a batch of num_tasks size and pause for 1 sec
    t1 = time.time()
    for i in range(num_workers):
         t = Thread(target=insert_to_col, args=(col, ))
         t.daemon = True 
         t.start()

    for i in range(num_tasks):
        q.put({"key": i, 'rand': [random.random() for j in range(40)],
               'message': 'a message '*100})
    # print(q.qsize())

    q.join()       # block until all tasks are done
    t2 = time.time()
    time.sleep(1)
    
    print("Final docs: {}, time: {}".format(col.count_documents({}), t2-t1))

#%% with multiple threads
import queue
from threading import Thread
import time
import random
import math

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
num_tasks = 1000

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
        q.put({"key": i, 'rand': [random.random() for j in range(40)],
               'message': 'a message '*100})

    # q.join()       # block until all tasks are done
    t2 = time.time()
    
    print("Final docs: {}, other proc: {:.2f}, insertion time: {:.2f}".format(col.count_documents({}), t1-t0, t2-t1))
    
    
#%% insert_one at a time (no thread)

import queue
import time
import random
import math

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

import queue
from threading import Thread
import time
import random
import math

dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)
 
# dbw.client.maxPoolSize = 200                         
col = dbw.db["test_collection"]
col.drop()  
        
def thread_insert(collection, document):
    collection.insert_one(document)
    
    
q = queue.Queue()
num_workers = 1
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
            

# %% test dbw

dbw = DBWriter(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, server_id=1, process_name=1, process_id=1, session_config_id=1)
 
# dbw.client.maxPoolSize = 200                         
col = dbw.db["test_collection"]
col.drop()  

    
    
    
    
    
    
