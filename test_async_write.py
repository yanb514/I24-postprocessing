from db_writer import DBWriter
import db_parameters
import time
import asyncio

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

#%% Test change stream
# Can only run on VM (where MongoDB is set up)
from pymongo import MongoClient
import pymongo

client = MongoClient(host=['localhost:27017'])
db=client["trajectories"]
col=db["test_change_stream"]

col.insert_one({"x":1})
print(col.count_documents({}))

#%%
try:
    resume_token = None
    # pipeline = [{'$match': {'operationType': 'insert'}}]
    with db.collection.watch() as stream:
        for insert_change in stream:
            print(insert_change)
            resume_token = stream.resume_token
except pymongo.errors.PyMongoError:
    # The ChangeStream encountered an unrecoverable error or the
    # resume attempt failed to recreate the cursor.
    if resume_token is None:
        # There is no usable resume token because there was a
        # failure during ChangeStream initialization.
        logging.error('...')
    else:
        # Use the interrupted ChangeStream's resume token to create
        # a new ChangeStream. The new stream will continue from the
        # last seen insert change without missing any events.
        with db.collection.watch(
                pipeline, resume_after=resume_token) as stream:
            for insert_change in stream:
                print(insert_change)