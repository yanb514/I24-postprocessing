from db_reader import DBReader
import db_parameters

# %% Test connection
dbr = DBReader(host=db_parameters.DEFAULT_HOST, port=db_parameters.DEFAULT_PORT, username=db_parameters.DEFAULT_USERNAME,   
               password=db_parameters.DEFAULT_PASSWORD,
               database_name=db_parameters.DB_NAME, collection_name=db_parameters.RAW_COLLECTION)



# %% Test read_query
dbr.create_index(["last_timestamp", "first_timestamp", "starting_x", "ending_x"])
res = dbr.read_query(query_filter = {"last_timestamp": {"$gt": 5, "$lt":309}}, query_sort = [("last_timestamp", "ASC"), ("starting_x", "ASC")],
                   limit = 0)

for doc in res:
    print("last timestamp: {:.2f}, starting_x: {:.2f}, ID: {}".format(doc["last_timestamp"], doc["starting_x"], doc["ID"]))


#%% Test read_query_range (no range_increment)
print("Using while-loop to read range")
rri = dbr.read_query_range(range_parameter='last_timestamp', range_greater_equal=300, range_less_than=330, range_increment=None)
while True:
    try:
        print(next(rri)["ID"]) # access documents in rri one by one
    except StopIteration:
        print("END OF ITERATION")
        break

print("Using for-loop to read range")
for result in dbr.read_query_range(range_parameter='last_timestamp', range_greater_equal=300, range_less_than=330, range_increment=None):
    print(result["ID"])
print("END OF ITERATION")


#%% Test read_query_range (with range_increment)
print("Using while-loop to read range")
rri = dbr.read_query_range(range_parameter='last_timestamp', range_greater_equal=300, range_less_than=330, range_increment=10)
while True:
    try:
        for doc in next(rri): # next(rri) is a cursor
            print(doc["ID"])
        print("read next range")
    except StopIteration:
        print("END OF ITERATION")
        break

print("Using for-loop to read range")
for interval in dbr.read_query_range(range_parameter='last_timestamp', range_greater_equal=300, range_less_than=330,range_increment=10):
    print("next interval ")
    for doc in interval:
        print(doc["ID"])
print("END OF ITERATION")