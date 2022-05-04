import pymongo
import db_parameters
import csv
import urllib.parse
from threading import Thread


def write_data_from_csv(**kwargs):
    '''
    Write trajectories from csv files to database
    First check if csv data is sorted either by ID or by timestamp (frame)
    '''
    # connect to DBWriter
    dbw = DBWriter(**kwargs)

    order_by_time = input("Are the rows ordered by time [Y/N]?")
    order_by_ID = input("Are the rows ordered by ID [Y/N]?")
    
    # TODO: if kwargs is passed on correctly?
    if order_by_ID == "Y":
        write_csv_to_db_by_ID(dbw, *kwargs)
        
    elif order_by_time == "Y":
        write_csv_to_db_by_time(dbw, **kwargs)
    
    else:
        raise NotImplementedError("Currently do not support reading csv files that are not ordered by time or ID")

def write_csv_to_db_by_ID(dbwriter, collection_name, file_location, file_name, x_range, idle_time = None, verbose = True):
    '''
    Read csv file row by row, write to db if ID switches
    Currently no schema enforcement 
    Needs to manually check the column names
    TODO: not tested
    '''
    if collection_name in dbwriter.db:
        col = dbwriter.db[collection_name]
        rewrite = input("Re-write to existing collection called {} [Y/N]?".format(collection_name))
    if rewrite == "Y":
        X_MIN, X_MAX = x_range
        col.drop()
        col = dbwriter.db[collection_name]
        
        prevID = -1 # if curr_ID!= prevID, write to database 
        traj = {} # to start
        traj['timestamp'] = []
        traj['raw_timestamp'] = []
        traj['road_segment_id'] = []
        traj['x_position'] = []
        traj['y_position'] = []
        traj['configuration_id']=1
        traj['local_fragment_id']=1
        traj['compute_node_id']=1
        traj['coarse_vehicle_class']= 0
        traj['fine_vehicle_class']=1
                        
                        
        for file in file_name:
            print("In file {}".format(file))
            line = 0
            with open (file_location+file,'r') as f:
                reader=csv.reader(f)
                next(reader) # skip the header
                
                for row in reader:
                    line += 1
                    ID = int(float(row[4]))
                    curr_time = float(row[3])
                    curr_x = float(row[41])
                    
                    if curr_x > X_MAX or curr_x < X_MIN:
                        continue
                    
                    if line % 10000 == 0 and verbose:
                        print("line: {}, curr_time: {:.2f}, x:{:.2f},  gtID: {} ".format(line, curr_time, curr_x, ID))
                    
                    if ID!=prevID and prevID!=-1: 
                        # write prevID to database
                        traj['db_write_timestamp'] = 0
                        traj['first_timestamp']=traj['timestamp'][0]
                        traj['last_timestamp']=traj['timestamp'][-1]
                        traj['starting_x']=traj['x_position'][0]
                        traj['ending_x']=traj['x_position'][-1]
                        traj['flags'] = ['gt']
                        traj['ID']=prevID
                        
        #                print("** write {} to db".format(ID))
                        col.insert_one(traj)
                        
                        traj = {} # create a new trajectory for the next iteration
                        traj['configuration_id']=1
                        traj['local_fragment_id']=1
                        traj['compute_node_id']=1
                        traj['coarse_vehicle_class']=int(row[5])
                        traj['fine_vehicle_class']=1
                        traj['timestamp']=[float(row[3])]
                        traj['raw_timestamp']=[float(1.0)]
                        traj['road_segment_id']=[int(row[49])]
                        traj['x_position']=[3.2808*float(row[41])]
                        traj['y_position']=[3.2808*float(row[42])]
                        traj['direction']=int(float(row[37]))
                        traj['ID']=ID
                    
                    else: #  keep augmenting trajectory
                        traj['timestamp'].extend([float(row[3])])
                        traj['raw_timestamp'].extend([float(1.0)])
                        traj['road_segment_id'].extend([float(row[49])])
                        traj['x_position'].extend([3.2808*float(row[41])])
                        traj['y_position'].extend([3.2808*float(row[42])])
                        traj['length']=[3.2808*float(row[45])]
                        traj['width']=[3.2808*float(row[44])]
                        traj['height']=[3.2808*float(row[46])]     
                        
                    prevID = ID
                    
            f.close()
    
        # flush out the last trajectory in cache
        print("writing the last trajectory")
        traj['db_write_timestamp'] = 0
        traj['first_timestamp']=traj['timestamp'][0]
        traj['last_timestamp']=traj['timestamp'][-1]
        traj['starting_x']=traj['x_position'][0]
        traj['ending_x']=traj['x_position'][-1]
        traj['flags'] = ['gt']
        traj['direction']=int(float(row[37]))
        traj['ID']=ID
        traj['length']=[3.2808*float(row[45])]
        traj['width']=[3.2808*float(row[44])]
        traj['height']=[3.2808*float(row[46])]

        print("** write {} to db".format(ID))
        col.insert_one(traj)


        # finally, add fragment_ids to ground truth
        # print("Adding fragment IDs")
        # colraw = db["raw_trajectories_one"]
        
        # for rawdoc in colraw.find({}):
        #     _id = rawdoc.get('_id')
        #     raw_ID=rawdoc.get('ID')
        #     gt_ID=raw_ID//100000
        #     if colgt.count_documents({ 'ID': gt_ID }, limit = 1) != 0: # if gt_ID exists in colgt
        #         # update
        #         colgt.update_one({'ID':gt_ID},{'$push':{'fragment_ids':_id}},upsert=True)

    return

def write_csv_to_db_by_time(dbwriter, collection_name, file_location, file_name, x_range, idle_time=1, verbose=True):
    '''
    Read csv file row by row, keep in a lru cache. Write to db if the first item in lru (least recently used) idles more than idle_time
    Currently no schema enforcement
    '''
    
    if collection_name in dbwriter.db:
        col = dbwriter.db[collection_name]
        rewrite = input("Re-write to existing collection called {} [Y/N]?".format(collection_name))
    if rewrite == "Y":
        from collections import OrderedDict
        lru = OrderedDict()
        X_MIN, X_MAX = x_range
        col.drop()
        col = dbwriter.db[collection_name]

        for file in file_name:
            print("In file {}".format(file))
            line = 0
            with open(file_location+file,'r') as f:
                reader = csv.reader(f)
                next(reader) # skip the header
                
                for row in reader:
                    line += 1
                    ID = int(float(row[3]))
                    curr_time = float(row[2])
                    curr_x = float(row[40])
                    if curr_x > X_MAX or curr_x < X_MIN:
                        continue
                    
                    if verbose and line % 10000 == 0:
                        print("line: {}, curr_time: {:.2f}, x:{:.2f},  lru size: {} ".format(line, curr_time, curr_x, len(lru)))
        #                break
                    
                    if ID not in lru: # create new
                        traj = {}
                        traj['configuration_id']=1
                        traj['local_fragment_id']=1
                        traj['compute_node_id']=1
                        traj['coarse_vehicle_class']=int(row[4])
                        traj['fine_vehicle_class']=1
                        traj['timestamp']=[float(row[2])]
                        traj['raw_timestamp']=[float(1.0)]
                        
                        traj['road_segment_id']=[int(row[48])]
                        traj['x_position']=[3.2808*float(row[40])]
                        traj['y_position']=[3.2808*float(row[41])]
                        
                        traj['length']=[3.2808*float(row[44])]
                        traj['width']=[3.2808*float(row[43])]
                        traj['height']=[3.2808*float(row[45])]
                        traj['direction']=int(float(row[36]))
                        traj['ID']=float(row[3])
                        
                        lru[ID] = traj
                        
                    else:
                        traj = lru[ID]
                        traj['timestamp'].extend([float(row[2])])
                        traj['raw_timestamp'].extend([float(1.0)])
                
                        traj['road_segment_id'].extend([float(row[48])])
                        traj['x_position'].extend([3.2808*float(row[40])])
                        traj['y_position'].extend([3.2808*float(row[41])])
                        
                        traj['length'].extend([3.2808*float(row[44])])
                        traj['width'].extend([3.2808*float(row[43])])
                        traj['height'].extend([3.2808*float(row[45])])
                        
                        lru.move_to_end(ID)
                        
                    while lru[next(iter(lru))]["timestamp"][-1] < curr_time - idle_time:
                        ID, traj = lru.popitem(last=False) #FIFO
                #        d=datetime.utcnow()
                #        traj['db_write_timestamp']=calendar.timegm(d.timetuple()) #epoch unix time
                        traj['db_write_timestamp'] = 0
                        traj['first_timestamp']=traj['timestamp'][0]
                        traj['last_timestamp']=traj['timestamp'][-1]
                        traj['starting_x']=traj['x_position'][0]
                        traj['ending_x']=traj['x_position'][-1]
                        traj['flags'] = ['fragment']
                        
        #                print("** write {} to db".format(ID))
                        col.insert_one(traj)
                
            f.close()
            
        # flush out all fragmentes in cache
        print("flush out all the rest of LRU of size {}".format(len(lru)))
        for ID, traj in lru.items():
            traj['db_write_timestamp'] = 0
            traj['first_timestamp']=traj['timestamp'][0]
            traj['last_timestamp']=traj['timestamp'][-1]
            traj['starting_x']=traj['x_position'][0]
            traj['ending_x']=traj['x_position'][-1]
            traj['flags'] = ['fragment']
            
            col.insert_one(traj)

    return
        
        
class DBWriter:
    """
    MongoDB database writer; uses asynchronous query mechanism in "motor" package by default.
    """

    def __init__(self, host, port, username, password, database_name,
                 server_id, process_name, process_id, session_config_id, num_workers = 200):
        """
        :param host: Database connection host name.
        :param port: Database connection port number.
        :param username: Database authentication username.
        :param password: Database authentication password.
        :param database_name: Name of database to connect to (do not confuse with collection name).
        :param server_id: ID value for the server this writer is running on.
        :param process_name: Name of the process this writer is attached to (writing from).
        :param process_id: ID value for the process this writer is attached to (writing from).
        :param session_config_id: Configuration ID value that was assigned to this run/session of data processing.
        """
        username = urllib.parse.quote_plus(username)
        password = urllib.parse.quote_plus(password)
        
        self.host, self.port = host, port
        self.username, self.password = username, password
        self.db_name = database_name
        self.server_id = server_id
        self.process_name = process_name
        self.process_id = process_id
        self.session_config_id = session_config_id

        # Connect immediately upon instantiation.
        self.client = pymongo.MongoClient('mongodb://%s:%s@%s' % (username, password, host))
        # self.client = pymongo.MongoClient(host=host, port=port, username=username, password=password,
        #                                   connect=True, connectTimeoutMS=5000)
        try:
            self.client.admin.command('ping')
        except pymongo.errors.ConnectionFailure:
            print("Server not available")
            raise ConnectionError("Could not connect to MongoDB using pymongo.")
            
        self.db = self.client[database_name]
        


    def thread_insert(self, collection, document):
        '''
        A wrapper around pymongo insert_one, which is a thread-safe operation
        '''
        collection.insert_one(document)
        
    def write_one_trajectory(self, thread = True, collection_name = "test_collection", **kwargs):
        """
        Write an arbitrary document specified in kwargs to a specified collection. No schema enforcment.
        :param thread: a boolean indicating if multi-threaded write is used
        :param collection_name: a string for write collection destination
        
        Use case:
        e.g.1. 
        dbw.write_one_trajectory(timestamp = [1,2,3], x_position = [12,22,33])
        e.g.2. 
        traj = {"timestamp": [1,2,3], "x_position": [12,22,33]}
        dbw.write_one_trajectory(**traj)
        """
        
        col = self.db[collection_name]
        doc = {} 
        for key,val in kwargs.items():
            doc[key] = val

        # add extra fields in doc
        configuration_id = self.session_config_id
        compute_node_id = self.server_id   
        
        doc["configuration_id"] = configuration_id
        doc["compute_node_id"] = compute_node_id
        
        if not thread:
            col.insert_one(doc)
        else:
            # fire off a thread
            t = Thread(target=self.thread_insert, args=(col, doc,))
            t.daemon = True
            t.start()    
            
            
    def write_fragment(self, thread = True, **kwargs):
        """
        Blocking write using insert_one for each document (slow)
        Write a raw trajectory according to the data schema, found here:
            https://docs.google.com/document/d/1xli3N-FCvIYhvg7HqaQSOcKY44B6ZktGHgcsRkjf7Vg/edit?usp=sharing
        Values that are in the schema, but assigned by the database are: db_write_timestamp, _id.
        Values that are in the schema, but calculated implicitly from others are: first_timestamp, last_timestamp,
            starting_x, ending_x.
        Values that are in the schema, but given to DBWriter at instantiation are: configuration_id, compute_node_id.
        :param local_fragment_id: Integer unique to each tracked vehicle per compute_node_ID and configuration_ID,
            generated by the tracker
        :param coarse_vehicle_class: Vehicle coarse class number
        :param fine_vehicle_class: Vehicle fine class number
        :param timestamps: Array of corrected timestamps; may be corrected to reduce timestamp errors.
        :param raw_timestamps: Raw timestamps from video frames as reported by the camera, may contain bias or errors.
        :param road_segment_id: Unique road segment ID; differentiates mainline from ramps.
        :param x_positions: Array of back-center x-position along the road segment in feet. X=0 is beginning of segment.
        :param y_positions: Array of back-center y-position across the road segment in feet. Y=0 is located at the left
            yellow line, i.e., the left-most edge of the left-most lane of travel in each direction.
        :param lengths: Vehicle length in feet.
        :param widths: Vehicle width in feet.
        :param heights: Vehicle height in feet.
        :param direction: Indicator of roadway direction (-1 or +1).
        :param flags: List of any string flags describing the data.
        :param camera_snapshots: Possibly empty array of JPEG compressed images (as bytes) of vehicles.
        
        :param thread: set True if create a thread for insert in the background (fast), False if blocking insert_one
        :return: None
        """
        col = self.db[db_parameters.RAW_COLLECTION]
        
        configuration_id = self.session_config_id
        compute_node_id = self.server_id
        doc = {}
        
        for field_name in db_parameters.RAW_SCHEMA:
            try:
                doc[field_name] = kwargs[field_name]
            except:
                pass
            
        # add extra fields in doc
        doc["configuration_id"] = configuration_id
        doc["compute_node_id"] = compute_node_id
        
        if not thread:
            col.insert_one(doc)
        else:
            # fire off a thread
            t = Thread(target=self.thread_insert, args=(col, doc,))
            t.daemon = True
            t.start()
    

    def write_stitched_trajectory(self, thread = True, **kwargs):
        """
        Write a stitched trajectory reference document according to the data schema, found here:
            https://docs.google.com/document/d/1vyLgsz6y0SrpTXWZNOS5fSgMnmwCr3xD0AB6MgYWl-w/edit?usp=sharing
        Values that are in the schema, but assigned by the database are: db_write_timestamp, _id.
        Values that are in the schema, but given to DBWriter at instantiation are: configuration_id.
        :param fragment_ids: List of fragment IDs associated to current vehicle stitched trajectory,
            sorted by start_timestamp.
        :return: None
        """
        col = self.db[db_parameters.STITCHED_COLLECTION]
        
        configuration_id = self.session_config_id
        compute_node_id = self.server_id
        doc = {}
        
        for field_name in db_parameters.STITCHED_SCHEMA:
            try:
                doc[field_name] = kwargs[field_name]
            except:
                pass
            
        # add extra fields in doc
        doc["configuration_id"] = configuration_id
        doc["compute_node_id"] = compute_node_id
        
        if not thread:
            col.insert_one(doc)
        else:
            # fire off a thread
            t = Thread(target=self.thread_insert, args=(col, doc,))
            t.daemon = True
            t.start()
    
    
    def write_reconciled_trajectory(self, thread = True, **kwargs):
        """
        Write a reconciled/post-processed trajectory according to the data schema, found here:
            https://docs.google.com/document/d/1Qh4OYOhOi1Kh-7DEwFfLx8NX8bjaFdviD2Q0GsfgR9k/edit?usp=sharing
        Values that are in the schema, but assigned by the database are: db_write_timestamp, _id
        Values that are in the schema, but calculated implicitly from others are: first_timestamp, last_timestamp,
            starting_x, ending_x.
        Values that are in the schema, but given to DBWriter at instantiation are: configuration_id.
        :param vehicle_id: Same vehicle_id assigned during stitching.
        :param coarse_vehicle_class: Vehicle coarse class number.
        :param fine_vehicle_class: Vehicle fine class number.
        :param timestamps: Corrected timestamps; may be corrected to reduce timestamp errors.
        :param road_segment_id: Unique road segment ID; differentiates mainline from ramps.
        :param x_positions: Array of back-center x-position along the road segment in feet. X=0 is beginning of segment.
        :param y_positions: Array of back-center y-position across the road segment in feet. Y=0 is located at the left
            yellow line, i.e., the left-most edge of the left-most lane of travel in each direction.
        :param length: Vehicle length in feet.
        :param width: Vehicle width in feet.
        :param height: Vehicle height in feet.
        :param direction: Indicator of roadway direction (-1 or +1).
        :param flags: List of any string flags describing the data.
        :return: None
        """
        col = self.db[db_parameters.RECONCILED_COLLECTION]
        
        configuration_id = self.session_config_id
        compute_node_id = self.server_id
        doc = {}
        
        for field_name in db_parameters.RECONCILED_SCHEMA:
            try:
                doc[field_name] = kwargs[field_name]
            except:
                pass
            
        # add extra fields in doc
        doc["configuration_id"] = configuration_id
        doc["compute_node_id"] = compute_node_id
        
        if not thread:
            col.insert_one(doc)
        else:
            # fire off a thread
            t = Thread(target=self.thread_insert, args=(col, doc,))
            t.daemon = True
            t.start()
            
            
    def write_metadata(self, metadata):
        pass


    def write_ground_truth_trajectory(self, thread = True, **kwargs):
        """
        Write a ground truth trajectory according to the data schema, found here:
            https://docs.google.com/document/d/1zbjPycZlGNPOwuPVtY5GkS3LvIZwMDOtL7yFc575kSw/edit?usp=sharing
        Values that are in the schema, but assigned by the database are: db_write_timestamp, _id
        Values that are in the schema, but calculated implicitly from others are: first_timestamp, last_timestamp,
            starting_x, ending_x.
        Values that are in the schema, but given to DBWriter at instantiation are: configuration_id.
        :param vehicle_id: Same vehicle_id assigned during stitching.
        :param fragment_ids: Array of fragment_id values associated to current vehicle_id.
        :param coarse_vehicle_class: Vehicle coarse class number.
        :param fine_vehicle_class: Vehicle fine class number.
        :param timestamps: Corrected timestamps; may be corrected to reduce timestamp errors.
        :param road_segment_id: Unique road segment ID; differentiates mainline from ramps.
        :param x_positions: Array of back-center x-position along the road segment in feet. X=0 is beginning of segment.
        :param y_positions: Array of back-center y-position across the road segment in feet. Y=0 is located at the left
            yellow line, i.e., the left-most edge of the left-most lane of travel in each direction.
        :param length: Vehicle length in feet.
        :param width: Vehicle width in feet.
        :param height: Vehicle height in feet.
        :param direction: Indicator of roadway direction (-1 or +1).
        :return: None
        """
        col = self.db[db_parameters.GT_COLLECTION]
        
        configuration_id = self.session_config_id
        compute_node_id = self.server_id
        doc = {}
        
        for field_name in db_parameters.GT_SCHEMA:
            try:
                doc[field_name] = kwargs[field_name]
            except:
                pass
            
        # add extra fields in doc
        doc["configuration_id"] = configuration_id
        doc["compute_node_id"] = compute_node_id
        
        if not thread:
            col.insert_one(doc)
        else:
            # fire off a thread
            t = Thread(target=self.thread_insert, args=(col, doc,))
            t.daemon = True
            t.start()
            
    
    def __del__(self):
        """
        Upon DBReader deletion, close the client/connection.
        :return: None
        """
        try:
            self.client.close()
        except:
            pass