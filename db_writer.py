import pymongo
import db_parameters
import csv
import urllib.parse
from threading import Thread
import schema
from i24_logger.log_writer import logger 
        
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
        # self.client = pymongo.MongoClient(host=host, port=port, username=username, 
        #                                   password=password,
        #                                   connect=True, 
        #                                   connectTimeoutMS=5000,
        #                                   authSource='admin'
        #                                   )
        try:
            self.client.admin.command('ping')
        except pymongo.errors.ConnectionFailure:
            print("Server not available")
            raise ConnectionError("Could not connect to MongoDB using pymongo.")
            
        self.db = self.client[database_name]
        
        # create three critical collections
        try: self.db.create_collection(db_parameters.RAW_COLLECTION)
        except: pass   
        try: self.db.create_collection(db_parameters.STITCHED_COLLECTION)
        except: pass
        try: self.db.create_collection(db_parameters.RECONCILED_COLLECTION)
        except: pass
    
        # set rules for schema. enable schema checking when insert using
        # col.insert_one(doc)
        # disable schema checking: col.insert_one(doc, bypass_document_validation=False)
        self.db.command("collMod", db_parameters.RAW_COLLECTION, validator=schema.RAW_SCHEMA)
        self.db.command("collMod", db_parameters.STITCHED_COLLECTION, validator=schema.STITCHED_SCHEMA)
        self.db.command("collMod", db_parameters.RECONCILED_COLLECTION, validator=schema.RECONCILED_SCHEMA)
        

    def thread_insert(self, collection, document):
        '''
        A wrapper around pymongo insert_one, which is a thread-safe operation
        bypass_document_validation = True: enforce schema
        '''
        try:
            collection.insert_one(document, bypass_document_validation = db_parameters.BYPASS_VALIDATION)
        except: # schema violated
            logger.warning("insert failed, please follow schema")
        
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
            
    
    # simple query functions
    def get_first(self, index_name):
        '''
        get the first document from MongoDB by index_name
        '''
        return self.collection.find_one(sort=[(index_name, pymongo.ASCENDING)])
        
    def get_last(self, index_name):
        '''
        get the last document from MongoDB by index_name        
        '''
        return self.collection.find_one(sort=[(index_name, pymongo.DESCENDING)])
    
    def find_one(self, index_name, index_value):
        return self.collection.find_one({index_name: index_value})
        
    def is_empty(self):
        return self.count() == 0
        
    def get_keys(self): 
        oneKey = self.collection.find().limit(1)
        for key in oneKey:
            return key.keys()
        
    def create_index(self, indices):
        all_field_names = self.collection.find_one({}).keys()
        existing_indices = self.collection.index_information().keys()
        for index in indices:
            if index in all_field_names:
                if index+"_1" not in existing_indices and index+"_-1" not in existing_indices:
                    self.collection.create_index(index)     
        return
    
    def get_range(self, index_name, start, end): 
        return self.collection.find({
            index_name : { "$gte" : start, "$lt" : end}}).sort(index_name, pymongo.ASCENDING)
    
    def count(self):
        return self.collection.count_documents({})
    
    def get_min(self, index_name):
        return self.get_first(index_name)[index_name]
    
    def get_max(self, index_name):
        return self.get_last(index_name)[index_name]
    
    def exists(self, index_name, value):
        return self.collection.count_documents({index_name: value }, limit = 1) != 0
    
    
    def __del__(self):
        """
        Upon DBReader deletion, close the client/connection.
        :return: None
        """
        try:
            self.client.close()
        except:
            pass