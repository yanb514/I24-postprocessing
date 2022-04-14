import multiprocessing

import pymongo
import pymongo.errors

from collections.abc import Iterable
from typing import Union
Numeric = Union[int, float]


def format_flat(columns, document, impute_none_for_missing=False):
    """
    Takes a `document` returned from MongoDB and organizes it into an ordered list of values specified by `columns`.
    :param columns: List of column names corresponding to document fields.
    :param document: Dictionary-like document from MongoDB with fields corresponding to column names.
    :param impute_none_for_missing: If True, sets value to None for any missing columns in `document`.
    :return: List of values corresponding to columns.
    """
    # TODO: fill this in
    # TODO: type hints correct to the data type the document will come through as
    return []


def read_query_once(host: str, port: int, username: str, password: str, database_name: str, collection_name: str,
                    query_filter: Union[dict, None], query_sort: Iterable[tuple[str, str]] = None, limit: int = 0):
    """
    Executes a single database read query using the DBReader object, which is destroyed immediately upon completion.
    :param host: Database connection host name.
    :param port: Database connection port number.
    :param username: Database authentication username.
    :param password: Database authentication password.
    :param database_name: Name of database to connect (do not confuse with collection name).
    :param collection_name: Name of database collection from which to query.
    :param query_filter: Currently a dict following pymongo convention (need to abstract this).
    :param query_sort: List of tuples: (field_to_sort, sort_direction); direction is ASC/ASCENDING or DSC/DESCENDING.
    :param limit: Numerical limit for number of documents returned by query.
    :return:
    """
    dbr = DBReader(host=host, port=port, username=username, password=password,
                   database_name=database_name, collection_name=collection_name)
    result = dbr.read_query(query_filter=query_filter, query_sort=query_sort, limit=limit)
    return result


def read_data_into_csv(host: str, port: int, username: str, password: str, database_name: str, collection_name: str,
                       csv_file_path: str, query_filter: Union[dict, None],
                       query_sort: Iterable[tuple[str, str]] = None, limit: int = 0):
    """
    Executes a single database read query and writes the results to a CSV file with pre-defined format.l
    :param host: Database connection host name.
    :param port: Database connection port number.
    :param username: Database authentication username.
    :param password: Database authentication password.
    :param database_name: Name of database to connect to (do not confuse with collection name).
    :param collection_name: Name of database collection from which to query.
    :param csv_file_path: Absolute or relative file path at which to save CSV file; directories must exist already.
    :param query_filter: Currently a dict following pymongo convention (need to abstract this).
    :param query_sort: List of tuples: (field_to_sort, sort_direction); direction is ASC/ASCENDING or DSC/DESCENDING.
    :param limit: Numerical limit for number of documents returned by query.
    :return:
    """
    # TODO: fill these in; they should probably be stored elsewhere in some configuration file
    if collection_name == 'raw':
        csv_columns = []
    elif collection_name == 'stitched':
        csv_columns = []
    elif collection_name == 'reconciled':
        csv_columns = []
    elif collection_name == 'metadata':
        csv_columns = []
    else:
        raise ValueError("Invalid database collection name.")

    dbr = DBReader(host=host, port=port, username=username, password=password,
                   database_name=database_name, collection_name=collection_name)
    result = dbr.read_query(query_filter=query_filter, query_sort=query_sort, limit=limit)
    import csv
    with open(csv_file_path, 'w') as wf:
        writer = csv.writer(csvfile=wf, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        for document in result:
            doc_fmt = format_flat(columns=csv_columns, document=document)
            writer.writerow(doc_fmt)


def live_data_reader(host: str, port: int, username: str, password: str, database_name: str, db_collection: str,
                     ready_queue: multiprocessing.Queue):
    """
    Runs a database stream update listener on top of a managed cache that buffers data for a safe amount of time so
        that it can be assured to be time-ordered.
    ** THIS PROCEDURE AND FUNCTION IS STILL UNDER DEVELOPMENT **
    :param host: Database connection host name.
    :param port: Database connection port number.
    :param username: Database authentication username.
    :param password: Database authentication password.
    :param database_name: Name of database to connect to (do not confuse with collection name).
    :param db_collection: Name of database collection from which to query.
    :param ready_queue: Process-safe queue to which records that are "ready" are written.
    :return:
    """
    # TODO: determine the strategy to use here
    pass


class DBReader:
    """
    MongoDB database reader, specific to a collection in the database. This object is typically fairly persistent, i.e.,
        sticks around for a while to execute multiple queries.
    """

    def __init__(self, host: str, port: int, username: str, password: str, database_name: str, collection_name: str):
        """
        Connect to the specified MongoDB instance, test the connection, then set the specific database and collection.
        :param host: Database connection host name.
        :param port: Database connection port number.
        :param username: Database authentication username.
        :param password: Database authentication password.
        :param database_name: Name of database to connect to (do not confuse with collection name).
        :param collection_name: Name of database collection from which to query.
        """
        # Connect immediately upon instantiation.
        self.client = pymongo.MongoClient(host=host, port=port, username=username, password=password,
                                          connect=True, connectTimeoutMS=5000)
        # Test out the connection with a ping and raise a ConnectionError if it isn't available.
        # Connection timeout specified during creation of self.client.
        try:
            self.client.admin.command('ping')
        except pymongo.errors.ConnectionFailure:
            print("Server not available")
            raise ConnectionError("Could not connect to MongoDB.")

        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

        # Class variables that will be set and reset during iterative read across a range.
        self.range_iter_parameter = None
        self.range_iter_sort = None
        self.range_iter_start = None
        self.range_iter_start_closed_interval = None
        self.range_iter_increment = None
        self.range_iter_stop = None
        self.range_iter_stop_closed_interval = None

    def __del__(self):
        """
        Upon DBReader deletion, close the client/connection.
        :return: None
        """
        try:
            self.client.close()
        except pymongo.errors.PyMongoError:
            pass

    def read_query(self, query_filter: Union[dict, None], query_sort: Iterable[tuple[str, str]] = None,
                   limit: int = 0) -> pymongo.cursor.Cursor:
        """
        Executes a read query against the database collection.
        :param query_filter: Currently a dict following pymongo convention (need to abstract this).
        :param query_sort: List of tuples: (field_to_sort, sort_direction); direction is ASC/ASCENDING or DSC/DESCENDING
        :param limit: Numerical limit for number of documents returned by query.
        :return:
        """
        if query_sort is not None:
            sort_fields = []
            for sort_field, sort_dir in query_sort:
                if sort_dir.upper() in ('ASC', 'ASCENDING'):
                    sort_fields.append((sort_field, pymongo.ASCENDING))
                elif sort_dir.upper() in ('DSC', 'DESCENDING'):
                    sort_fields.append((sort_field, pymongo.DESCENDING))
                else:
                    raise ValueError("Invalid direction for sort. Use 'ASC'/'ASCENDING' or 'DSC'/'DESCENDING'.")
        else:
            sort_fields = None

        # If user passed None, substitute an empty dictionary (per the PyMongo convention).
        if query_filter is None:
            filter_field = {}
        else:
            filter_field = query_filter

        # TODO: check this is the desired/correct syntax and doesn't need more options
        result = self.collection.find(filter=filter_field, limit=limit, sort=sort_fields)
        # TODO: document the return types and interaction
        return result

    # TODO: also datetime for range bounds??
    def read_query_range(self, range_parameter: str,
                         range_greater_than: Union[Numeric, None] = None,
                         range_greater_equal: Union[Numeric, None] = None,
                         range_less_than: Union[Numeric, None] = None,
                         range_less_equal: Union[Numeric, None] = None,
                         range_increment: Union[Numeric, None] = None,
                         query_sort: Iterable[tuple[str, str]] = None,
                         limit: int = 0) -> Union[pymongo.cursor.Cursor, Iterable]:
        """
        Iterate across a query range in portions.
        Usage:
        ```
            # Method 1: FOR loop across function call
            for result in dbr.read_query_range(range_parameter='t', range_greater_than=0, range_less_equal=100,
                                                range_increment=10):
                print(result)
            # Method 2: WHILE loop with next(...)
            rqr = dbr.read_query_range(range_parameter='t', range_greater_equal=0, range_less_than=100,
                                        range_increment=10)
            while True:
                try:
                    result = next(rri)
                    print(result)
                except StopIteration:
                    print("END OF ITERATION")
                    break
        ```
        :param range_parameter: Document field across which to run range queries.
        :param range_greater_than: Sets a '>' bound on `range_parameter` for the query or successive queries.
        :param range_greater_equal: Sets a '>' bound on `range_parameter` for the query or successive queries.
        :param range_less_than: Sets a '>' bound on `range_parameter` for the query or successive queries.
        :param range_less_equal: Sets a '>' bound on `range_parameter` for the query or successive queries.
        :param range_increment: When None, executes the range query as a one-off and returns result; otherwise,
            returns iterable of queries/results.
        :param query_sort: List of tuples: (field_to_sort, sort_direction); direction is ASC/ASCENDING or DSC/DESCENDING
        :param limit: Numerical limit for number of documents returned by query.
        :return: iterator across range-segmented queries (each query executes when __next__() is called in iteration)
        """
        if range_greater_than is None and range_greater_equal is None and range_less_than is None \
                and range_less_equal is None:
            raise ValueError("Must specify lower and or upper bound (inclusive or exlusive) for range query.")
        if (range_greater_than is None and range_greater_equal is None) or \
                (range_less_than is None and range_less_equal is None):
            raise NotImplementedError("Infinite ranges not currently supported.")

        if range_increment is None:
            # TODO: construct this filter with the right lt/lte/gt/gte syntax based on inputs
            range_filter = None
            return 'param: {}'.format(range_parameter), '>{}'.format(range_greater_than), \
                   '≥{}'.format(range_greater_equal), '<{}'.format(range_less_than), '≤{}'.format(range_less_equal)
            return self.read_query(query_filter=range_filter, query_sort=query_sort, limit=limit)
        else:
            self.range_iter_parameter = range_parameter
            self.range_iter_increment = range_increment
            self.range_iter_sort = query_sort

            if range_greater_equal is not None:
                self.range_iter_start = range_greater_equal
                self.range_iter_start_closed_interval = True
            elif range_greater_than is not None:
                self.range_iter_start = range_greater_than
                self.range_iter_start_closed_interval = False
            else:
                # Currently, this point should not be reachable, given the check against infinite ranges.
                pass

            if range_less_equal is not None:
                self.range_iter_stop = range_less_equal
                self.range_iter_stop_closed_interval = True
            elif range_less_than is not None:
                self.range_iter_stop = range_less_than
                self.range_iter_stop_closed_interval = False
            else:
                # Currently, this point should not be reachable, given the check against infinite ranges.
                pass

        return iter(self)

    def read_stream(self):
        pass

    def __iter__(self):
        if self.range_iter_parameter is None or self.range_iter_start is None or self.range_iter_increment is None \
                or self.range_iter_stop is None or self.range_iter_start_closed_interval is None \
                or self.range_iter_stop_closed_interval is None:
            raise AttributeError("Iterable DBReader only supported via `read_query_range(...).")
        return DBReadRangeIterator(self)


class DBReadRangeIterator:
    """
    Iterable class for executing successive queries using a DBReader. The range iteration values must be set in the
        DBReader before instantiating this object. They will be set back to None upon the end of iteration.
    """

    def __init__(self, db_reader: DBReader):
        self._reader = db_reader
        self._current_lower_value = self._reader.range_iter_start
        self._current_upper_value = self._current_lower_value + self._reader.range_iter_increment
        # Initialize first/last iteration indicator variables.
        self._first_iter = True
        self._last_iter_exit_flag = False

    def _reset_range_iter(self):
        """
        Goes into the DBReader instance and resets all of its range iteration values back to None.
        :return: None
        """
        self._reader.range_iter_parameter = None
        self._reader.range_iter_sort = None
        self._reader.range_iter_start = None
        self._reader.range_iter_start_closed_interval = None
        self._reader.range_iter_increment = None
        self._reader.range_iter_stop = None
        self._reader.range_iter_stop_closed_interval = None

    def _update_values(self):
        """
        Increments the current iteration lower and upper bound. No interval open/closed indication needed because
            iterations other than the first and last are always [lower, upper) interval format.
        :return: None
        """
        self._current_lower_value = self._current_upper_value
        self._current_upper_value = self._current_upper_value + self._reader.range_iter_increment

    def __next__(self):
        """
        Runs the next range query based on the current values (self._current_...). Computes the next current values
            as well as the open/closed intervals. Sets and reacts to a flag for last iteration and raises
            StopIteration exception when complete.
        :return: result of next read query within the iteration range
        """
        # If the last iteration set this flag, then we need to stop iteration.
        # But if this current iteration is the last one that will return anything, we'll set the flag this time.
        if self._last_iter_exit_flag is True:
            self._reset_range_iter()
            raise StopIteration

        # Check if this will be the last query -- i.e., the current upper value met or exceeded the range stop.
        if self._current_upper_value >= self._reader.range_iter_stop:
            # Set the flag to exit next iteration.
            self._last_iter_exit_flag = True
            query_upper_value = self._reader.range_iter_stop
        else:
            query_upper_value = self._current_upper_value

        # If this is the first iteration, check whether we are doing open or closed interval on greater-than side.
        if self._first_iter is True:
            if self._reader.range_iter_start_closed_interval is True:
                gt, gte = None, self._current_lower_value
            else:
                gt, gte = self._current_lower_value, None
        # After first iteration, always do closed interval on greater-than side.
        else:
            gt, gte = None, self._current_lower_value

        # If this is the last iteration, check whether we are doing open or closed interval on the less-than side.
        # We will only reach this point if this is the last results-gathering iteration.
        # The exit flag indicates we're about to stop, but we still need to get one more set of results.
        if self._last_iter_exit_flag is True:
            if self._reader.range_iter_stop_closed_interval is True:
                lt, lte = None, query_upper_value
            else:
                lt, lte = query_upper_value, None
        # Before last iteration, always do open interval on less-than side.
        else:
            lt, lte = query_upper_value, None

        # Now that the range is calculated, execute outright (no increment) using `DBReader.read_query_range(...)`.
        # We use the range function so that we don't have to do the formatting of the query filter manually.
        iter_result = self._reader.read_query_range(range_parameter=self._reader.range_iter_parameter,
                                                    range_greater_than=gt, range_greater_equal=gte,
                                                    range_less_than=lt, range_less_equal=lte,
                                                    query_sort=self._reader.range_iter_sort, range_increment=None)
        # No matter what, this is not the first iteration anymore at this point.
        self._first_iter = False

        # Increment the values for the next iteration.
        # Even if this is the last results-gathering iteration, it's fine to increment the values.
        self._update_values()
        return iter_result

    def __iter__(self):
        """
        Needed in order to place DBReader.read_range_query(...) into a FOR loop.
        :return: self
        """
        return self
