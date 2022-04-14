from db_reader import DBReader

dbr = DBReader(host='localhost', port=0, username='user', password='password',
               database_name='db', collection_name='col')
rri = dbr.read_query_range(range_parameter='t', range_greater_equal=0, range_less_than=100, range_increment=10)
while True:
    try:
        print(next(rri))
    except StopIteration:
        print("END OF ITERATION")
        break

for result in dbr.read_query_range(range_parameter='t', range_greater_than=0, range_less_equal=100, range_increment=10):
    print(result)
print("END OF ITERATION")
