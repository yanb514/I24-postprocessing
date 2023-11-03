"""
This script prepares files for box upload
1. read a trajectory collection from mongoDB
2. write to json files in smaller chunks (decided by chunk_size sec)
3. zip all .json files into a .zip file
4. ssh to a remote server
"""
from i24_database_api import DBClient
import os
import json
from bson.json_util import dumps
import sys

import os
import zipfile



# # Step 3: SSH connect to another server
# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect('servername.edu', username='user', password='password')

# # Step 4: Move the abc.zip file to the server
# sftp = ssh.open_sftp()
# sftp.put(zip_filename, zip_filename)

# # Cleanup: Close the SSH connection and remove the local zip file
# sftp.close()
# ssh.close()
# os.remove(zip_filename)

# print("File transfer completed.")




def zip_files(keyword, directory):
    # keyword is collection_name
    files = [filename for filename in os.listdir(directory) if filename.startswith(keyword) and ".json" in filename]

    # Step 2: Zip the selected files to abc.zip
    zip_filename = f"{keyword}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in files:
            file_path = os.path.join(directory, file)
            zip_file.write(file_path, arcname=file)
    return

def decimal_range(start, stop, increment):
    while start < stop: # and not math.isclose(start, stop): Py>3.5
        yield start
        start += increment
        

def main(database_name="", collection_name="", chunk_size = 600):
    
    with open(os.environ["USER_CONFIG_DIRECTORY"]+"/db_param.json") as f:
        db_param = json.load(f)
     
    # chunk_size = 100 # 600sec=10min
    dbc = DBClient(**db_param, database_name=database_name, collection_name = collection_name)
    
    start = dbc.collection.find_one(sort=[("first_timestamp", 1)])["first_timestamp"]
    end = dbc.collection.find_one(sort=[("last_timestamp", -1)])["first_timestamp"]
    # end = start+200
    
    
    # specify query - iterative ranges
    idx = 0
    for s in decimal_range(start, end, chunk_size):

        print("Writing JSON (approx) {:.1f} %".format((s-start)/(end-start)*100), end="\r")
        cursor = dbc.collection.find({"first_timestamp": {"$gte": s, "$lt": s+chunk_size}}).sort("first_timestamp",1)

        with open('{}_{:02d}.json'.format(collection_name, idx), 'w') as file: 
            json.dump(json.loads(dumps(cursor)), file)

        idx += 1
        
    print("Completed writing {} json files".format(idx))

    # zip them!
    print("zipping files...")
    directory = os.getcwd()
    zip_files(collection_name, directory)
    print(f"Completed zipping to {collection_name}.zip")
    
        
        
if __name__ == '__main__':
    
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
        
    else:
        print("invalid input")