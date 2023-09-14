import haversine
import pymongo
import time
from dotenv import load_dotenv
import os
from pymongo import MongoClient
import datetime
import numpy as np

URI = "mongodb://it2810-45.idi.ntnu.no:27017/webdev"


load_dotenv()


class DbConnector:
    """
    Connects to the MongoDB server on the Ubuntu virtual machine.
    Connector needs HOST, USER and PASSWORD to connect.

    Example:
    HOST = "tdt4225-00.idi.ntnu.no" // Your server IP address/domain name
    USER = "testuser" // This is the user you created and added privileges for
    PASSWORD = "test123" // The password you set for said user
    """

    def __init__(
        self,
        DATABASE="assignment3_mongodb_1",
        HOST="localhost",
        PORT="27017",
        USER="gruppe13",
        PASSWORD=os.getenv("PASSWORD"),
    ):
        # uri = "mongodb://%s:%s@%s:%s/" % (USER, PASSWORD, HOST, PORT)
        uri = URI
        # Connect to the databases
        try:
            self.client = MongoClient(uri)
            self.db = self.client[DATABASE]
        except Exception as e:
            print("ERROR: Failed to connect to db:", e)

        # get database information
        print("You are connected to the database:", self.db.name)
        print("-----------------------------------------------\n")

    def close_connection(self):
        # close the cursor
        # close the DB connection
        self.client.close()
        print("\n-----------------------------------------------")
        print("Connection to %s-db is closed" % self.db.name)

    def update_field(self):
        movies = np.genfromtxt(
            "netflix_titles.csv", skip_header=1, delimiter='""', dtype=str
        )
        print(movies)
        print("test")
        print(self.db["movies"].find())


def main():
    db = DbConnector()
    result = db.update_field()
    for i in result:
        print(i)
    # print(result)
    # for i in result:
    #     print(i)
    # print(db.update_field())


main()
