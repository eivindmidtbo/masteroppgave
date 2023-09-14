import datetime
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import time
import pymongo
import haversine
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

    def __init__(self,
                 DATABASE='assignment3_mongodb_1',
                 HOST="localhost",
                 PORT="27017",
                 USER="gruppe13",
                 PASSWORD=os.getenv('PASSWORD')):
        uri = "mongodb://%s:%s@%s:%s/" % (USER, PASSWORD, HOST, PORT)

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

    def create_collection(self, collection_name):
        self.db.create_collection(collection_name)
        print("Collection %s created" % collection_name)

    def remove_collection(self, collection_name):
        if (collection_name in self.db.list_collection_names()):
            self.db.drop_collection(collection_name)
            print("Collection %s removed" % collection_name)

    def create_indexes(self):
        print("creating indexes...")
        self.db["activity"].create_index("user_id")
        self.db["activity"].create_index("transportation_mode")
        self.db["activity"].create_index("start_date_time")
        self.db["trackpoint"].create_index("activity_id")
        self.db["trackpoint"].create_index([("location", pymongo.GEOSPHERE)])
    
    ############################################################
    ###################### Inserts #############################
    ############################################################

    def batch_insert_users(self, user_list):
        print("inserting users...")
        start_time = time.time()
        try:
            self.db["user"].insert_many(user_list)
            print("finished insert in %s seconds" % (time.time() - start_time))
        except Exception as e:
            print(e)

    def batch_insert_activities_with_id(self, activities):
        print("inserting activities...")
        start_time = time.time()
        try:
            self.db["activity"].insert_many(activities)
            print("finished insert in %s seconds" % (time.time() - start_time))
        except Exception as e:
            print(e)

    def insert_trackpoints_with_id(self, trackpoints):
        try:
            self.db["trackpoint"].insert_many(trackpoints)
        except Exception as e:
            print(e)


    ############################################################
    ###################### Queries #############################
    ############################################################

    ## TASK 2 ##
    def find_average_activities_per_user(self):
        return self.db["activity"].aggregate([
            {
                "$group": {
                    "_id": "$user_id",
                    "count": {"$sum": 1}
                }
            },
            {
                "$group": {
                    "_id": "null",
                    "avg": {"$avg": "$count"}
                }
            }
        ])


    ## TASK 3 ##
    def find_top_20_users_with_most_activities(self):
        return self.db["activity"].aggregate([
            {
                "$group": {
                    "_id": "$user_id",
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"count": -1}
            },
            {
                "$limit": 20
            }
        ])


    ## TASK 4 ##
    def find_users_who_have_taken_taxi(self):
        return self.db["activity"].aggregate([
            {
                "$match": {
                    "transportation_mode": "taxi"
                }
            },
            {
                "$group": {
                    "_id": "$user_id"
                }
            }
        ])


    ## TASK 5 ##
    def find_all_types_of_transportation_modes(self):
        return self.db["activity"].aggregate([
            {
                "$match": {
                    "transportation_mode": {"$ne": None}
                }
            },
            {
                "$group": {
                    "_id": "$transportation_mode",
                    "count": {"$sum": 1}
                }
            }
        ])


    ## TASK 6 a ##
    def find_year_with_most_activities(self):
        return self.db["activity"].aggregate([
            {
                "$group": {
                    "_id": {"$year": "$start_date_time"},
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"count": -1}
            },
            {
                "$limit": 1
            }
        ])


    ## TASK 6 b ##
    def find_year_with_most_recorded_hours(self):
        return self.db["activity"].aggregate([
            {
                "$group": {
                    "_id": {"$year": "$start_date_time"},
                    "count": {"$sum": {"$subtract": ["$end_date_time",
                                                     "$start_date_time"]}}
                }
            },
            {
                "$sort": {"count": -1}
            },
            {
                "$limit": 1
            }
        ])


    ## TASK 7 ##
    def find_trackpoints_of_activity(self, activity_id):
        return self.db["trackpoint"].find({"activity_id": activity_id})

    def find_activity_ids(self):
        return self.db["activity"].find({
            "user_id": "112",
            "start_date_time": {"$gte": datetime.datetime(2008, 1, 1, 0, 0, 0)},
            "transportation_mode": "walk"
        })


    ## TASK 8 ##
    def find_all_user_ids(self):
        return self.db["user"].find({}, {"user_id": 1})

    def find_all_activity_ids_of_user(self, user_id):
        return self.db["activity"].aggregate([
            {
                "$match": {
                    "user_id": user_id,
                }
            },
            {
                "$project": {
                    "activity_id": 1
                }
            }])

    def find_altitude_of_activity(self, activity_id):
        return self.db["trackpoint"].find({"activity_id": activity_id}, {"altitude": 1})


    ## TASK 9 ##
    def find_date_time_of_activity(self, activity_id):
        return self.db["trackpoint"].find({"activity_id": activity_id}, {"date_time": 1})


    ## TASK 10 ##
    def find_users_with_activity_with_trackpoint_at_location(self):
        results = self.db["trackpoint"].aggregate([
            {
                '$geoNear': {
                    'near': {
                        'type': 'Point',
                        'coordinates': [
                            116.397, 39.916
                        ]
                    },
                    'distanceField': 'dist.calculated',
                    'maxDistance': 100,
                    'includeLocs': 'dist.location',
                    'spherical': True
                }
            }])
        activity_ids = []
        users = []
        for result in results:
            activity_ids.append(result["activity_id"])

        for user in self.db["activity"].aggregate([
            {
                "$match": {
                    "_id": {"$in": activity_ids}
                }
            },
            {
                "$group": {
                    "_id": "$user_id"
                }
            }
        ]):
            users.append(user)
        return users


    ## TASK 11 ##
    def find_most_frequent_transportation_mode_for_each_user(self):
        return self.db["activity"].aggregate([
            {
                "$match": {
                    "transportation_mode": {"$ne": None}
                }
            },
            {
                "$group": {
                    "_id": {"user_id": "$user_id", "transportation_mode": "$transportation_mode"},
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"count": -1}
            },
            {
                "$group": {
                    "_id": "$_id.user_id",
                    "transportation_mode": {"$first": "$_id.transportation_mode"}
                }
            },
            {
                "$sort": {"_id": 1}
            },
        ])