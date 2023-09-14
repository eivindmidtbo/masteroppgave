import mysql.connector as mysql
# from decouple import config
import os
from dotenv import load_dotenv
import time
from haversine import haversine, Unit

load_dotenv()


class DbConnector:
    """
    Connects to the MySQL server on the Ubuntu virtual machine.
    Connector needs HOST, DATABASE, USER and PASSWORD to connect,
    while PORT is optional and should be 3306.

    Example:
    HOST = "tdt4225-00.idi.ntnu.no" // Your server IP address/domain name
    DATABASE = "testdb" // Database name, if you just want to connect to MySQL server, leave it empty
    USER = "testuser" // This is the user you created and added privileges for
    PASSWORD = "test123" // The password you set for said user
    """

    def __init__(self,
                 # HOST="tdt4225-13.idi.ntnu.no",
                 HOST="localhost",
                 PORT=3306,
                 DATABASE="exercise2",
                 # USER="gruppe13",
                 USER="root",
                 PASSWORD=os.getenv('PASSWORD')):
        # Connect to the database
        try:
            print("password: " + os.getenv('PASSWORD'))
            self.db_connection = mysql.connect(
                host=HOST, database=DATABASE, user=USER, password=PASSWORD, port=PORT)
        except Exception as e:
            print("ERROR: Failed to connect to db:", e)

        self.start_time = time.time()
        # Get the db cursor
        self.cursor = self.db_connection.cursor()

        print("Connected to:", self.db_connection.get_server_info())
        # get database information
        self.cursor.execute("select database();")
        database_name = self.cursor.fetchone()
        print("You are connected to the database:", database_name)
        print("-----------------------------------------------\n")

    def close_connection(self):
        # close the cursor
        self.cursor.close()
        # close the DB connection
        self.db_connection.close()
        print("\n-----------------------------------------------")
        print("Connection to %s is closed" %
              self.db_connection.get_server_info())

    def create_Table(self, query):
        self.cursor.execute(query)
        self.db_connection.commit()

    def drop_table(self, table):
        self.cursor.execute("DROP TABLE IF EXISTS %s" % table)
        self.db_connection.commit()

    def batch_insert_users(self, user_list):
        print("inserting users...")
        query = "INSERT INTO user (id, has_labels) VALUES (%s, %s)"
        self.cursor.executemany(query, user_list)
        self.db_connection.commit()
        print("finished insert!")

    def get_last_inserted_id(self):
        return self.cursor.lastrowid
    
    def remove_invalid_altitudes(self):  
        query = """UPDATE trackpoint SET altitude = %s WHERE altitude = %s"""
        val = (int(), int(-777))
        #query = """UPDATE trackpoint SET altitude = 0 WHERE altitude = IN(-777)"""
        self.cursor.execute(query, val)
        self.db_connection.commit()  

    def insert_activity_with_id(self, activity):
        try:
            if(activity["transportation_mode"] != False):
                query = "INSERT INTO activity (id, user_id, transportation_mode, start_date_time, end_date_time) VALUES ('%s', '%s', '%s', '%s', '%s')"
                self.cursor.execute(query % (activity["id"], 
                                            activity["user_id"], 
                                            activity["transportation_mode"], 
                                            activity["start_date_time"], 
                                            activity["end_date_time"]))
            else: 
                query = "INSERT INTO activity (id, user_id, start_date_time, end_date_time) VALUES ('%s', '%s', '%s', '%s')"
                self.cursor.execute(query % (activity["id"], 
                                            activity["user_id"], 
                                            activity["start_date_time"], 
                                            activity["end_date_time"]))
            self.db_connection.commit()
        except Exception as e:
            print(e)

    def insert_trackpoints_with_id(self, trackpoints):
        try:
            query = "INSERT INTO trackpoint (id, activity_id, lat, lon, altitude, date_days, date_time) VALUES {}".format(trackpoints)
            self.cursor.execute(query)
            self.db_connection.commit()
        except Exception as e:
            print(e)
    
    def update_activity_labels(self, labels):
        query = "UPDATE activity SET transportation_mode = null WHERE transportation_mode is not null"
        self.cursor.execute(query)
        self.db_connection.commit()

        for label in labels:
            query = "UPDATE activity SET transportation_mode = '%s' WHERE start_date_time = '%s' AND end_date_time = '%s' AND user_id = '%s'"
            self.cursor.execute(query % (label["transportation_mode"], label["start_time"], label["end_time"], label["user_id"]))
            self.db_connection.commit()

    ########################################
    ## TASK 2 QUERIES   ####################
    ########################################

    def total_distance(self):
        query = """SELECT lat, lon FROM trackpoint WHERE activity_id IN 
        (SELECT id FROM activity WHERE user_id = 112 AND transportation_mode = 'walk' 
        AND YEAR(start_date_time)='2008')"""
        self.cursor.execute(query)
        coordinates = self.cursor.fetchall()
        total_distance = 0
        for index in range(len(coordinates)):
            if index == len(coordinates)-1:
                break
            distance = haversine(coordinates[index], coordinates[index+1])
            total_distance += distance
        print(total_distance)    
    
    def most_altitude_gained(self):
        query = """SELECT act1.user_id, SUM(altitude_gained) * 0.3048 
                FROM 
                    (SELECT tp1.activity_id AS act_id, 
                    SUM(t2.altitude - tp1.altitude) AS altitude_gained 
                    FROM trackpoint 
                    AS tp1 JOIN trackpoint AS t2 
                    ON tp1.id = t2.id - 1 
                    WHERE t2.altitude > tp1.altitude 
                    GROUP BY tp1.activity_id) AS t, 
                activity AS act1 WHERE act_id = act1.id 
                GROUP BY act1.user_id 
                ORDER BY SUM(altitude_gained) 
                DESC LIMIT 20"""
        self.cursor.execute(query)
        distances = self.cursor.fetchall()
        for i, info in enumerate(distances):
            print(i,"||", info[0],"|", info[1], "m")
    
    def find_users_with_transportation_mode(self):
        query = """SELECT user_id, transportation_mode, COUNT(transportation_mode) 
                FROM activity 
                WHERE transportation_mode IS NOT NULL
                GROUP BY user_id, transportation_mode 
                ORDER BY user_id, COUNT(transportation_mode) DESC"""
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        already_printed = []
        print("id ","|", "transportation_mode")
        print("----------------------------")
        for i, info in enumerate(result):
            if info[0] not in already_printed:
                print(info[0],"|", info[1])
                already_printed.append(info[0])
        return result
def main():
    db_connector = DbConnector()
    db_connector.find_users_with_transportation_mode()

main()