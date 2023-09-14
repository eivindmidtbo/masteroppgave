import os
import pandas as pd
import numpy as np
import time
import datetime as dt
BASE_PATH = "./dataset"


class Process_data:
    def __init__(self, db_connector):
        self.users = {}
        self.users_with_labels = []
        self.activities = []
        self.activity_id_counter = 1
        self.trackpoint_id_counter = 1
        self.db_connector = db_connector
        self.labels = []

    # Read single user
    def read_user(self, user_id):
        path = BASE_PATH + "/Data"
        self.users[user_id] = {}
        for root, dirs, files in os.walk(path+"/"+user_id):
            root = root.replace("\\", "/")  # For windows
            if "/Trajectory" in root:
                userInfo = {
                    "_id": user_id,
                    "path": root,
                    "used_files": [],
                    "has_label": self.user_has_label(user_id),
                    "labels": {}
                }
                self.users[user_id].update(userInfo)

    def read_user_activities(self, user_id):
        user = self.users[user_id]
        for filename in os.listdir(user["path"]):
            if filename.endswith(".plt"):
                path = user["path"] + "/" + filename
                activity_trackpoints = np.genfromtxt(
                    path, skip_header=6, delimiter=',', dtype=str, usecols=(0, 1, 3, 5, 6))
                if (len(activity_trackpoints) > 2500):
                    continue
                res = self.match_label(activity_trackpoints, user)
                activity = dict(_id=int(self.activity_id_counter),
                                user_id=str(user_id),
                                transportation_mode=res,
                                start_date_time=self.string_to_datetime(activity_trackpoints[0][3] +
                                " " + activity_trackpoints[0][4]),
                                end_date_time=self.string_to_datetime(activity_trackpoints[-1][3] +
                                " " + activity_trackpoints[-1][4])
                                )
                self.activities.append(activity)

                trackpoints = []
                for trackpoint in activity_trackpoints:
                    trackpoint_date_time = trackpoint[3] + \
                        " " + trackpoint[4]
                    altitude = trackpoint[2]

                    # Fixing invalid altitudes
                    if altitude == -777:
                        altitude = 0
                    formatted_trackpoint = dict(_id=int(self.trackpoint_id_counter),
                                                activity_id=int(
                                                    self.activity_id_counter),
                                                location=dict(type="Point",
                                                              coordinates=[float(trackpoint[1]), float(trackpoint[0])]),
                                                altitude=int(float(altitude)),
                                                date_time=self.string_to_datetime(trackpoint_date_time))
                    self.trackpoint_id_counter += 1
                    trackpoints.append(formatted_trackpoint)
                self.db_connector.insert_trackpoints_with_id(trackpoints)
                self.activity_id_counter += 1

    # Find all users with labels

    def find_users_with_labels(self):
        path = BASE_PATH + "/labeled_ids.txt"
        labeled_users = pd.read_csv(path, header=None).to_numpy()
        formatted = []
        for element in labeled_users:
            formatted.append('{:03}'.format(element[0]))
        self.users_with_labels = formatted

    # Take in labels and activity, check if finds a match
    def match_label(self, activity, user):
        if (user["has_label"]):
            first_trackpoint = activity[0]
            last_trackpoint = activity[-1]
            starttime = first_trackpoint[3] + \
                " " + first_trackpoint[4]
            endtime = last_trackpoint[3] + \
                " " + last_trackpoint[4]
            labels = user["labels"]
            if starttime in labels:
                if endtime == labels[starttime]["end_time"]:
                    print("Found match " +
                          labels[starttime]["transportation_mode"])
                    return labels[starttime]["transportation_mode"]
        return None

    # Read labels, return labels as a dictionary
    def read_labels(self, user_id):
        user = self.users[user_id]
        label_dict = {}
        if user["has_label"]:
            with open(user["path"].replace("/Trajectory", "/labels.txt"), "r") as file:
                labels = file.readlines()[1:]
                for line in labels:
                    label = line.split()
                    start = self.convert_timeformat(label[0] + " " + label[1])
                    end = self.convert_timeformat(label[2] + " " + label[3])
                    label_dict[start] = {"end_time": end,
                                         "transportation_mode": label[4]}
                    new_label = {
                        "start_time": start,
                        "end_time": end,
                        "transportation_mode": label[4],
                        "user_id": user_id
                    }
                    self.labels.append(new_label)
            self.users[user_id]["labels"] = label_dict

    def convert_timeformat(self, date):
        return date.replace("/", "-")

    def string_to_datetime(self, date_string):
        return dt.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

    def user_has_label(self, userID):
        if len(self.users_with_labels) == 0:
            self.read_labeled_users()
        return userID in self.users_with_labels

    # Read all users and save data
    def read_all_users(self):
        self.find_users_with_labels()
        path = BASE_PATH + "/Data"
        self.users = {}
        userIDs = []
        for file in os.listdir(path):
            userIDs.append(os.path.join(path, file)[-3:])
        for userID in userIDs:
            self.read_user(userID)

    def process(self):
        start_time = time.time()
        self.read_all_users()
        # Reformat users
        user_list = []
        for user in self.users.values():
            if (user.get("_id")):
                user_list.append(
                    dict(_id=user["_id"], has_labels=int(user["has_label"])))

        self.db_connector.batch_insert_users(user_list)
        #shouldRead = False
        for user in self.users.values():
            try:
                # if user["_id"] == "024":
                #    shouldRead = True
                # if shouldRead:
                #    print(user)
                print("reading user: " + user["_id"])
                if (user["has_label"]):
                    self.read_labels(user["_id"])
                self.read_user_activities(user["_id"])
            # Fixes error with empty user objects
            except Exception as e:
                print(user)
                print(e)
        self.db_connector.batch_insert_activities_with_id(self.activities)
        self.db_connector.close_connection()
        print("Finished processing in --- %s seconds ---" %
              (time.time() - start_time))
