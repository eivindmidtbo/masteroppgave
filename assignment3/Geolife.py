from DbConnector import DbConnector
from Process_data import Process_data
import pprint as pp
from haversine import haversine
from datetime import datetime


COLLECTIONS = ["user", "activity", "trackpoint"]


def insert_data(db_connector):
    # Instantiating fresh collections
    for collection in COLLECTIONS:
        db_connector.remove_collection(collection)
        db_connector.create_collection(collection)
    db_connector.create_indexes()

    process_data = Process_data(db_connector)

    # Inserting data
    process_data.process()

def check_if_invalid(date_time1, date_time2):
    diff = (date_time2-date_time1).total_seconds()
    if diff >= 300:
        return True
    else:
        return False

def main():
    db_connector = DbConnector()

    ## Uncomment when running first time
    # insert_data(db_connector)

    # Querying data
    def task_2():
        task2 = db_connector.find_average_activities_per_user()
        print("Average activities per user: %s" % list(task2)[0]["avg"])

    def task_3():
        task3 = db_connector.find_top_20_users_with_most_activities()
        print("User_id  | Activity count")
        for user in list(task3):
            print("{:8} | {:8}".format(user["_id"], user["count"]))

    def task_4():
        task4 = db_connector.find_users_who_have_taken_taxi()
        print("Users who have taken a taxi:")
        for i in task4:
            print(i["_id"])

    def task_5():
        task5 = db_connector.find_all_types_of_transportation_modes()
        print("Transportation mode | Count")
        for i in task5:
            print("{:19} | {:8}".format(i["_id"], i["count"]))

    def task_6a():
        task6a = db_connector.find_year_with_most_activities()
        print("Year with most activities: %s" % list(task6a)[0]["_id"])

    def task_6b():
        task6b = db_connector.find_year_with_most_recorded_hours()
        print("Year with most recorded hours: %s" % list(task6b)[0]["_id"])

    def task_7():
        activities = db_connector.find_activity_ids()
        distance = 0
        is_first = True
        for i in activities:
            trackpoints = db_connector.find_trackpoints_of_activity(i["_id"])
            for tp in trackpoints:
                if is_first:
                    old = (tp["location"]["coordinates"][1],
                           tp["location"]["coordinates"][0])
                    is_first = False
                else:
                    new = (tp["location"]["coordinates"][1],
                           tp["location"]["coordinates"][0])
                    distance += haversine(old, new)
                    old = new
        print("TOTAL DISTANCE: ", distance)

    def task_8():
        print("Task 8")

        user_ids = db_connector.find_all_user_ids()
        user_activity_ids = {}
        for user in user_ids:
            activities = db_connector.find_all_activity_ids_of_user(
                user["_id"])
            activity_ids = []
            for a in activities:
                activity_ids.append(a["_id"])
            user_activity_ids[user["_id"]] = activity_ids

        user_altitude = {}
        for user in user_activity_ids:
            user_altitude[user] = 0
            for activity in user_activity_ids[user]:
                activity_alt_gained = 0
                old_alt = -1000
                altitudes = db_connector.find_altitude_of_activity(activity)
                for alt in altitudes:
                    if alt["altitude"] == -777:
                        continue

                    new_alt = alt["altitude"]
                    if old_alt == -1000:
                        old_alt = new_alt
                    if new_alt > old_alt:
                        activity_alt_gained += new_alt - old_alt
                    old_alt = new_alt
                user_altitude[user] += activity_alt_gained*0.3048
                activity_alt_gained = 0
                
        sorted_user_altitudes = sorted(
            user_altitude.items(), key=lambda x: x[1], reverse=True)
        print("User_id  | Altitude gained")
        count = 1
        for key, altitude in sorted_user_altitudes:
            if count > 20:
                break
            count += 1
            print("{:8} | {:8}".format(key, round(altitude)))

    def task_9():
        user_ids = db_connector.find_all_user_ids()
        user_invalid_activities = {}
        for user in user_ids:
            print("Finding invalid actitities for user: " + user["_id"])
            activities = db_connector.find_all_activity_ids_of_user(
                user["_id"])
            user_invalid_activities[user["_id"]] = 0
            for a in activities:
                prev_tp = "first"
                for tp in db_connector.find_date_time_of_activity(a["_id"]):
                    if prev_tp == "first":
                        prev_tp = tp
                    else:
                        if check_if_invalid(prev_tp["date_time"], tp["date_time"]) == True:
                            user_invalid_activities[user["_id"]] += 1
                            break
                        prev_tp = tp
        sorted_user_invalid_activities = sorted(
            user_invalid_activities.items(), key=lambda x: x[1], reverse=True)
        print("User_id  | Invalid activities")
        for key, count in sorted_user_invalid_activities:
            if count != 0:
                print("{:8} | {:8}".format(key, count))

    def task_10():
        task10 = db_connector.find_users_with_activity_with_trackpoint_at_location()
        print("Users that have been to the Forbidden City of Beijing.")
        for i in task10:
            print(i["_id"])

    def task_11():
        task11 = db_connector.find_most_frequent_transportation_mode_for_each_user()
        print("User_id | Transportation mode")
        for i in task11:
            print("{:7} | {:19}".format(i["_id"], i["transportation_mode"]))

    # Uncomment to run task
    task_2()
    task_3()
    #task_4()
    #task_5()
    #task_6a()
    #task_6b()
    #task_7()
    #task_8()
    #task_9()
    #task_10()
    #task_11()
main()
