
from DbConnector import DbConnector
from Process_data import Process_data

TABLES = {}
TABLES['user'] = (
    "CREATE TABLE IF NOT EXISTS `user` ("
    "  `id` VARCHAR(5) NOT NULL,"
    "  `has_labels` BOOLEAN,"
    "  PRIMARY KEY (`id`)"
    ") ENGINE=InnoDB"
)
TABLES['activity'] = (
    "CREATE TABLE IF NOT EXISTS `activity` ("
    "  `id` INT NOT NULL AUTO_INCREMENT,"
    "  `user_id` VARCHAR(5),"
    "  `transportation_mode` VARCHAR(15),"
    "  `start_date_time` DATETIME,"
    "  `end_date_time` DATETIME,"
    "  PRIMARY KEY (`id`),"
    "  CONSTRAINT `user_fk` FOREIGN KEY(`user_id`) REFERENCES user(`id`) ON DELETE CASCADE"
    ") ENGINE=InnoDB"
)
TABLES['trackpoint'] = (
    "CREATE TABLE IF NOT EXISTS `trackpoint` ("
    "  `id` INT NOT NULL AUTO_INCREMENT,"
    "  `activity_id` INT,"
    "  `lat` DOUBLE,"
    "  `lon` DOUBLE,"
    "  `altitude` INT,"
    "  `date_days` DOUBLE,"
    "  `date_time` DATETIME,"
    "  PRIMARY KEY (`id`),"
    "  CONSTRAINT `activity_fk` FOREIGN KEY (`activity_id`) REFERENCES `activity` (`id`) ON DELETE CASCADE ON UPDATE CASCADE"
    ") ENGINE=InnoDB"
)


def main():
    print("Hello World!")
    db_connector = DbConnector()
    print(db_connector)
    process_data = Process_data(db_connector)

    # Drop tables
    for table_name in reversed(list(TABLES.keys())):
        db_connector.drop_table(table_name)

    # Creating tables
    for table in TABLES.values():
        db_connector.create_Table(table)

    # Inserting data
    process_data.process()

main()
