import mysql.connector
from sys import platform

def connect():
    """
    Establishes a connection to a MySQL server, checks for system type.
    
    Returns:
        mysql.connector.connection.MySQLConnection: A connection object to the MySQL database.
    """
    if platform == "linux" or platform == "linux2":
        return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="max",
        auth_plugin='mysql_native_password'
        )
    elif platform == "win32" or platform == "win64":
        return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="max"
        ) 