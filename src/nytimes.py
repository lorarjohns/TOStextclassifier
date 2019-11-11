import sqlite3

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Exception as e:
        print(e)

    return conn

db_file = "nytimes.db"

def select_articles(pct):
    """
    """
    cur = conn.cursor()
    cur.execute(f"SELECT id, content FROM articles WHERE id % 10 < 2;")

    rows = cur.fetchall()
    return rows

if __name__ == "__main__":
    conn = create_connection("nytimes.db")
    samp = select_articles(2)
    print(samp[0])
