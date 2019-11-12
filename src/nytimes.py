import sqlite3
from pathlib import Path

import spacy

from tqdm import tqdm

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

db_file = "/Users/lorajohns/Development/GitHub_Repos/nytimes_copy/nytimes.db"
path = Path(db_file).resolve()

def select_articles(conn):

    """
    Get sample articles from database
    """
    cur = conn.cursor()
    cur.execute(f"SELECT content FROM articles WHERE id % 10 < 1;")

    rows = cur.fetchall()
    return rows

if __name__ == "__main__":
    conn = create_connection(path)
    samp = select_articles(conn)
    texts = [samp[i][0] for i, article in enumerate(samp)]
    nlp = spacy.load("en_core_web_md")
    docs = tqdm(nlp.pipe(texts[:15]))
    
    for doc in docs:
        for ent in doc.ents:
            print(ent, ent.label_)
