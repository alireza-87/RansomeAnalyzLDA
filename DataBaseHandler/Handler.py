#Alireza.karimi.67@gmail.com
import json
import sqlite3


class DatabaseHandler():
    need_upgrade = False
    conn = None

    def __init__(self):
        self.conn = sqlite3.connect('DataBase.db')
        if self.need_upgrade:
            self.upgrade_tables()
        else:
            self.create_tables()

    def create_tables(self):
        cursor=self.conn.cursor()
        cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Sample(id INTEGER PRIMARY KEY,
                                       SampleHash TEXT,Lable TEXT)
                ''')
        self.conn.commit()

    def upgrade_tables(self):
        cursor=self.conn.cursor()
        cursor.execute('''DROP TABLE Sample''')
        self.create_tables()

    def clear_table_Dataset(self):
        cursor = self.conn.cursor()
        cursor.execute('''DROP TABLE Dataset''')
        cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Dataset(id INTEGER PRIMARY KEY, nodefrom INTEGER,
                                       nodeto INTEGER, nodefromapi TEXT, nodetoapi TEXT,nodeweight INTEGER,sampleid TEXT)
                ''')
        self.conn.commit()

    def insert_a_sample(self,sample_hash,isMal):
        self.conn.execute('''INSERT INTO Sample(SampleHash,Lable)
                      VALUES(?,?)''', (sample_hash,str(isMal)))
        self.conn.commit()

    def select_sample_all(self):
        cursor=self.conn.cursor()
        query=cursor.execute('SELECT * FROM Sample')
        samples=[]
        for row in query:
            samples.append(row)
        return samples

    def select_sample(self,sample_hash):
        cursor=self.conn.cursor()
        query=cursor.execute('SELECT * FROM Sample WHERE SampleHash=?',[sample_hash])
        return query.fetchone()

    def update_sample_lable(self,sample_id,lable):
        cursor = self.conn.cursor()
        cursor.execute('''UPDATE Sample SET Lable=? WHERE SampleHash LIKE ?''', (lable, sample_id))
        self.conn.commit()
        return cursor.rowcount