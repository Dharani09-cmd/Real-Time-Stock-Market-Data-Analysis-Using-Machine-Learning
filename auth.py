import sqlite3, hashlib

def connect_db():
    return sqlite3.connect("users.db", check_same_thread=False)

def create_table():
    conn=connect_db()
    c=conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
    conn.commit(); conn.close()

def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def signup(u,p):
    conn=connect_db(); c=conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?,?)",(u,hash_password(p)))
        conn.commit(); return True
    except: return False
    finally: conn.close()

def login(u,p):
    conn=connect_db(); c=conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?",(u,hash_password(p)))
    r=c.fetchone(); conn.close()
    return r is not None
