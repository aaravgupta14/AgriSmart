"""Thin PyMySQL helper - one connection per request (stored on flask.g)."""
import pymysql
from flask import g

import config


def get_db():
    if "db" not in g:
        g.db = pymysql.connect(**config.DB, cursorclass=pymysql.cursors.DictCursor,
                               autocommit=True, charset="utf8mb4")
    return g.db


def close_db(_=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def query(sql, args=None, one=False):
    with get_db().cursor() as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()
    return (rows[0] if rows else None) if one else rows


def execute(sql, args=None):
    with get_db().cursor() as cur:
        cur.execute(sql, args)
        return cur.lastrowid


def callproc(name, args=()):
    """Call a stored procedure and return ALL of its result sets."""
    out = []
    with get_db().cursor() as cur:
        cur.callproc(name, args)
        while True:
            rows = cur.fetchall()
            if cur.description:
                out.append(list(rows))
            if not cur.nextset():
                break
    return out
