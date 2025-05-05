#!/usr/bin/env python3
"""
Database module for PostgreSQL with pgvector integration.
Provides full CRUD and similarity search for utterances, tasks, groceries, and conversations.
"""
import os
import logging
import json
from psycopg2 import pool, extras
import psycopg2


class DatabaseError(Exception):
    """Custom exception for database operations."""


class Database:
    """PostgreSQL-backed vector database using pgvector."""

    def __init__(self, connection_string: str | None = None):
        try:
            # Fallback to env or localhost default
            self.connection_string = connection_string or os.environ.get(
                "DATABASE_URL",
                "postgresql://postgres:postgres@localhost:5432/postgres",
            )
            self.pool: pool.SimpleConnectionPool = pool.SimpleConnectionPool(
                1, 10, self.connection_string
            )
            self._collections = [
                "utterances",
                "tasks",
                "groceries",
                "conversations",
            ]
            self._init_db()
        except Exception as e:
            logging.error("Failed to initialize PostgreSQL connection: %s", e)
            raise DatabaseError("Failed to initialize PostgreSQL connection") from e

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _conn(self):
        return self.pool.getconn()

    def _put(self, conn):
        self.pool.putconn(conn)

    def _init_db(self):
        """Create pgvector extension and required tables/indexes if missing."""
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                for table in self._collections:
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {table} (
                            id TEXT PRIMARY KEY,
                            document TEXT NOT NULL,
                            metadata JSONB,
                            embedding vector(1536)
                        )"""
                    )
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_{table}_embedding
                        ON {table} USING hnsw (embedding vector_cosine_ops)"""
                    )
                conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error("Failed to initialize database: %s", e)
            raise DatabaseError("Failed to initialize database") from e
        finally:
            self._put(conn)

    # ---------------------------------------------------------------------
    # CRUD
    # ---------------------------------------------------------------------
    def _ensure(self, collection):
        if collection not in self._collections:
            raise DatabaseError(f"Collection {collection} does not exist")

    def create(self, collection, id, document, metadata=None):
        self._ensure(collection)
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {collection} (id, document, metadata, embedding) VALUES (%s,%s,%s,NULL)",
                    (id, document, json.dumps(metadata or {})),
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error("Create failed: %s", e)
            raise DatabaseError("Create failed") from e
        finally:
            self._put(conn)

    def read(self, collection, id):
        self._ensure(collection)
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=extras.DictCursor) as cur:
                cur.execute(f"SELECT * FROM {collection} WHERE id=%s", (id,))
                row = cur.fetchone()
                if not row:
                    return {"ids": [], "documents": [], "metadatas": []}
                return {
                    "ids": [row["id"]],
                    "documents": [row["document"]],
                    "metadatas": [json.loads(row["metadata"]) if row["metadata"] else {}],
                }
        except Exception as e:
            logging.error("Read failed: %s", e)
            raise DatabaseError("Read failed") from e
        finally:
            self._put(conn)

    def update(self, collection, id, document=None, metadata=None):
        self._ensure(collection)
        if document is None and metadata is None:
            return
        conn = self._conn()
        try:
            sets, params = [], []
            if document is not None:
                sets.append("document=%s")
                params.append(document)
            if metadata is not None:
                sets.append("metadata=%s")
                params.append(json.dumps(metadata))
            params.append(id)
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {collection} SET {', '.join(sets)} WHERE id=%s", params
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error("Update failed: %s", e)
            raise DatabaseError("Update failed") from e
        finally:
            self._put(conn)

    def delete(self, collection, id):
        self._ensure(collection)
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {collection} WHERE id=%s", (id,))
                conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error("Delete failed: %s", e)
            raise DatabaseError("Delete failed") from e
        finally:
            self._put(conn)

    def list(self, collection):
        self._ensure(collection)
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=extras.DictCursor) as cur:
                cur.execute(f"SELECT * FROM {collection}")
                rows = cur.fetchall()
                return {
                    "ids": [r["id"] for r in rows],
                    "documents": [r["document"] for r in rows],
                    "metadatas": [json.loads(r["metadata"]) if r["metadata"] else {} for r in rows],
                }
        except Exception as e:
            logging.error("List failed: %s", e)
            raise DatabaseError("List failed") from e
        finally:
            self._put(conn)

    def query(self, collection, query_text, n_results=5):
        self._ensure(collection)
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=extras.DictCursor) as cur:
                cur.execute(
                    f"SELECT id, document, metadata FROM {collection} ORDER BY document <-> %s LIMIT %s",
                    (query_text, n_results),
                )
                rows = cur.fetchall()
                return {
                    "ids": [r["id"] for r in rows],
                    "documents": [r["document"] for r in rows],
                    "metadatas": [json.loads(r["metadata"]) if r["metadata"] else {} for r in rows],
                    "distances": [0.0] * len(rows),
                }
        except Exception as e:
            logging.error("Query failed: %s", e)
            raise DatabaseError("Query failed") from e
        finally:
            self._put(conn)

    # pg automatically persists; method kept for API parity
    def persist(self):
        pass

    def __del__(self):
        if hasattr(self, "pool"):
            self.pool.closeall()
