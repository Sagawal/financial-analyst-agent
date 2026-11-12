
import os
import json

import psycopg
from pgvector.psycopg import register_vector
from langchain_core.documents import Document
from dotenv import load_dotenv

from rag.vector_store import generate_chunk_id, sanitize_metadata
from rag.embeddings import BGEEmbeddings

from pgvector import Vector


load_dotenv()

TABLE_NAME = "financial_document_chunks"


class SupabaseFinancialVectorStore:

    def __init__(self, embedding_model: BGEEmbeddings):
        self.embedding_model = embedding_model

        self.database_url = os.getenv("SUPABASE_DB_URL")

        if not self.database_url:
            raise ValueError("SUPABASE_DB_URL is missing from .env")

    def _get_connection(self):
        conn = psycopg.connect(
            self.database_url,
            connect_timeout=15
        )
        register_vector(conn)
        return conn

    def add_documents(self, documents: list[Document]) -> list[str]:

        if not documents:
            return []

        ids = [generate_chunk_id(doc) for doc in documents]

        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)

        records = []

        for doc_id, doc, embedding in zip(ids, documents, embeddings):
            metadata = sanitize_metadata(doc.metadata)

            records.append((
                doc_id,
                doc.page_content,
                json.dumps(metadata),
                Vector(embedding)
            ))

        query = f"""
            INSERT INTO {TABLE_NAME}
                (id, content, metadata, embedding)
            VALUES (%s, %s, %s::jsonb, %s)
            ON CONFLICT (id)
            DO UPDATE SET
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding;
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, records)

        return ids

    @staticmethod
    def _build_metadata_filter(filters: dict | None):

        if not filters:
            return "", []

        conditions = []

        def parse_filter(filter_dict):

            for key, value in filter_dict.items():

                if key == "$and":
                    for condition in value:
                        parse_filter(condition)

                elif isinstance(value, dict) and "$eq" in value:
                    conditions.append((key, value["$eq"]))

                else:
                    conditions.append((key, value))

        parse_filter(filters)

        sql_conditions = [
            "metadata @> %s::jsonb"
            for _ in conditions
        ]

        parameters = [
            json.dumps({key: value})
            for key, value in conditions
        ]

        return " AND ".join(sql_conditions), parameters

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filters: dict | None = None
    ):

        query_embedding = Vector(
            self.embedding_model.embed_query(query)
        )

        filter_sql, filter_params = self._build_metadata_filter(filters)

        sql = f"""
            SELECT
                id,
                content,
                metadata,
                embedding <=> %s AS distance
            FROM {TABLE_NAME}
            {"WHERE " + filter_sql if filter_sql else ""}
            ORDER BY embedding <=> %s
            LIMIT %s;
        """

        params = [
            query_embedding,
            *filter_params,
            query_embedding,
            k
        ]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results = []

        for row in rows:
            _, content, metadata, distance = row

            document = Document(
                page_content=content,
                metadata=metadata
            )

            results.append((document, float(distance)))

        return results

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: dict | None = None
    ) -> list[Document]:

        results = self.similarity_search_with_score(
            query=query,
            k=k,
            filters=filters
        )

        return [document for document, _ in results]

    def get_collection_count(self) -> int:

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {TABLE_NAME}"
                )
                return cur.fetchone()[0]
