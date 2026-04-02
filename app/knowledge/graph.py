# app/knowledge/graph.py
"""
SQLite-backed knowledge graph for Indian healthcare data.

Provides instant, structured lookups for common healthcare queries
without LLM inference. This is the competitive moat — any competitor
can download Qwen3, but not this curated Indian healthcare dataset.

Design:
- SQLite for zero-dependency, portable storage
- Full-text search (FTS5) for fuzzy matching
- JSON data files loaded on startup
- Bilingual (English + Hindi) query support
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class KnowledgeGraph:
    """
    Structured knowledge graph for Indian healthcare data.

    Supports:
    - Exact and fuzzy lookups
    - Category browsing
    - Bilingual queries (English + Hindi)
    - Extensible data loading from JSON files
    """

    def __init__(self, db_path: str = "./data/knowledge.db") -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._load_data()
        logger.info("KnowledgeGraph initialised at %s", db_path)

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        cur = self._conn.cursor()

        cur.executescript("""
            CREATE TABLE IF NOT EXISTS categories (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                name_hi TEXT,
                description TEXT
            );

            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id TEXT NOT NULL,
                key TEXT NOT NULL,
                key_hi TEXT,
                value TEXT NOT NULL,
                value_hi TEXT,
                source TEXT,
                tags TEXT,
                FOREIGN KEY (category_id) REFERENCES categories(id)
            );

            CREATE TABLE IF NOT EXISTS medicines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand_name TEXT NOT NULL,
                generic_name TEXT NOT NULL,
                generic_name_hi TEXT,
                category TEXT,
                jan_aushadhi_price REAL,
                market_price REAL,
                savings_percent REAL,
                usage TEXT
            );

            CREATE TABLE IF NOT EXISTS drug_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drug_a TEXT NOT NULL,
                drug_b TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                recommendation TEXT
            );

            CREATE TABLE IF NOT EXISTS icd10_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symptom TEXT NOT NULL,
                symptom_hi TEXT,
                icd10_code TEXT NOT NULL,
                condition_name TEXT NOT NULL,
                condition_name_hi TEXT,
                severity TEXT,
                see_doctor_urgency TEXT
            );
        """)

        # Full-text search indexes
        cur.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                key, value, tags, content=facts, content_rowid=id
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS medicines_fts USING fts5(
                brand_name, generic_name, category, usage,
                content=medicines, content_rowid=id
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS icd10_fts USING fts5(
                symptom, condition_name,
                content=icd10_map, content_rowid=id
            );
        """)

        self._conn.commit()

    # ── Data Loading ───────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Load JSON data files into SQLite if tables are empty."""
        cur = self._conn.cursor()

        # Check if already loaded
        cur.execute("SELECT COUNT(*) FROM facts")
        if cur.fetchone()[0] > 0:
            logger.info("Knowledge data already loaded — skipping")
            return

        logger.info("Loading knowledge data from %s", DATA_DIR)

        self._load_json("irdai_regulations.json", self._import_facts)
        self._load_json("ayushman_bharat.json", self._import_facts)
        self._load_json("generic_medicines.json", self._import_medicines)
        self._load_json("drug_interactions.json", self._import_drug_interactions)
        self._load_json("icd10_symptoms.json", self._import_icd10)

        # Rebuild FTS indexes
        cur.executescript("""
            INSERT INTO facts_fts(facts_fts) VALUES('rebuild');
            INSERT INTO medicines_fts(medicines_fts) VALUES('rebuild');
            INSERT INTO icd10_fts(icd10_fts) VALUES('rebuild');
        """)
        self._conn.commit()
        logger.info("✅ Knowledge data loaded successfully")

    def _load_json(self, filename: str, importer) -> None:
        """Load a single JSON file and import with the given function."""
        filepath = DATA_DIR / filename
        if not filepath.exists():
            logger.warning("Data file not found: %s", filepath)
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            importer(data)
            logger.info("  Loaded %s", filename)
        except Exception:
            logger.exception("  Failed to load %s", filename)

    def _import_facts(self, data: dict) -> None:
        """Import facts from IRDAI/PMJAY JSON format."""
        cur = self._conn.cursor()
        category = data.get("category", {})
        cat_id = category.get("id", "unknown")

        cur.execute(
            "INSERT OR REPLACE INTO categories (id, name, name_hi, description) VALUES (?, ?, ?, ?)",
            (cat_id, category.get("name"), category.get("name_hi"), category.get("description")),
        )

        for fact in data.get("facts", []):
            cur.execute(
                "INSERT INTO facts (category_id, key, key_hi, value, value_hi, source, tags) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    cat_id,
                    fact.get("key"),
                    fact.get("key_hi"),
                    fact.get("value"),
                    fact.get("value_hi"),
                    fact.get("source"),
                    json.dumps(fact.get("tags", [])),
                ),
            )
        self._conn.commit()

    def _import_medicines(self, data: dict) -> None:
        """Import generic medicine data."""
        cur = self._conn.cursor()
        for med in data.get("medicines", []):
            cur.execute(
                "INSERT INTO medicines (brand_name, generic_name, generic_name_hi, "
                "category, jan_aushadhi_price, market_price, savings_percent, usage) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    med.get("brand_name"),
                    med.get("generic_name"),
                    med.get("generic_name_hi"),
                    med.get("category"),
                    med.get("jan_aushadhi_price"),
                    med.get("market_price"),
                    med.get("savings_percent"),
                    med.get("usage"),
                ),
            )
        self._conn.commit()

    def _import_drug_interactions(self, data: dict) -> None:
        """Import drug interaction data."""
        cur = self._conn.cursor()
        for interaction in data.get("interactions", []):
            cur.execute(
                "INSERT INTO drug_interactions (drug_a, drug_b, severity, description, recommendation) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    interaction.get("drug_a"),
                    interaction.get("drug_b"),
                    interaction.get("severity"),
                    interaction.get("description"),
                    interaction.get("recommendation"),
                ),
            )
        self._conn.commit()

    def _import_icd10(self, data: dict) -> None:
        """Import ICD-10 symptom mapping data."""
        cur = self._conn.cursor()
        for entry in data.get("mappings", []):
            cur.execute(
                "INSERT INTO icd10_map (symptom, symptom_hi, icd10_code, "
                "condition_name, condition_name_hi, severity, see_doctor_urgency) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.get("symptom"),
                    entry.get("symptom_hi"),
                    entry.get("icd10_code"),
                    entry.get("condition_name"),
                    entry.get("condition_name_hi"),
                    entry.get("severity"),
                    entry.get("see_doctor_urgency"),
                ),
            )
        self._conn.commit()

    # ── Query Methods ──────────────────────────────────────────────────────────

    def query(self, question: str) -> Optional[str]:
        """
        Try to answer a question from structured knowledge.

        Returns a formatted answer string, or None if no relevant data found.
        """
        q_lower = question.lower()

        # Try each knowledge domain
        answer = (
            self._query_facts(q_lower)
            or self._query_medicines(q_lower)
            or self._query_drug_interactions(q_lower)
            or self._query_icd10(q_lower)
        )

        return answer

    def _query_facts(self, query: str) -> Optional[str]:
        """Search IRDAI/PMJAY facts using full-text search."""
        cur = self._conn.cursor()
        # Clean query for FTS
        fts_query = " OR ".join(
            w for w in query.split()
            if len(w) > 2 and w not in ("what", "is", "the", "for", "how", "does", "are", "can", "about")
        )
        if not fts_query:
            return None

        try:
            cur.execute(
                "SELECT f.key, f.value, f.value_hi, f.source, c.name "
                "FROM facts_fts fts "
                "JOIN facts f ON fts.rowid = f.id "
                "JOIN categories c ON f.category_id = c.id "
                "WHERE facts_fts MATCH ? "
                "ORDER BY rank LIMIT 3",
                (fts_query,),
            )
            rows = cur.fetchall()
            if not rows:
                return None

            parts = []
            for row in rows:
                parts.append(f"**{row['key']}** ({row['name']})\n{row['value']}")
                if row["source"]:
                    parts[-1] += f"\n_Source: {row['source']}_"

            return "\n\n".join(parts)
        except Exception:
            return None

    def _query_medicines(self, query: str) -> Optional[str]:
        """Search generic medicine alternatives."""
        cur = self._conn.cursor()

        # Direct brand name search first
        cur.execute(
            "SELECT * FROM medicines WHERE LOWER(brand_name) LIKE ? OR LOWER(generic_name) LIKE ? LIMIT 5",
            (f"%{query}%", f"%{query}%"),
        )
        rows = cur.fetchall()

        if not rows:
            # Try FTS
            fts_query = " OR ".join(w for w in query.split() if len(w) > 2)
            if not fts_query:
                return None
            try:
                cur.execute(
                    "SELECT m.* FROM medicines_fts fts "
                    "JOIN medicines m ON fts.rowid = m.id "
                    "WHERE medicines_fts MATCH ? LIMIT 5",
                    (fts_query,),
                )
                rows = cur.fetchall()
            except Exception:
                return None

        if not rows:
            return None

        parts = ["**Generic Medicine Alternatives:**\n"]
        for row in rows:
            line = f"• **{row['brand_name']}** → Generic: **{row['generic_name']}**"
            if row["jan_aushadhi_price"] and row["market_price"]:
                line += (
                    f"\n  Market: ₹{row['market_price']:.0f} → "
                    f"Jan Aushadhi: ₹{row['jan_aushadhi_price']:.0f} "
                    f"(Save {row['savings_percent']:.0f}%)"
                )
            if row["usage"]:
                line += f"\n  Usage: {row['usage']}"
            parts.append(line)

        return "\n".join(parts)

    def _query_drug_interactions(self, query: str) -> Optional[str]:
        """Search for drug interaction warnings."""
        cur = self._conn.cursor()

        cur.execute(
            "SELECT * FROM drug_interactions "
            "WHERE LOWER(drug_a) LIKE ? OR LOWER(drug_b) LIKE ? LIMIT 5",
            (f"%{query}%", f"%{query}%"),
        )
        rows = cur.fetchall()

        if not rows:
            return None

        parts = ["⚠️ **Drug Interaction Warnings:**\n"]
        for row in rows:
            severity_icon = {"severe": "🔴", "moderate": "🟡", "mild": "🟢"}.get(
                row["severity"], "⚪"
            )
            parts.append(
                f"{severity_icon} **{row['drug_a']}** + **{row['drug_b']}** "
                f"({row['severity'].upper()})\n"
                f"  {row['description']}\n"
                f"  _Recommendation: {row['recommendation']}_"
            )

        return "\n\n".join(parts)

    def _query_icd10(self, query: str) -> Optional[str]:
        """Search symptom → condition mapping."""
        cur = self._conn.cursor()

        # Direct symptom search
        cur.execute(
            "SELECT * FROM icd10_map WHERE LOWER(symptom) LIKE ? LIMIT 5",
            (f"%{query}%",),
        )
        rows = cur.fetchall()

        if not rows:
            fts_query = " OR ".join(w for w in query.split() if len(w) > 3)
            if not fts_query:
                return None
            try:
                cur.execute(
                    "SELECT m.* FROM icd10_fts fts "
                    "JOIN icd10_map m ON fts.rowid = m.id "
                    "WHERE icd10_fts MATCH ? LIMIT 5",
                    (fts_query,),
                )
                rows = cur.fetchall()
            except Exception:
                return None

        if not rows:
            return None

        parts = ["**Possible conditions based on symptoms:**\n"]
        for row in rows:
            urgency_icon = {
                "immediate": "🔴",
                "within_24h": "🟠",
                "within_week": "🟡",
                "routine": "🟢",
            }.get(row["see_doctor_urgency"], "⚪")
            parts.append(
                f"{urgency_icon} **{row['condition_name']}** (ICD-10: {row['icd10_code']})\n"
                f"  Symptom: {row['symptom']} | Severity: {row['severity']}"
            )

        parts.append(
            "\n⚠️ *This is for informational reference only. "
            "Please consult a qualified healthcare professional for proper diagnosis.*"
        )
        return "\n".join(parts)

    # ── Utility ────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        """Return counts of data in each table."""
        cur = self._conn.cursor()
        stats = {}
        for table in ("categories", "facts", "medicines", "drug_interactions", "icd10_map"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cur.fetchone()[0]
        return stats

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
