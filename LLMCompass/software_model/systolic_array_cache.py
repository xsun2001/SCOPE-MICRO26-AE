import csv
import hashlib
import json
import os
import sqlite3


def _llmcompass_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SystolicArrayCache:
    _DB_FILENAME = "look_up_table.sqlite3"
    _NAMESPACE_VERSION = 1
    _ARCHITECTURE_TEMPLATE = {
        "IfmapSramSzkB": 1024,
        "FilterSramSzkB": 1024,
        "OfmapSramSzkB": 1024,
        "IfmapOffset": 0,
        "FilterOffset": 10000000,
        "OfmapOffset": 20000000,
        "Bandwidth": 100,
        "MemoryBanks": 1,
    }
    _RUN_PRESETS_TEMPLATE = {
        "InterfaceBandwidth": "CALC",
    }
    _LAYOUT_SECTION_TEMPLATE = {
        "IfmapCustomLayout": False,
        "FilterCustomLayout": False,
        "IfmapSRAMBankBandwidth": 10,
        "IfmapSRAMBankNum": 10,
        "IfmapSRAMBankPort": 2,
        "FilterSRAMBankBandwidth": 10,
        "FilterSRAMBankNum": 10,
        "FilterSRAMBankPort": 2,
    }
    _LAYOUT_FILE_ROW = "matmul1,1,1,1,1,1,1,1,2,3,1,2,3,1,2,3,4,1,2,3,4,"

    def __init__(self, array_height: int, array_width: int):
        self.array_height = int(array_height)
        self.array_width = int(array_width)
        self.namespace_payload = self._build_namespace_payload()
        self.namespace = self._build_namespace()
        cache_path_override = os.environ.get("LLMCOMPASS_SYSTOLIC_CACHE_PATH")
        if cache_path_override:
            self.db_path = os.path.abspath(cache_path_override)
        else:
            self.db_path = os.path.join(
                _llmcompass_root(), "systolic_array_model", self._DB_FILENAME
            )
        self.legacy_csv_path = os.path.join(
            _llmcompass_root(),
            "systolic_array_model",
            f"look_up_table_{self.array_height}_{self.array_width}.csv",
        )
        self.enable_legacy_import = (
            os.environ.get("LLMCOMPASS_SYSTOLIC_CACHE_DISABLE_LEGACY_IMPORT", "")
            .strip()
            .lower()
            not in {"1", "true", "yes", "on"}
        )
        self._conn = None

    def _build_namespace_payload(self) -> dict:
        return {
            "namespace_version": self._NAMESPACE_VERSION,
            "backend": "SCALE-Sim",
            "topology_kind": "gemm",
            "architecture_template": self._ARCHITECTURE_TEMPLATE,
            "layout_section_template": self._LAYOUT_SECTION_TEMPLATE,
            "layout_file_row": self._LAYOUT_FILE_ROW,
            "run_presets_template": self._RUN_PRESETS_TEMPLATE,
            "raw_cycle_scaling": "pre-mac_per_clock",
            "input_type_gemm": True,
        }

    def _build_namespace(self) -> str:
        payload = self.namespace_payload
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        db_exists = os.path.exists(self.db_path) and os.path.getsize(self.db_path) > 0
        conn = sqlite3.connect(
            self.db_path,
            timeout=30,
            isolation_level=None,
        )
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as exc:
            if (not db_exists) or "locked" not in str(exc).lower():
                conn.close()
                raise
        conn.execute("PRAGMA synchronous=NORMAL")
        self._conn = conn
        self._ensure_schema()
        self._register_namespace()
        if self.enable_legacy_import:
            self._import_legacy_csv_if_needed()
        return conn

    def _ensure_schema(self) -> None:
        conn = self._conn
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS systolic_array_results (
                namespace TEXT NOT NULL,
                array_height INTEGER NOT NULL,
                array_width INTEGER NOT NULL,
                dataflow TEXT NOT NULL,
                m INTEGER NOT NULL,
                n INTEGER NOT NULL,
                k INTEGER NOT NULL,
                cycle_count INTEGER NOT NULL,
                util_rate REAL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (namespace, array_height, array_width, dataflow, m, n, k)
            ) WITHOUT ROWID
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS legacy_csv_imports (
                namespace TEXT NOT NULL,
                array_height INTEGER NOT NULL,
                array_width INTEGER NOT NULL,
                source_path TEXT NOT NULL,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (namespace, array_height, array_width, source_path)
            ) WITHOUT ROWID
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_namespaces (
                namespace TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            ) WITHOUT ROWID
            """
        )

    def _register_namespace(self) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO cache_namespaces (
                namespace,
                payload_json
            ) VALUES (?, ?)
            """,
            (
                self.namespace,
                json.dumps(
                    self.namespace_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )

    def _import_legacy_csv_if_needed(self) -> None:
        if not os.path.exists(self.legacy_csv_path):
            return

        conn = self._conn
        imported = conn.execute(
            """
            SELECT 1
            FROM legacy_csv_imports
            WHERE namespace = ? AND array_height = ? AND array_width = ? AND source_path = ?
            """,
            (
                self.namespace,
                self.array_height,
                self.array_width,
                self.legacy_csv_path,
            ),
        ).fetchone()
        if imported is not None:
            return

        rows = []
        with open(self.legacy_csv_path, newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 8:
                    continue
                rows.append(
                    (
                        self.namespace,
                        int(row[3]),
                        int(row[4]),
                        row[5],
                        int(row[0]),
                        int(row[1]),
                        int(row[2]),
                        int(float(row[6])),
                        float(row[7]),
                        "legacy_csv",
                    )
                )

        if not rows:
            conn.execute(
                """
                INSERT OR IGNORE INTO legacy_csv_imports (
                    namespace, array_height, array_width, source_path
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    self.namespace,
                    self.array_height,
                    self.array_width,
                    self.legacy_csv_path,
                ),
            )
            return

        conn.execute("BEGIN IMMEDIATE")
        try:
            imported = conn.execute(
                """
                SELECT 1
                FROM legacy_csv_imports
                WHERE namespace = ? AND array_height = ? AND array_width = ? AND source_path = ?
                """,
                (
                    self.namespace,
                    self.array_height,
                    self.array_width,
                    self.legacy_csv_path,
                ),
            ).fetchone()
            if imported is not None:
                conn.execute("COMMIT")
                return
            conn.executemany(
                """
                INSERT INTO systolic_array_results (
                    namespace,
                    array_height,
                    array_width,
                    dataflow,
                    m,
                    n,
                    k,
                    cycle_count,
                    util_rate,
                    source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, array_height, array_width, dataflow, m, n, k)
                DO UPDATE SET
                    cycle_count = excluded.cycle_count,
                    util_rate = excluded.util_rate,
                    source = excluded.source,
                    updated_at = CURRENT_TIMESTAMP
                """,
                rows,
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO legacy_csv_imports (
                    namespace, array_height, array_width, source_path
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    self.namespace,
                    self.array_height,
                    self.array_width,
                    self.legacy_csv_path,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def fetch(self, m: int, n: int, k: int, dataflow: str):
        conn = self._connect()
        return conn.execute(
            """
            SELECT cycle_count, util_rate
            FROM systolic_array_results
            WHERE namespace = ? AND array_height = ? AND array_width = ?
              AND dataflow = ? AND m = ? AND n = ? AND k = ?
            """,
            (
                self.namespace,
                self.array_height,
                self.array_width,
                dataflow,
                int(m),
                int(n),
                int(k),
            ),
        ).fetchone()

    def upsert(
        self,
        m: int,
        n: int,
        k: int,
        dataflow: str,
        cycle_count: int,
        util_rate,
        source: str,
    ) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO systolic_array_results (
                namespace,
                array_height,
                array_width,
                dataflow,
                m,
                n,
                k,
                cycle_count,
                util_rate,
                source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, array_height, array_width, dataflow, m, n, k)
            DO UPDATE SET
                cycle_count = excluded.cycle_count,
                util_rate = excluded.util_rate,
                source = excluded.source,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                self.namespace,
                self.array_height,
                self.array_width,
                dataflow,
                int(m),
                int(n),
                int(k),
                int(cycle_count),
                None if util_rate is None else float(util_rate),
                source,
            ),
        )

    @classmethod
    def write_scalesim_config(
        cls,
        path: str,
        array_height: int,
        array_width: int,
        dataflow: str,
    ) -> None:
        with open(path, "w") as handle:
            handle.writelines("[general]\n")
            handle.writelines("run_name = systolic_array\n\n")
            handle.writelines("[architecture_presets]\n")
            handle.writelines(f"ArrayHeight:    {array_height}\n")
            handle.writelines(f"ArrayWidth:     {array_width}\n")
            for key, value in cls._ARCHITECTURE_TEMPLATE.items():
                handle.writelines(f"{key}:    {value}\n")
            handle.writelines(f"Dataflow : {dataflow}\n\n")
            handle.writelines("[layout]\n")
            for key, value in cls._LAYOUT_SECTION_TEMPLATE.items():
                handle.writelines(f"{key}: {value}\n")
            handle.writelines("\n")
            handle.writelines("[run_presets]\n")
            for key, value in cls._RUN_PRESETS_TEMPLATE.items():
                handle.writelines(f"{key}: {value}\n")

    @classmethod
    def write_scalesim_layout(cls, path: str) -> None:
        with open(path, "w") as handle:
            handle.writelines(
                "Layer name,"
                "IFMAP Height Intraline Factor,"
                "IFMAP Width Intraline Factor,"
                "Filter Height Intraline Factor,"
                "Filter Width Intraline Factor,"
                "Channel Intraline Factor,"
                "Num Filter Intraline Factor,"
                "IFMAP Height Intraline Order,"
                "IFMAP Width Intraline Order,"
                "Channel Intraline Order,"
                "IFMAP Height Interline Order,"
                "IFMAP Width Interline Order,"
                "Channel Interline Order,"
                "Num Filter Intraline Order,"
                "Channel Intraline Order,"
                "Filter Height Intraline Order,"
                "Filter Width Intraline Order,"
                "Num Filter Interline Order,"
                "Channel Interline Order,"
                "Filter Height Interline Order,"
                "Filter Width Interline Order,\n"
            )
            handle.writelines(f"{cls._LAYOUT_FILE_ROW}\n")
