"""
setup_db.py — Creates a sample SQLite database for IndiGo Airlines (manufacturing/supply chain).

Tables:
  - products    : Aircraft fleet models
  - assemblies  : Major aircraft assemblies/systems
  - parts       : Component parts within assemblies
  - suppliers   : Aerospace suppliers for parts

FK chain: products ← assemblies ← parts ← suppliers
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "manufacturing.db")


def create_database():
    # Remove existing DB for a clean slate
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    # ── Products: Aircraft fleet models ──────────────────────────────
    cur.execute("""
        CREATE TABLE products (
            product_id      INTEGER PRIMARY KEY,
            product_name    TEXT NOT NULL,
            category        TEXT,
            unit_price      REAL,
            weight_kg       REAL,
            created_date    TEXT
        )
    """)

    products = [
        (1, "Airbus A320neo",  "Narrow-body",  110000000, 44300, "2016-01-25"),
        (2, "Airbus A321neo",  "Narrow-body",  129500000, 48500, "2017-03-15"),
        (3, "ATR 72-600",      "Turboprop",     26000000, 13500, "2011-10-01"),
        (4, "Airbus A320ceo",  "Narrow-body",   98000000, 42600, "1988-02-22"),
        (5, "Airbus A321XLR",  "Narrow-body",  135000000, 49000, "2024-06-01"),
    ]
    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", products)

    # ── Assemblies: Major aircraft systems ───────────────────────────
    cur.execute("""
        CREATE TABLE assemblies (
            assembly_id     INTEGER PRIMARY KEY,
            assembly_name   TEXT NOT NULL,
            product_id      INTEGER NOT NULL,
            assembly_type   TEXT,
            num_components  INTEGER,
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
    """)

    assemblies = [
        # A320neo assemblies
        (1,  "Landing Gear Assembly",    1, "Structural",   45),
        (2,  "PW1100G Engine Assembly",  1, "Propulsion",   120),
        (3,  "Avionics Suite",           1, "Electronics",  85),
        (4,  "Hydraulic System",         1, "Mechanical",   60),
        # A321neo assemblies
        (5,  "Landing Gear Assembly",    2, "Structural",   48),
        (6,  "PW1133G Engine Assembly",  2, "Propulsion",   130),
        (7,  "Cabin Pressurization Unit",2, "Environmental",55),
        # ATR 72-600 assemblies
        (8,  "PW127M Engine Assembly",   3, "Propulsion",   95),
        (9,  "Propeller Assembly",       3, "Propulsion",   35),
        (10, "Flight Control System",    3, "Avionics",     70),
        # A320ceo assemblies
        (11, "CFM56 Engine Assembly",    4, "Propulsion",   110),
        (12, "APU System",              4, "Power",         40),
    ]
    cur.executemany("INSERT INTO assemblies VALUES (?,?,?,?,?)", assemblies)

    # ── Parts: Component parts within assemblies ─────────────────────
    cur.execute("""
        CREATE TABLE parts (
            part_id         INTEGER PRIMARY KEY,
            part_name       TEXT NOT NULL,
            assembly_id     INTEGER NOT NULL,
            material        TEXT,
            unit_cost       REAL,
            lead_time_days  INTEGER,
            FOREIGN KEY (assembly_id) REFERENCES assemblies(assembly_id)
        )
    """)

    parts = [
        # Landing Gear Assembly (A320neo)
        (1,  "Main Landing Gear Strut",     1, "Titanium Alloy",   85000,  90),
        (2,  "Nose Wheel Tire",             1, "Rubber Composite",  1200,  14),
        (3,  "Brake Disc",                  1, "Carbon-Carbon",    12000,  30),
        (4,  "Shock Absorber",              1, "Steel Alloy",       9500,  45),
        # PW1100G Engine (A320neo)
        (5,  "Fan Blade",                   2, "Titanium Alloy",   45000,  120),
        (6,  "Turbine Disc",                2, "Nickel Superalloy", 72000, 150),
        (7,  "Combustion Chamber Liner",    2, "Ceramic Composite", 38000, 100),
        (8,  "FADEC Controller",            2, "Electronics",       55000,  60),
        # Avionics Suite (A320neo)
        (9,  "Navigation Radio",            3, "Electronics",       28000,  45),
        (10, "Weather Radar Antenna",       3, "Composite",         35000,  55),
        (11, "Flight Data Recorder",        3, "Titanium Housing",  42000,  40),
        # Hydraulic System (A320neo)
        (12, "Hydraulic Pump",              4, "Aluminum Alloy",    18000,  35),
        (13, "Actuator Valve",              4, "Stainless Steel",    7500,  25),
        # A321neo Landing Gear
        (14, "Main Landing Gear Strut",     5, "Titanium Alloy",   92000,  95),
        (15, "Brake Disc",                  5, "Carbon-Carbon",    12500,  30),
        # PW1133G Engine (A321neo)
        (16, "High-Pressure Turbine Blade", 6, "Single Crystal Alloy", 68000, 160),
        (17, "Fuel Nozzle",                 6, "Inconel",           15000,  50),
        # ATR engines
        (18, "Propeller Blade",             9, "Carbon Fiber",      22000,  70),
        (19, "Reduction Gearbox",           8, "Steel Alloy",       48000,  80),
        # CFM56 Engine (A320ceo)
        (20, "LP Turbine Blade",           11, "Nickel Alloy",      32000, 110),
    ]
    cur.executemany("INSERT INTO parts VALUES (?,?,?,?,?,?)", parts)

    # ── Suppliers: Aerospace part suppliers ──────────────────────────
    cur.execute("""
        CREATE TABLE suppliers (
            supplier_id     INTEGER PRIMARY KEY,
            supplier_name   TEXT NOT NULL,
            country         TEXT,
            rating          REAL,
            part_id         INTEGER NOT NULL,
            contract_start  TEXT,
            FOREIGN KEY (part_id) REFERENCES parts(part_id)
        )
    """)

    suppliers = [
        (1,  "Safran Landing Systems",   "France",       4.5,  1, "2018-04-01"),
        (2,  "Michelin Aircraft Tires",  "France",       4.2,  2, "2019-01-15"),
        (3,  "Honeywell Aerospace",      "USA",          4.7,  3, "2017-09-01"),
        (4,  "Pratt & Whitney",          "USA",          4.8,  5, "2016-06-01"),
        (5,  "Rolls-Royce Plc",          "UK",           4.6,  6, "2017-01-10"),
        (6,  "Collins Aerospace",        "USA",          4.4,  9, "2019-07-20"),
        (7,  "L3Harris Technologies",    "USA",          4.3, 11, "2020-02-01"),
        (8,  "Parker Hannifin",          "USA",          4.1, 12, "2018-11-15"),
        (9,  "Eaton Aerospace",          "Ireland",      4.0, 13, "2019-05-10"),
        (10, "Safran Landing Systems",   "France",       4.5, 14, "2018-08-01"),
        (11, "Honeywell Aerospace",      "USA",          4.7, 15, "2017-09-01"),
        (12, "Pratt & Whitney",          "USA",          4.8, 16, "2017-03-15"),
        (13, "Woodward Inc.",            "USA",          4.2, 17, "2020-01-01"),
        (14, "Ratier-Figeac",           "France",        4.3, 18, "2012-06-01"),
        (15, "Pratt & Whitney Canada",   "Canada",       4.6, 19, "2011-10-01"),
    ]
    cur.executemany("INSERT INTO suppliers VALUES (?,?,?,?,?,?)", suppliers)

    conn.commit()
    conn.close()
    print(f"Database created: {DB_PATH}")
    print("Tables: products(5), assemblies(12), parts(20), suppliers(15)")


def verify_database():
    """Quick verification of the created database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    print(f"\nTables found: {[t[0] for t in tables]}")

    for (table_name,) in tables:
        count = cur.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        cols = cur.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        fks = cur.execute(f"PRAGMA foreign_key_list('{table_name}')").fetchall()
        col_names = [c[1] for c in cols]
        print(f"  {table_name}: {count} rows | columns: {col_names}")
        if fks:
            for fk in fks:
                print(f"    FK: {fk[3]} → {fk[2]}({fk[4]})")

    conn.close()


if __name__ == "__main__":
    create_database()
    verify_database()
