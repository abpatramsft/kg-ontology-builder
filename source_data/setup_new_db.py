"""
setup_new_db.py — Creates an SQLite database for IndiGo Airlines operations
from CSV files located in ../source_data_file/.

Tables (with FK relationships):
  airports               (PK: airport_id)
  aircraft               (PK: aircraft_id)           → airports
  passengers             (PK: passenger_id)
  crew                   (PK: crew_id)               → airports
  fare_classes           (PK: fare_class_id)
  routes                 (PK: route_id)              → airports × 2
  flights                (PK: flight_id)             → routes, aircraft
  bookings               (PK: booking_id)            → passengers, flights, fare_classes
  flight_crew_assignments(PK: assignment_id)         → flights, crew
  incidents              (PK: incident_id)           → flights, aircraft
  maintenance            (PK: maintenance_id)        → aircraft
"""

import csv
import glob
import json
import os
import shutil
import sqlite3
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "..", "source_data_files")
DB_PATH = os.path.join(SCRIPT_DIR, "airlines.db")


def _read_csv(filename: str) -> list[dict]:
    """Read a CSV file and return a list of row dicts."""
    path = os.path.join(CSV_DIR, filename)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def create_database():
    # Remove existing DB for a clean slate
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    # ── 1. airports ─────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE airports (
            airport_id      TEXT PRIMARY KEY,
            iata_code       TEXT NOT NULL,
            icao_code       TEXT NOT NULL,
            name            TEXT NOT NULL,
            city            TEXT,
            state           TEXT,
            country         TEXT,
            latitude        REAL,
            longitude       REAL,
            elevation_ft    INTEGER,
            timezone        TEXT
        )
    """)
    for r in _read_csv("airports.csv"):
        cur.execute(
            "INSERT INTO airports VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["airport_id"], r["iata_code"], r["icao_code"], r["name"],
                r["city"], r["state"], r["country"],
                float(r["latitude"]) if r["latitude"] else None,
                float(r["longitude"]) if r["longitude"] else None,
                int(r["elevation_ft"]) if r["elevation_ft"] else None,
                r["timezone"],
            ),
        )

    # ── 2. aircraft ─────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE aircraft (
            aircraft_id         TEXT PRIMARY KEY,
            registration        TEXT NOT NULL,
            model               TEXT,
            manufacturer        TEXT,
            variant             TEXT,
            capacity_economy    INTEGER,
            capacity_business   INTEGER,
            year_manufactured   INTEGER,
            engine_type         TEXT,
            engines_count       INTEGER,
            range_km            INTEGER,
            status              TEXT,
            base_airport_id     TEXT,
            msn                 TEXT,
            FOREIGN KEY (base_airport_id) REFERENCES airports(airport_id)
        )
    """)
    for r in _read_csv("aircraft.csv"):
        cur.execute(
            "INSERT INTO aircraft VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["aircraft_id"], r["registration"], r["model"],
                r["manufacturer"], r["variant"],
                int(r["capacity_economy"]) if r["capacity_economy"] else None,
                int(r["capacity_business"]) if r["capacity_business"] else None,
                int(r["year_manufactured"]) if r["year_manufactured"] else None,
                r["engine_type"],
                int(r["engines_count"]) if r["engines_count"] else None,
                int(r["range_km"]) if r["range_km"] else None,
                r["status"],
                r["base_airport_id"] if r["base_airport_id"] else None,
                r["msn"],
            ),
        )

    # ── 3. passengers ───────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE passengers (
            passenger_id        TEXT PRIMARY KEY,
            first_name          TEXT NOT NULL,
            last_name           TEXT NOT NULL,
            email               TEXT,
            phone               TEXT,
            nationality         TEXT,
            date_of_birth       TEXT,
            passport_number     TEXT,
            frequent_flyer_id   TEXT,
            tier                TEXT,
            signup_date         TEXT
        )
    """)
    for r in _read_csv("passengers.csv"):
        cur.execute(
            "INSERT INTO passengers VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["passenger_id"], r["first_name"], r["last_name"],
                r["email"], r["phone"], r["nationality"],
                r["date_of_birth"], r["passport_number"],
                r["frequent_flyer_id"], r["tier"], r["signup_date"],
            ),
        )

    # ── 4. crew ─────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE crew (
            crew_id             TEXT PRIMARY KEY,
            first_name          TEXT NOT NULL,
            last_name           TEXT NOT NULL,
            role                TEXT,
            employee_id         TEXT,
            license_number      TEXT,
            base_airport_id     TEXT,
            years_experience    INTEGER,
            nationality         TEXT,
            date_of_birth       TEXT,
            join_date           TEXT,
            qualification       TEXT,
            FOREIGN KEY (base_airport_id) REFERENCES airports(airport_id)
        )
    """)
    for r in _read_csv("crew.csv"):
        cur.execute(
            "INSERT INTO crew VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["crew_id"], r["first_name"], r["last_name"],
                r["role"], r["employee_id"], r["license_number"],
                r["base_airport_id"] if r["base_airport_id"] else None,
                int(r["years_experience"]) if r["years_experience"] else None,
                r["nationality"], r["date_of_birth"], r["join_date"],
                r["qualification"],
            ),
        )

    # ── 5. fare_classes ─────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE fare_classes (
            fare_class_id           TEXT PRIMARY KEY,
            fare_class_code         TEXT NOT NULL UNIQUE,
            fare_class_name         TEXT NOT NULL UNIQUE,
            base_price_factor       REAL,
            flexibility             TEXT,
            changes_fee_inr         INTEGER,
            cancellation_fee_inr    TEXT,
            baggage_allowance_kg    INTEGER,
            meal_included           TEXT,
            seat_selection          TEXT,
            priority_boarding       TEXT,
            lounge_access           TEXT
        )
    """)
    for r in _read_csv("fare_classes.csv"):
        cur.execute(
            "INSERT INTO fare_classes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["fare_class_id"], r["fare_class_code"], r["fare_class_name"],
                float(r["base_price_factor"]) if r["base_price_factor"] else None,
                r["flexibility"],
                int(r["changes_fee_inr"]) if r.get("changes_fee_inr", "").isdigit() else r.get("changes_fee_inr"),
                r["cancellation_fee_inr"],
                int(r["baggage_allowance_kg"]) if r["baggage_allowance_kg"] else None,
                r["meal_included"], r["seat_selection"],
                r["priority_boarding"], r["lounge_access"],
            ),
        )

    # ── 6. routes ───────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE routes (
            route_id                TEXT PRIMARY KEY,
            origin_airport_id       TEXT NOT NULL,
            destination_airport_id  TEXT NOT NULL,
            distance_km             INTEGER,
            flight_time_min         INTEGER,
            is_international        TEXT,
            frequency_per_week      INTEGER,
            FOREIGN KEY (origin_airport_id)      REFERENCES airports(airport_id),
            FOREIGN KEY (destination_airport_id) REFERENCES airports(airport_id)
        )
    """)
    for r in _read_csv("routes.csv"):
        cur.execute(
            "INSERT INTO routes VALUES (?,?,?,?,?,?,?)",
            (
                r["route_id"], r["origin_airport_id"],
                r["destination_airport_id"],
                int(r["distance_km"]) if r["distance_km"] else None,
                int(r["flight_time_min"]) if r["flight_time_min"] else None,
                r["is_international"],
                int(r["frequency_per_week"]) if r["frequency_per_week"] else None,
            ),
        )

    # ── 7. flights ──────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE flights (
            flight_id               TEXT PRIMARY KEY,
            flight_number           TEXT NOT NULL,
            route_id                TEXT NOT NULL,
            aircraft_id             TEXT NOT NULL,
            scheduled_departure     TEXT,
            scheduled_arrival       TEXT,
            actual_departure        TEXT,
            actual_arrival          TEXT,
            status                  TEXT,
            delay_minutes           INTEGER,
            gate                    TEXT,
            terminal                TEXT,
            FOREIGN KEY (route_id)    REFERENCES routes(route_id),
            FOREIGN KEY (aircraft_id) REFERENCES aircraft(aircraft_id)
        )
    """)
    for r in _read_csv("flights.csv"):
        cur.execute(
            "INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["flight_id"], r["flight_number"], r["route_id"],
                r["aircraft_id"], r["scheduled_departure"],
                r["scheduled_arrival"], r["actual_departure"],
                r["actual_arrival"], r["status"],
                int(r["delay_minutes"]) if r.get("delay_minutes", "").lstrip("-").isdigit() else None,
                r["gate"], r["terminal"],
            ),
        )

    # ── 8. bookings ─────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE bookings (
            booking_id          TEXT PRIMARY KEY,
            passenger_id        TEXT NOT NULL,
            flight_id           TEXT NOT NULL,
            booking_date        TEXT,
            fare_class          TEXT NOT NULL,
            seat_number         TEXT,
            ticket_price_inr    REAL,
            check_in_status     TEXT,
            baggage_checkin_kg  REAL,
            baggage_cabin_kg    REAL,
            meal_preference     TEXT,
            web_checkin          TEXT,
            FOREIGN KEY (passenger_id) REFERENCES passengers(passenger_id),
            FOREIGN KEY (flight_id)    REFERENCES flights(flight_id),
            FOREIGN KEY (fare_class)   REFERENCES fare_classes(fare_class_name)
        )
    """)
    for r in _read_csv("bookings.csv"):
        cur.execute(
            "INSERT INTO bookings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["booking_id"], r["passenger_id"], r["flight_id"],
                r["booking_date"], r["fare_class"], r["seat_number"],
                float(r["ticket_price_inr"]) if r["ticket_price_inr"] else None,
                r["check_in_status"],
                float(r["baggage_checkin_kg"]) if r["baggage_checkin_kg"] else None,
                float(r["baggage_cabin_kg"]) if r["baggage_cabin_kg"] else None,
                r["meal_preference"], r["web_checkin"],
            ),
        )

    # ── 9. flight_crew_assignments ──────────────────────────────────
    cur.execute("""
        CREATE TABLE flight_crew_assignments (
            assignment_id       TEXT PRIMARY KEY,
            flight_id           TEXT NOT NULL,
            crew_id             TEXT NOT NULL,
            role_on_flight      TEXT,
            is_commander        TEXT,
            FOREIGN KEY (flight_id) REFERENCES flights(flight_id),
            FOREIGN KEY (crew_id)   REFERENCES crew(crew_id)
        )
    """)
    for r in _read_csv("flight_crew_assignments.csv"):
        cur.execute(
            "INSERT INTO flight_crew_assignments VALUES (?,?,?,?,?)",
            (
                r["assignment_id"], r["flight_id"], r["crew_id"],
                r["role_on_flight"], r["is_commander"],
            ),
        )

    # ── 10. incidents ───────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE incidents (
            incident_id         TEXT PRIMARY KEY,
            flight_id           TEXT,
            aircraft_id         TEXT,
            incident_date       TEXT,
            incident_type       TEXT,
            severity            TEXT,
            phase_of_flight     TEXT,
            description         TEXT,
            resolution_status   TEXT,
            reported_to_dgca    TEXT,
            FOREIGN KEY (flight_id)   REFERENCES flights(flight_id),
            FOREIGN KEY (aircraft_id) REFERENCES aircraft(aircraft_id)
        )
    """)
    for r in _read_csv("incidents.csv"):
        cur.execute(
            "INSERT INTO incidents VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                r["incident_id"],
                r["flight_id"] if r["flight_id"] else None,
                r["aircraft_id"] if r["aircraft_id"] else None,
                r["incident_date"], r["incident_type"], r["severity"],
                r["phase_of_flight"], r["description"],
                r["resolution_status"], r["reported_to_dgca"],
            ),
        )

    # ── 11. maintenance ─────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE maintenance (
            maintenance_id      TEXT PRIMARY KEY,
            aircraft_id         TEXT NOT NULL,
            maintenance_type    TEXT,
            start_date          TEXT,
            end_date            TEXT,
            technician_id       TEXT,
            facility            TEXT,
            status              TEXT,
            cost_inr            REAL,
            findings            TEXT,
            next_due_date       TEXT,
            FOREIGN KEY (aircraft_id) REFERENCES aircraft(aircraft_id)
        )
    """)
    for r in _read_csv("maintenance.csv"):
        cur.execute(
            "INSERT INTO maintenance VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["maintenance_id"], r["aircraft_id"],
                r["maintenance_type"], r["start_date"], r["end_date"],
                r["technician_id"], r["facility"], r["status"],
                float(r["cost_inr"]) if r.get("cost_inr", "").replace(".", "", 1).isdigit() else None,
                r["findings"],
                r["next_due_date"] if r.get("next_due_date") else None,
            ),
        )

    # ── 12. invoices (single table — nested data stored as JSON text) ──
    cur.execute("""
        CREATE TABLE invoices (
            invoice_number          TEXT PRIMARY KEY,
            purchase_order          TEXT,
            customer_name           TEXT,
            customer_address        TEXT,
            delivery_date           TEXT,
            payable_by              TEXT,
            total_quantity          INTEGER,
            total_price             REAL,
            products                TEXT,
            returns                 TEXT,
            products_signatures     TEXT,
            returns_signatures      TEXT,
            source_pdf              TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database created: {DB_PATH}")


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
                print(f"    FK: {fk[3]} -> {fk[2]}({fk[4]})")

    conn.close()


def process_invoices():
    """Find PDF invoices in source dir, extract JSON via pdf_extractor, load into DB."""
    pdfs = glob.glob(os.path.join(CSV_DIR, "*.pdf"))
    if not pdfs:
        print("\nNo PDF files found — skipping invoice extraction.")
        return

    extractor_script = os.path.join(SCRIPT_DIR, "..", "utils", "pdf_extractor.py")
    json_files: list[str] = []

    for pdf_path in pdfs:
        # Check if extraction JSON already exists
        json_path = pdf_path + ".Extraction.json"
        if os.path.exists(json_path):
            print(f"  Found existing extraction: {os.path.basename(json_path)}")
        else:
            print(f"  Extracting: {os.path.basename(pdf_path)} ...")
            subprocess.run(
                [sys.executable, extractor_script, "--pdf", pdf_path],
                check=True,
            )
        if os.path.exists(json_path):
            json_files.append(json_path)

    if not json_files:
        print("  No extraction JSONs produced.")
        return

    # Load extracted invoices into the DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        inv_num = data.get("InvoiceNumber", "")
        source_pdf = os.path.basename(jf).replace(".Extraction.json", "")

        # Insert invoice as a single row (nested arrays stored as JSON text)
        cur.execute(
            "INSERT OR REPLACE INTO invoices VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                inv_num,
                data.get("PurchaseOrderNumber", ""),
                data.get("CustomerName", ""),
                data.get("CustomerAddress", ""),
                data.get("DeliveryDate", ""),
                data.get("PayableBy", ""),
                data.get("TotalQuantity", 0),
                data.get("TotalPrice", 0.0),
                json.dumps(data.get("Products", [])),
                json.dumps(data.get("Returns", [])),
                json.dumps(data.get("ProductsSignatures", [])),
                json.dumps(data.get("ReturnsSignatures", [])),
                source_pdf,
            ),
        )

        print(f"  Loaded invoice #{inv_num} ({source_pdf}): "
              f"{len(data.get('Products', []))} products, "
              f"{len(data.get('Returns', []))} returns")

    conn.commit()
    conn.close()


def copy_txt_files():
    """Copy any .txt files from the source CSV directory into this folder."""
    txts = glob.glob(os.path.join(CSV_DIR, "*.txt"))
    for txt in txts:
        dest = os.path.join(SCRIPT_DIR, os.path.basename(txt))
        shutil.copy2(txt, dest)
        print(f"  Copied: {os.path.basename(txt)}")
    if not txts:
        print("No .txt files found in source directory.")


if __name__ == "__main__":
    create_database()
    process_invoices()
    copy_txt_files()
    verify_database()
