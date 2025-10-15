# app.py
from flask import Flask, jsonify, abort, send_file
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError
import re

_engine = None

def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("Missing DB_URL (or DATABASE_URL) environment variable.")
    # Normalize old 'postgres://' scheme to 'postgresql://'
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    _engine = create_engine(
        db_url,
        pool_pre_ping=True,
    )
    return _engine

def create_app():
    app = Flask(__name__)

    @app.get("/", endpoint="health")
    def health():
        return "<p>Server working!</p>"

    @app.get("/img", endpoint="show_img")
    def show_img():
        return send_file("amygdala.gif", mimetype="image/gif")

    @app.get("/terms/<term>/studies", endpoint="terms_studies")
    def get_studies_by_term(term):
        # Query DB for studies that mention `term` and include their coords + top terms
        eng = get_engine()
        # Normalize user input to the same canonical form used when importing terms
        def norm_term_input(t: str) -> str:
            if t is None:
                return t
            s = t.strip().lower()
            # convert spaces to underscores and collapse multiple underscores
            s = re.sub(r"\s+", "_", s)
            s = re.sub(r"_+", "_", s)
            s = s.strip("_")
            return s

        term_norm = norm_term_input(term)
        with eng.begin() as conn:
            # normalize stored term in SQL: collapse spaces/underscores to single underscore and trim
            rows = conn.execute(text("""
                SELECT at.study_id, at.contrast_id, at.weight,
                       ST_X(c.geom) AS x, ST_Y(c.geom) AS y, ST_Z(c.geom) AS z
                FROM ns.annotations_terms at
                LEFT JOIN ns.coordinates c ON at.study_id = c.study_id
                WHERE trim(both '_' from regexp_replace(lower(at.term), '[\\s_]+', '_', 'g')) = :term
                ORDER BY at.weight DESC
                LIMIT 100
            """), {"term": term_norm}).mappings().all()

            # Collect study ids to fetch top terms for each study
            study_ids = list({r["study_id"] for r in rows})
            if study_ids:
                # Build IN clause safely
                params = {f"id{i}": sid for i, sid in enumerate(study_ids)}
                in_clause = ", ".join([f":id{i}" for i in range(len(study_ids))])
                term_rows = conn.execute(text(f"""
                    SELECT study_id, term, weight
                    FROM ns.annotations_terms
                    WHERE study_id IN ({in_clause})
                    ORDER BY study_id, weight DESC
                """), params).mappings().all()
            else:
                term_rows = []

        # Aggregate top terms per study (top 5)
        top_terms = {}
        for tr in term_rows:
            sid = tr["study_id"]
            top_terms.setdefault(sid, []).append({"term": tr["term"], "weight": float(tr["weight"])})

        result = []
        for r in rows:
            sid = r["study_id"]
            result.append({
                "study_id": sid,
                "contrast_id": r["contrast_id"],
                "weight_for_query_term": float(r["weight"]) if r["weight"] is not None else None,
                "coords": None if r["x"] is None else [float(r["x"]), float(r["y"]), float(r["z"])],
                "top_terms": top_terms.get(sid, [])[:5]
            })
        return jsonify({"query_term": term_norm, "count": len(result), "studies": result})

    @app.get("/locations/<coords>/studies", endpoint="locations_studies")
    def get_studies_by_coordinates(coords):
        # Find nearest studies to coords and return their coordinates + top terms
        try:
            x, y, z = map(float, coords.split("_"))
        except Exception:
            abort(400, "Invalid coordinates format; expected x_y_z")

        eng = get_engine()
        with eng.begin() as conn:
            # Use KNN ordering (requires PostGIS + GIST on geom). Limit to 50 nearest studies.
            rows = conn.execute(text("""
                SELECT study_id, ST_X(geom) AS x, ST_Y(geom) AS y, ST_Z(geom) AS z
                FROM ns.coordinates
                ORDER BY geom <-> ST_SetSRID(ST_MakePoint(:x, :y, :z), 4326)::geometry(POINTZ,4326)
                LIMIT 50
            """), {"x": x, "y": y, "z": z}).mappings().all()

            study_ids = [r["study_id"] for r in rows]
            if study_ids:
                params = {f"id{i}": sid for i, sid in enumerate(study_ids)}
                in_clause = ", ".join([f":id{i}" for i in range(len(study_ids))])
                term_rows = conn.execute(text(f"""
                    SELECT study_id, term, weight
                    FROM ns.annotations_terms
                    WHERE study_id IN ({in_clause})
                    ORDER BY study_id, weight DESC
                """), params).mappings().all()
            else:
                term_rows = []

        # Aggregate top terms per study (top 5)
        top_terms = {}
        for tr in term_rows:
            sid = tr["study_id"]
            top_terms.setdefault(sid, []).append({"term": tr["term"], "weight": float(tr["weight"])})

        result = []
        for r in rows:
            sid = r["study_id"]
            result.append({
                "study_id": sid,
                "coords": [float(r["x"]), float(r["y"]), float(r["z"])],
                "top_terms": top_terms.get(sid, [])[:5]
            })
        return jsonify({"query_coords": [x, y, z], "count": len(result), "nearest": result})
    
    # Updated: dissociate by two terms -> run A\B and B\A against the DB
    @app.get("/dissociate/terms/<term_a>/<term_b>", endpoint="dissociate_terms")
    def dissociate_terms(term_a, term_b):
        def norm_term(t):
            # reuse same normalization as other endpoints
            s = t.strip().lower()
            s = re.sub(r"\s+", "_", s)
            s = re.sub(r"_+", "_", s)
            return s.strip("_")

        a_term = norm_term(term_a)
        b_term = norm_term(term_b)

        eng = get_engine()
        try:
            with eng.begin() as conn:
                conn.execute(text("SET search_path TO ns, public;"))

                # Get study ids for each term (limit to 1000 to avoid huge IN lists)
                # normalize term comparison in SQL to handle spaces/underscores/case
                a_rows = conn.execute(text("""
                    SELECT DISTINCT study_id
                    FROM ns.annotations_terms
                    WHERE trim(both '_' from regexp_replace(lower(term), '[\\s_]+', '_', 'g')) = :term
                    LIMIT 1000
                """), {"term": a_term}).scalars().all()

                b_rows = conn.execute(text("""
                    SELECT DISTINCT study_id
                    FROM ns.annotations_terms
                    WHERE trim(both '_' from regexp_replace(lower(term), '[\\s_]+', '_', 'g')) = :term
                    LIMIT 1000
                """), {"term": b_term}).scalars().all()

                a_ids = set(a_rows)
                b_ids = set(b_rows)

                a_only_ids = list(a_ids - b_ids)
                b_only_ids = list(b_ids - a_ids)
                overlap_ids = list(a_ids & b_ids)

                def fetch_details(study_ids):
                    if not study_ids:
                        return []
                    params = {f"id{i}": sid for i, sid in enumerate(study_ids)}
                    in_clause = ", ".join([f":id{i}" for i in range(len(study_ids))])

                    coords_rows = conn.execute(text(f"""
                        SELECT study_id, ST_X(geom) AS x, ST_Y(geom) AS y, ST_Z(geom) AS z
                        FROM ns.coordinates
                        WHERE study_id IN ({in_clause})
                    """), params).mappings().all()

                    term_rows = conn.execute(text(f"""
                        SELECT study_id, term, weight
                        FROM ns.annotations_terms
                        WHERE study_id IN ({in_clause})
                        ORDER BY study_id, weight DESC
                    """), params).mappings().all()

                    top_terms = {}
                    for tr in term_rows:
                        sid = tr["study_id"]
                        top_terms.setdefault(sid, []).append({"term": tr["term"], "weight": float(tr["weight"]) if tr["weight"] is not None else None})

                    coords_map = {r["study_id"]: [float(r["x"]), float(r["y"]), float(r["z"])] if r["x"] is not None else None for r in coords_rows}

                    results = []
                    for sid in study_ids:
                        results.append({
                            "study_id": sid,
                            "coords": coords_map.get(sid),
                            "top_terms": top_terms.get(sid, [])[:5]
                        })
                    return results

                a_only = fetch_details(a_only_ids)
                b_only = fetch_details(b_only_ids)
                overlap = fetch_details(overlap_ids)

            return jsonify({
                "term_a_raw": term_a,
                "term_b_raw": term_b,
                "term_a": a_term,
                "term_b": b_term,
                "a_only_count": len(a_only),
                "b_only_count": len(b_only),
                "overlap_count": len(overlap),
                "a_only": a_only,
                "b_only": b_only,
                "overlap": overlap
            }), 200
        except OperationalError as e:
            abort(500, f"DB error: {e}")
        except Exception as e:
            abort(500, str(e))

    # Updated: dissociate by two coordinate triples -> nearest-based A\B and B\A
    @app.get("/dissociate/locations/<coords_a>/<coords_b>", endpoint="dissociate_locations")
    def dissociate_locations(coords_a, coords_b):
        def parse_coords(s):
            parts = s.split("_")
            if len(parts) != 3:
                abort(400, f"Invalid coordinates '{s}'")
            try:
                return [float(p) for p in parts]
            except ValueError:
                abort(400, f"Invalid coordinates '{s}'")

        a = parse_coords(coords_a)
        b = parse_coords(coords_b)

        eng = get_engine()
        try:
            with eng.begin() as conn:
                conn.execute(text("SET search_path TO ns, public;"))

                # Find nearest N studies to each point (use KNN). Adjust SRID if different.
                N = 100
                a_rows = conn.execute(text("""
                    SELECT study_id
                    FROM ns.coordinates
                    ORDER BY geom <-> ST_SetSRID(ST_MakePoint(:x, :y, :z), 4326)::geometry(POINTZ,4326)
                    LIMIT :n
                """), {"x": a[0], "y": a[1], "z": a[2], "n": N}).scalars().all()

                b_rows = conn.execute(text("""
                    SELECT study_id
                    FROM ns.coordinates
                    ORDER BY geom <-> ST_SetSRID(ST_MakePoint(:x, :y, :z), 4326)::geometry(POINTZ,4326)
                    LIMIT :n
                """), {"x": b[0], "y": b[1], "z": b[2], "n": N}).scalars().all()

                a_ids = set(a_rows)
                b_ids = set(b_rows)

                a_only_ids = list(a_ids - b_ids)
                b_only_ids = list(b_ids - a_ids)
                overlap_ids = list(a_ids & b_ids)

                def fetch_details(study_ids):
                    if not study_ids:
                        return []
                    params = {f"id{i}": sid for i, sid in enumerate(study_ids)}
                    in_clause = ", ".join([f":id{i}" for i in range(len(study_ids))])

                    coords_rows = conn.execute(text(f"""
                        SELECT study_id, ST_X(geom) AS x, ST_Y(geom) AS y, ST_Z(geom) AS z
                        FROM ns.coordinates
                        WHERE study_id IN ({in_clause})
                    """), params).mappings().all()

                    term_rows = conn.execute(text(f"""
                        SELECT study_id, term, weight
                        FROM ns.annotations_terms
                        WHERE study_id IN ({in_clause})
                        ORDER BY study_id, weight DESC
                    """), params).mappings().all()

                    top_terms = {}
                    for tr in term_rows:
                        sid = tr["study_id"]
                        top_terms.setdefault(sid, []).append({"term": tr["term"], "weight": float(tr["weight"]) if tr["weight"] is not None else None})

                    coords_map = {r["study_id"]: [float(r["x"]), float(r["y"]), float(r["z"])] if r["x"] is not None else None for r in coords_rows}

                    results = []
                    for sid in study_ids:
                        results.append({
                            "study_id": sid,
                            "coords": coords_map.get(sid),
                            "top_terms": top_terms.get(sid, [])[:5]
                        })
                    return results

                a_only = fetch_details(a_only_ids)
                b_only = fetch_details(b_only_ids)
                overlap = fetch_details(overlap_ids)

            return jsonify({
                "query_a": a,
                "query_b": b,
                "a_only_count": len(a_only),
                "b_only_count": len(b_only),
                "overlap_count": len(overlap),
                "a_only": a_only,
                "b_only": b_only,
                "overlap": overlap
            }), 200
        except OperationalError as e:
            abort(500, f"DB error: {e}")
        except Exception as e:
            abort(500, str(e))

    @app.get("/test_db", endpoint="test_db")
    
    def test_db():
        eng = get_engine()
        payload = {"ok": False, "dialect": eng.dialect.name}

        try:
            with eng.begin() as conn:
                # Ensure we are in the correct schema
                conn.execute(text("SET search_path TO ns, public;"))
                payload["version"] = conn.exec_driver_sql("SELECT version()").scalar()

                # Counts
                payload["coordinates_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.coordinates")).scalar()
                payload["metadata_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.metadata")).scalar()
                payload["annotations_terms_count"] = conn.execute(text("SELECT COUNT(*) FROM ns.annotations_terms")).scalar()

                # Samples
                try:
                    rows = conn.execute(text(
                        "SELECT study_id, ST_X(geom) AS x, ST_Y(geom) AS y, ST_Z(geom) AS z FROM ns.coordinates LIMIT 3"
                    )).mappings().all()
                    payload["coordinates_sample"] = [dict(r) for r in rows]
                except Exception:
                    payload["coordinates_sample"] = []

                try:
                    # Select a few columns if they exist; otherwise select a generic subset
                    rows = conn.execute(text("SELECT * FROM ns.metadata LIMIT 3")).mappings().all()
                    payload["metadata_sample"] = [dict(r) for r in rows]
                except Exception:
                    payload["metadata_sample"] = []

                try:
                    rows = conn.execute(text(
                        "SELECT study_id, contrast_id, term, weight FROM ns.annotations_terms LIMIT 3"
                    )).mappings().all()
                    payload["annotations_terms_sample"] = [dict(r) for r in rows]
                except Exception:
                    payload["annotations_terms_sample"] = []

            payload["ok"] = True
            return jsonify(payload), 200

        except Exception as e:
            payload["error"] = str(e)
            return jsonify(payload), 500

        @app.get("/ui", endpoint="ui")
        def ui():
                # Minimal single-page UI to query the API endpoints from a browser
                html = """
                <!doctype html>
                <html>
                <head>
                    <meta charset="utf-8" />
                    <title>NeuralInfo — quick UI</title>
                    <style>body{font-family:system-ui,Arial;max-width:900px;margin:40px;} textarea{width:100%;height:320px}</style>
                </head>
                <body>
                    <h1>NeuralInfo — quick UI</h1>

                    <section>
                        <h2>Query by term</h2>
                        <input id="term" placeholder="posterior_cingulate" style="width:60%" />
                        <button onclick="runTerm()">Query</button>
                    </section>

                    <section style="margin-top:18px;">
                        <h2>Query by coordinates</h2>
                        <input id="coords" placeholder="0_-52_26" style="width:60%" />
                        <button onclick="runCoords()">Query</button>
                    </section>

                    <section style="margin-top:18px;">
                        <h2>Dissociate by two terms</h2>
                        <input id="term_a" placeholder="ventromedial_prefrontal" style="width:28%" />
                        <input id="term_b" placeholder="posterior_cingulate" style="width:28%;margin-left:8px;" />
                        <button onclick="runDissociateTerms()">Dissociate Terms</button>
                    </section>

                    <section style="margin-top:18px;">
                        <h2>Results</h2>
                        <textarea id="out" readonly></textarea>
                    </section>

                    <script>
                        async function handleResponse(res){
                            let txt;
                            try{
                                const j = await res.json();
                                txt = JSON.stringify(j,null,2);
                            }catch(e){
                                txt = await res.text();
                            }
                            document.getElementById('out').value = txt;
                        }

                        async function runTerm(){
                            const t = document.getElementById('term').value.trim();
                            if(!t){ alert('Enter a term (use underscores for spaces)'); return }
                            const res = await fetch(`/terms/${encodeURIComponent(t)}/studies`);
                            await handleResponse(res);
                        }

                        async function runCoords(){
                            const c = document.getElementById('coords').value.trim();
                            if(!c){ alert('Enter coords as x_y_z'); return }
                            const res = await fetch(`/locations/${encodeURIComponent(c)}/studies`);
                            await handleResponse(res);
                        }

                        async function runDissociateTerms(){
                            const a = document.getElementById('term_a').value.trim();
                            const b = document.getElementById('term_b').value.trim();
                            if(!a || !b){ alert('Enter both terms (use underscores for spaces)'); return }
                            const res = await fetch(`/dissociate/terms/${encodeURIComponent(a)}/${encodeURIComponent(b)}`);
                            await handleResponse(res);
                        }
                    </script>
                </body>
                </html>
                """
                return html
    return app

# WSGI entry point (no __main__)
app = create_app()
