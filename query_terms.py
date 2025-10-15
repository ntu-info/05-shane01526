import os
from sqlalchemy import text

# require DB_URL in environment
db = os.getenv('DB_URL')
if not db:
    print('Missing DB_URL in environment')
    raise SystemExit(1)

# import app.get_engine lazily to reuse existing function
from app import get_engine
eng = get_engine()

with eng.begin() as conn:
    print('Top 20 terms:')
    rows = conn.execute(text("SELECT term, COUNT(*) as cnt FROM ns.annotations_terms GROUP BY term ORDER BY cnt DESC LIMIT 20")).mappings().all()
    for r in rows:
        print(f"{r['term']}: {r['cnt']}")

    test_terms = ['posterior_cingulate', 'posterior cingulate', 'ventromedial_prefrontal', 'ventromedial prefrontal']
    print('\nTest counts for example terms:')
    for t in test_terms:
        c = conn.execute(text('SELECT COUNT(*) FROM ns.annotations_terms WHERE term = :t'), {'t': t}).scalar()
        print(f"'{t}': {c}")

print('\nDone')
