import json
import sqlite3
import sys

conn = sqlite3.connect("podcast_claude_mythos.db")
c = conn.cursor()

inserted = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    c.execute("SELECT id FROM podcasts WHERE rank = ?", (row["rank"],))
    pod = c.fetchone()
    if not pod:
        continue
    c.execute(
        """INSERT INTO mythos_mentions
           (podcast_id, discussed, confidence, episode_title, episode_date, summary, source_url)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            pod[0],
            row.get("discussed", 0),
            row.get("confidence", "none"),
            row.get("episode_title"),
            row.get("episode_date"),
            row.get("summary"),
            row.get("source_url"),
        ),
    )
    inserted += 1

conn.commit()
conn.close()
print(f"Inserted {inserted} rows.")
