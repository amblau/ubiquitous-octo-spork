import sqlite3

conn = sqlite3.connect("podcast_claude_mythos.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS podcasts (
    id INTEGER PRIMARY KEY,
    rank INTEGER,
    name TEXT UNIQUE,
    category TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS mythos_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    podcast_id INTEGER,
    discussed INTEGER,          -- 0/1 boolean
    confidence TEXT,            -- high / medium / low / none
    episode_title TEXT,
    episode_date TEXT,
    summary TEXT,
    source_url TEXT,
    researched_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (podcast_id) REFERENCES podcasts(id)
)
""")

podcasts = [
    (1, "The Joe Rogan Experience", "Top 10"),
    (2, "The Daily (NYT)", "Top 10"),
    (3, "Crime Junkie", "Top 10"),
    (4, "Call Her Daddy", "Top 10"),
    (5, "SmartLess", "Top 10"),
    (6, "Stuff You Should Know", "Top 10"),
    (7, "New Heights", "Top 10"),
    (8, "This American Life", "Top 10"),
    (9, "Dateline NBC", "Top 10"),
    (10, "NPR News Now", "Top 10"),
    (11, "Morbid", "True Crime & Mystery"),
    (12, "MrBallen Podcast", "True Crime & Mystery"),
    (13, "My Favorite Murder", "True Crime & Mystery"),
    (14, "Rotten Mango", "True Crime & Mystery"),
    (15, "48 Hours", "True Crime & Mystery"),
    (16, "Sword and Scale", "True Crime & Mystery"),
    (17, "Serial", "True Crime & Mystery"),
    (18, "Scamanda", "True Crime & Mystery"),
    (19, "This Past Weekend w/ Theo Von", "Comedy & Personality"),
    (20, "Bad Friends", "Comedy & Personality"),
    (21, "The 85 South Show", "Comedy & Personality"),
    (22, "Conan O'Brien Needs a Friend", "Comedy & Personality"),
    (23, "Good Hang with Amy Poehler", "Comedy & Personality"),
    (24, "Your Mom's House", "Comedy & Personality"),
    (25, "2 Bears, 1 Cave", "Comedy & Personality"),
    (26, "Up First (NPR)", "News, Politics & Society"),
    (27, "Pod Save America", "News, Politics & Society"),
    (28, "The Ben Shapiro Show", "News, Politics & Society"),
    (29, "The Dan Bongino Show", "News, Politics & Society"),
    (30, "Pardon My Take", "News, Politics & Society"),
    (31, "The Megyn Kelly Show", "News, Politics & Society"),
    (32, "The Ezra Klein Show", "News, Politics & Society"),
    (33, "Today, Explained", "News, Politics & Society"),
    (34, "The Diary Of A CEO", "Business, Tech & Wealth"),
    (35, "The Prof G Pod (Scott Galloway)", "Business, Tech & Wealth"),
    (36, "Planet Money", "Business, Tech & Wealth"),
    (37, "Acquired", "Business, Tech & Wealth"),
    (38, "All-In Podcast", "Business, Tech & Wealth"),
    (39, "My First Million", "Business, Tech & Wealth"),
    (40, "How I Built This", "Business, Tech & Wealth"),
    (41, "Huberman Lab", "Health, Science & Mindset"),
    (42, "On Purpose with Jay Shetty", "Health, Science & Mindset"),
    (43, "Radiolab", "Health, Science & Mindset"),
    (44, "Hidden Brain", "Health, Science & Mindset"),
    (45, "Ten Percent Happier", "Health, Science & Mindset"),
    (46, "The Rewatchables", "Culture, Sport & Entertainment"),
    (47, "The Bill Simmons Podcast", "Culture, Sport & Entertainment"),
    (48, "Fresh Air", "Culture, Sport & Entertainment"),
    (49, "Office Ladies", "Culture, Sport & Entertainment"),
    (50, "Armchair Expert", "Culture, Sport & Entertainment"),
]

c.executemany(
    "INSERT OR IGNORE INTO podcasts (rank, name, category) VALUES (?, ?, ?)",
    podcasts,
)

conn.commit()
conn.close()
print("DB initialized with", len(podcasts), "podcasts.")
