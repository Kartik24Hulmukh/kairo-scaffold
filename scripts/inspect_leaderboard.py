import pathlib
html_path = pathlib.Path("bench/leaderboard.html")
if html_path.exists():
    print(f"Leaderboard size: {html_path.stat().st_size} bytes")
    # Print first 20 lines
    with open(html_path, "r", encoding="utf-8") as f:
        print("".join(f.readlines()[:20]))
else:
    print("Leaderboard missing")
