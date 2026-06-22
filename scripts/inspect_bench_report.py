import pathlib

report_path = pathlib.Path(r"C:\Users\praja\OneDrive\Desktop\test-env\repositories\kairo-scaffold\bench\REPORT.md")
if report_path.exists():
    print(report_path.read_text(encoding="utf-8"))
else:
    print("No report yet")
