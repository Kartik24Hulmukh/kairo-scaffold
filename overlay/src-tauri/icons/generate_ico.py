import os

png_path = r"C:\Users\praja\OneDrive\Desktop\test-env\repositories\kairo-scaffold\overlay\src-tauri\icons\icon.png"
ico_path = r"C:\Users\praja\OneDrive\Desktop\test-env\repositories\kairo-scaffold\overlay\src-tauri\icons\icon.ico"

if os.path.exists(png_path):
    with open(png_path, "rb") as f:
        png_data = f.read()
else:
    png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00 \x00\x00\x00 \x08\x06\x00\x00\x00\x73R\x95\xa0\x00\x00\x00\x0bIDATx\x9cc\xfc\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82" # minimal transparent 32x32 png

header = b"\x00\x00\x01\x00\x01\x00"
w = 32
h = 32
size = len(png_data)
offset = 22

entry = bytes([
    w, h, 0, 0,
    1, 0,
    32, 0
]) + size.to_bytes(4, "little") + offset.to_bytes(4, "little")

with open(ico_path, "wb") as f:
    f.write(header + entry + png_data)

print("Icon generated successfully!")
