from PIL import Image

print("Opening screenshots...")
img1 = Image.open('scripts/state1.png')
img2 = Image.open('scripts/state2.png')

# Scale down images slightly to keep file size small and loading fast in README
max_size = (800, 500)
img1.thumbnail(max_size, Image.Resampling.LANCZOS)
img2.thumbnail(max_size, Image.Resampling.LANCZOS)

print("Compiling GIF...")
img1.save(
    'demo.gif',
    save_all=True,
    append_images=[img2],
    duration=3000,  # 3000ms (3s) per frame = 6s total loop
    loop=0
)
print("demo.gif compiled successfully!")
