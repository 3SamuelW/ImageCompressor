"""
Download real BMP images from public sources for testing.
Sources: Kodak Image Dataset (PNG converted to BMP), USC-SIPI (if accessible).
"""
import urllib.request
import os
from PIL import Image
import io

data_dir = '/data/'     # enter your route

sources = [
    (
        'https://r0k.us/graphics/kodak/kodak/kodim01.png',
        'real_kodak01.bmp',
        'Kodak Image Dataset kodim01 (natural scene - flowers) - https://r0k.us/graphics/kodak/'
    ),
    (
        'https://r0k.us/graphics/kodak/kodak/kodim23.png',
        'real_kodak23.bmp',
        'Kodak Image Dataset kodim23 (natural scene - lighthouse) - https://r0k.us/graphics/kodak/'
    ),
    (
        'https://r0k.us/graphics/kodak/kodak/kodim05.png',
        'real_kodak05.bmp',
        'Kodak Image Dataset kodim05 (natural scene - toy) - https://r0k.us/graphics/kodak/'
    ),
    (
        'https://r0k.us/graphics/kodak/kodak/kodim15.png',
        'real_kodak15.bmp',
        'Kodak Image Dataset kodim15 (natural scene - beach) - https://r0k.us/graphics/kodak/'
    ),
    # USC-SIPI Miscellaneous images (direct TIFF/BMP not available, use their JPEG previews)
    (
        'https://sipi.usc.edu/database/preview/misc/4.1.01.png',
        'real_sipi_girl.bmp',
        'USC-SIPI Image Database misc/4.1.01 (Girl) - https://sipi.usc.edu/database/'
    )
]

downloaded = []
failed = []

for url, fname, source in sources:
    out_path = os.path.join(data_dir, fname)
    try:
        print(f'Downloading {url} ...')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert('RGB')
        img = img.resize((256, 256), Image.LANCZOS)
        img.save(out_path, 'BMP')
        print(f'  Saved: {out_path}  size={img.size}')
        downloaded.append((fname, source))
    except Exception as e:
        print(f'  FAILED: {e}')
        failed.append((fname, url, str(e)))

print(f'\nDownloaded {len(downloaded)} images, failed {len(failed)}')
for fname, source in downloaded:
    print(f'  OK: {fname}  source: {source}')
for fname, url, err in failed:
    print(f'  FAIL: {fname}  url={url}  err={err}')
