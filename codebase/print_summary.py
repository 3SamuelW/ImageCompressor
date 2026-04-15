import csv
with open('/results/summary_table.csv') as f:      # enter your route
    rows = list(csv.DictReader(f))

print('=== LOSSLESS RESULTS ===')
for r in rows:
    if r['algorithm'] == 'lossless_RLE+Huffman':
        print(f"{r['image']:20s}  ratio={r['compression_ratio']:6s}  PSNR={r['psnr_db']:8s}  SSIM={r['ssim']:8s}  match={r['lossless_match']}")

print()
print('=== LOSSY q=50 RESULTS ===')
for r in rows:
    if r['algorithm'] == 'lossy_DCT' and r['quality'] == '50':
        print(f"{r['image']:20s}  ratio={r['compression_ratio']:6s}  PSNR={r['psnr_db']:8s}  SSIM={r['ssim']:8s}")
