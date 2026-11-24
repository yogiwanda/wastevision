from pathlib import Path

# Pemetaan nama file ke ID kelas di data.yaml
# 0: BIODEGRADABLE, 1: CARDBOARD, 2: GLASS, 3: METAL, 4: PAPER, 5: PLASTIC
mapping = {
    'paper': 4,
    'glass': 2,
    'plastic': 5,
    'metal': 3,
    'cardboard': 1,
}

root = Path("dataset/merged_dataset/labels")

for split in ("train", "val", "test"):
    folder = root / split
    if not folder.exists(): continue
    
    for f in folder.glob("*.txt"):
        filename = f.name.lower()
        new_class = 0 # Default: BIODEGRADABLE / Litter
        
        # Cek nama file, jika ada kata kuncinya, ganti ID kelasnya
        for key, val in mapping.items():
            if key in filename:
                new_class = val
                break
                
        # Baca file label dan ubah angka pertamanya
        lines = f.read_text().splitlines()
        new_lines = []
        for line in lines:
            if line.strip():
                parts = line.split()
                parts[0] = str(new_class) # Ubah ID kelas
                new_lines.append(" ".join(parts))
                
        f.write_text("\n".join(new_lines))

print("✅ Berhasil! Semua label telah diperbarui menjadi 6 kelas.")