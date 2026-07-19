"""Uretilen sentetik kliplerin cesitlilik (benzersizlik) kontrolu.
Her sinif icin dosya sayisi / benzersiz MD5 icerik sayisini raporlar.
Beklenen (duzeltme sonrasi): 50 dosya / 50 benzersiz."""
import hashlib, glob, os

BASES = ["data/synthetic", "data/synthetic_tur2",
         "data/synthetic_tur3", "data/synthetic_tur4"]
SINIFLAR = ["insaat_sesleri", "dini_mekan", "acil_servis", "diger_hayvanlar"]

print("=" * 56)
print("  CESITLILIK KONTROLU (dosya / benzersiz icerik)")
print("=" * 56)
for base in BASES:
    tur = base.split("/")[-1]
    for s in SINIFLAR:
        d = os.path.join(base, s)
        if not os.path.isdir(d):
            continue
        hs = set()
        n = 0
        for f in glob.glob(os.path.join(d, "*.wav")):
            n += 1
            with open(f, "rb") as fp:
                hs.add(hashlib.md5(fp.read()).hexdigest())
        bayrak = "OK" if n == len(hs) else "!! DUPLIKASYON"
        print(f"  {tur:<18} {s:<18} {n:>3} dosya / {len(hs):>3} benzersiz   {bayrak}")
print("=" * 56)
