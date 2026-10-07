"""
Iteratif veri artirma dongusu — sentetik ses + CNN ve LightGBM yeniden egitimi.

Akis:
  1. Orijinal test bolunmesini (seed=42) sabitle
  2. data/synthetic/<sinif>/ altindaki WAV'lardan ozellik cikar
  3. CNN: augmented train seti ile yeniden egit, test F1 karsilastir
  4. LightGBM: augmented ozellik matrisi ile yeniden egit, test F1 karsilastir
  5. Her iki model icin oncesi/sonrasi delta tablosu yazdir
  6. results/v1/iteratif_loop_<tarih>.json'a kaydet

Kullanim:
    python scripts/07_iterative_loop.py
    python scripts/07_iterative_loop.py --sadece_aug        # baseline ckpt kullan
    python scripts/07_iterative_loop.py --sadece_lgbm       # sadece LightGBM
    python scripts/07_iterative_loop.py --sadece_cnn        # sadece CNN
    python scripts/07_iterative_loop.py --epochs 20 --patience 5
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import dataloader_olustur, veri_bol
from src.models.cnn import SpectrogramCNN

# ── Sabitler ──────────────────────────────────────────────────────────────────
PARQUET_LISTE   = ROOT / "data" / "processed" / "veri_listesi.parquet"
PARQUET_LGBM    = ROOT / "data" / "processed" / "lgbm_ozellikler.parquet"
SENTETIK_DIR    = ROOT / "data" / "synthetic"
MODEL_DIR       = ROOT / "models"
CHECKPTS_DIR    = ROOT / "checkpoints"
RESULT_DIR      = ROOT / "results" / "v1"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# Kuyrugu temiz duraklatma: bu dosya varsa egitim epoch sonunda durumu kaydedip cikar (kod 4)
DURDUR_DOSYASI  = ROOT / "outputs" / "v2_loglar" / "kuyruk" / "DURDUR"
N_SINIF         = 26
SEED            = 42

SINIF_SIRASI = [
    "acil_servis", "at_sesleri", "ayak_sesleri", "cocuk_bebek_sesleri",
    "diger_hayvanlar", "dini_mekan", "doga_atmosferi", "endustri_fabrika",
    "ev_ici_sesler", "gemi_tekne_sesleri", "hava_olaylari", "insaat_sesleri",
    "kalabalik_sesleri", "kapi_sesleri", "kara_tasitlari", "kopek_sesleri",
    "kus_sesleri", "ofis_iletisim", "saat_can_sesleri", "sehir_atmosferi",
    "silah_patlama", "spor_eglence", "su_sesleri", "tarim_ciftlik",
    "tren_sesleri", "ucak_sesleri",
]
SINIF_IDX = {s: i for i, s in enumerate(SINIF_SIRASI)}


# ════════════════════════════════════════════════════════════════════════════
# ORTAK: Sentetik dosya listesi
# ════════════════════════════════════════════════════════════════════════════

def sentetik_wav_listesi(siniflar: list[str]) -> list[tuple[Path, str, int]]:
    """[(wav_yolu, sinif, sinif_idx), ...] — gecerli dosyalar."""
    sonuc = []
    for sinif in siniflar:
        idx = SINIF_IDX.get(sinif)
        if idx is None:
            continue
        for wav in sorted((SENTETIK_DIR / sinif).glob("*.wav")):
            if wav.stat().st_size > 1000:
                sonuc.append((wav, sinif, idx))
    return sonuc


# ════════════════════════════════════════════════════════════════════════════
# CNN BOLUMU
# ════════════════════════════════════════════════════════════════════════════

def sentetik_cnn_df(siniflar: list[str]) -> pd.DataFrame:
    """CNN icin veri_listesi formatinda DataFrame."""
    satirlar = []
    for wav, sinif, idx in sentetik_wav_listesi(siniflar):
        try:
            rel = wav.relative_to(ROOT)
        except ValueError:
            rel = wav
        satirlar.append({"dosya": str(rel), "sinif": sinif,
                          "sinif_idx": idx, "n_segs": 1})
    df = pd.DataFrame(satirlar)
    print(f"  Sentetik (CNN): {len(df)} satir  "
          + "  ".join(f"{s}={len(df[df.sinif==s])}" for s in siniflar))
    return df


def arka_plan(loader, derinlik: int = 4):
    """DataLoader'dan batch'leri arka plan is parcaciginda onceden hazirlar; GPU bir batch'i
    islerken sonraki batch'ler hazirlanir. num_workers=0 kullanildigi icin (Windows'ta isci
    surecleri ~19 GB'lik RAM on-yuklemesini kopyalar) veri hazirlama ana is parcaciginda GPU ile
    SIRAYLA calisiyordu (~23 ms + ~67 ms / batch). Karistirma sirasi iter() ile ANA is parcaciginda
    ayni RNG'den uretilir; is parcacigi ayni batch'leri ayni sirayla uretir -> sonuclar ozdes."""
    import queue
    import threading
    kuyruk = queue.Queue(maxsize=derinlik)
    bitti = object()
    it = iter(loader)                      # sampler permutasyonu burada (ana is parcacigi)

    def uret():
        try:
            for b in it:
                kuyruk.put(b)
            kuyruk.put(bitti)
        except BaseException as e:         # hatayi ana is parcacigina tasi
            kuyruk.put(e)

    t = threading.Thread(target=uret, daemon=True)
    t.start()
    while True:
        b = kuyruk.get()
        if b is bitti:
            break
        if isinstance(b, BaseException):
            raise b
        yield b
    t.join()


def egit_epoch_cnn(model, loader, optimizer, kayip_fn, scaler):
    model.train()
    toplam, n = 0.0, 0
    for x, y in arka_plan(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(DEVICE == "cuda")):
            kayip = kayip_fn(model(x), y)
        scaler.scale(kayip).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        toplam += kayip.item() * x.size(0)
        n += x.size(0)
    return toplam / n


def degerlendir_cnn(model, loader, kayip_fn):
    model.eval()
    toplam, n = 0.0, 0
    tahminler, gercekler = [], []
    with torch.no_grad():
        for x, y in arka_plan(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with autocast(enabled=(DEVICE == "cuda")):
                cikis = model(x)
                kayip = kayip_fn(cikis, y)
            toplam += kayip.item() * x.size(0)
            n += x.size(0)
            tahminler.extend(cikis.argmax(1).cpu().tolist())
            gercekler.extend(y.cpu().tolist())
    f1 = f1_score(gercekler, tahminler, average="macro", zero_division=0)
    return toplam / n, f1, tahminler, gercekler


def _ram_bekle(df_train, pay_gb: float = 3.0):
    """Egitim kumesi RAM'e on-yuklenir (~19 GB). Bos RAM yetmiyorsa (gunduz PC kullanimi)
    bellek yetersizliginden cokmek yerine RAM bosalana kadar bekler; beklerken DURDUR
    istegine uyar (kod 4). (USB SSD uzerinde karisik sirali mmap okuma egitim icin cok
    yavas oldugundan on-yuklemeden vazgecilmedi.)"""
    import psutil
    gerekli = 0
    for d in df_train["dosya"].unique():
        c = ROOT / "data" / "mel_cache" / Path(d).with_suffix(".npy")
        if c.exists():
            gerekli += c.stat().st_size
    esik = gerekli + pay_gb * 1e9

    def kullanilabilir():
        """(fiziksel bos RAM, kullanilabilir taahhut). Windows'ta tahsis siniri fiziksel RAM degil
        TAAHHUT limitidir (RAM + sayfalama dosyasi); 7 Eki'deki cokme fiziksel RAM yeterliyken
        taahhut yetmedigi icin oldu. Ikisinin kucugu esas alinir."""
        fiziksel = psutil.virtual_memory().available
        taahhut = fiziksel
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            taahhut = m.ullAvailPageFile
        return fiziksel, taahhut

    son_log = 0.0
    while min(kullanilabilir()) < esik:
        if DURDUR_DOSYASI.exists():
            print("  [DURDUR] RAM beklenirken duraklatma istegi alindi (kod 4).", flush=True)
            sys.exit(4)
        if time.time() - son_log > 300:
            f, t = kullanilabilir()
            print(f"  [RAM] Egitim icin {esik/1e9:.1f} GB gerekli; bos fiziksel {f/1e9:.1f} GB, "
                  f"kullanilabilir taahhut {t/1e9:.1f} GB; bekleniyor...", flush=True)
            son_log = time.time()
        time.sleep(30)
    f, t = kullanilabilir()
    print(f"  RAM: bos fiziksel {f/1e9:.1f} GB, kullanilabilir taahhut {t/1e9:.1f} GB; egitim kumesi "
          f"on-yukleme {gerekli/1e9:.1f} GB", flush=True)


def _guvenli_kaydet(nesne, yol: Path):
    """Atomik kayit (tmp + os.replace): kayit sirasinda kesinti olsa da eski dosya saglam kalir."""
    import os
    gecici = yol.with_name(yol.name + ".tmp")
    torch.save(nesne, gecici)
    os.replace(gecici, yol)


def egit_cnn(df_train, df_val, df_test, tag, epochs, batch_size, lr, patience,
             spec_augment=False, seed=None, model_yolu=None, durum_yolu=None,
             tahmin_dondur=False, kesinti_epoch=None):
    """
    v1 egitim proseduru. Yeni parametrelerin varsayilanlari v1 davranisini birebir korur:
      seed=None        -> SEED (42); v2 coklu seed kosularinda farkli deger verilir
                          (agirlik baslatma, karistirma sirasi, SpecAugment maskeleri).
      model_yolu=None  -> models/cnn_<tag>_best.pt
      durum_yolu=None  -> epoch duzeyinde devam KAPALI. Verilirse her epoch sonunda tam
                          durum (model, optimizer, LR takvimi, AMP olcegi, en iyi metrik,
                          sabir, gecmis, RNG'ler) atomik kaydedilir; dosya varsa kaldigi
                          epoch'tan birebir devam edilir.
      tahmin_dondur    -> test/val segment tahminleri + kayit (dosya) kimlikleri dondurulur.
    """
    tohum = SEED if seed is None else seed
    torch.manual_seed(tohum); np.random.seed(tohum)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    yol = Path(model_yolu) if model_yolu else MODEL_DIR / f"cnn_{tag}_best.pt"
    yol.parent.mkdir(parents=True, exist_ok=True)

    # HIZLANDIRMA (sonuc-notr): preload=True + num_workers=0.
    # Mel cache (float16) RAM'e bir kez yuklenir -> her epoch tekrarli np.load
    # ve disk I/O ortadan kalkar (GPU acliktan kurtulur). Veri birebir ayni
    # float16 cache oldugundan sonuc preload=False ile OZDESTIR.
    # Windows'ta num_workers>0 + preload Dataset'i her worker'a kopyalar (OOM);
    # bu nedenle num_workers=0 zorunlu (RAM diliminden okuma zaten anlik).
    _ram_bekle(df_train)
    dl_tr = dataloader_olustur(df_train, ROOT, mod="cnn",
                               batch_size=batch_size, num_workers=0, preload=True,
                               spec_augment=spec_augment,
                               sa_seed=0 if seed is None else tohum)
    # Val/test diskten (mmap, yalniz gereken dilim) okunur. RAM'e on-yukleme (~4 GB ek)
    # gunduz PC kullanimiyla birlikte bellek yetersizligine yol acti (5 Eki); mmap
    # neredeyse ayni hizda (~30 sn/val) ve ek RAM gerektirmez. Veri ozdes.
    dl_vl = dataloader_olustur(df_val,   ROOT, mod="cnn",
                               batch_size=batch_size*2, shuffle=False,
                               num_workers=0, preload=False)
    dl_te = dataloader_olustur(df_test,  ROOT, mod="cnn",
                               batch_size=batch_size*2, shuffle=False,
                               num_workers=0, preload=False)
    print(f"  Train={len(dl_tr.dataset):,}  Val={len(dl_vl.dataset):,}  "
          f"Test={len(dl_te.dataset):,}")

    model     = SpectrogramCNN(n_sinif=N_SINIF, pretrained=True).to(DEVICE)
    kayip_fn  = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    scaler    = GradScaler(enabled=(DEVICE == "cuda"))

    en_iyi, pc = 0.0, 0
    tarih = []
    t0_toplam = time.time()
    gecen_sn = 0.0
    bas_ep = 1
    durum_yolu = Path(durum_yolu) if durum_yolu else None

    if durum_yolu is not None and durum_yolu.exists():
        d = torch.load(durum_yolu, map_location="cpu", weights_only=False)
        model.load_state_dict(d["model_state"])
        optimizer.load_state_dict(d["optimizer"])
        scheduler.load_state_dict(d["scheduler"])
        scaler.load_state_dict(d["scaler"])
        en_iyi, pc, tarih, gecen_sn = d["en_iyi"], d["pc"], list(d["tarih"]), d["gecen_sn"]
        bas_ep = d["epoch"] + 1
        torch.set_rng_state(d["rng"]["torch"].cpu())
        if d["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in d["rng"]["cuda"]])
        np.random.set_state(d["rng"]["numpy"])
        dl_tr.dataset._sa_rng.bit_generator.state = d["rng"]["sa"]
        print(f"  Devam: epoch {d['epoch']} durumu yuklendi (en iyi val_F1={en_iyi:.4f}, sabir {pc})")
        if pc >= patience:
            bas_ep = epochs + 1

    for ep in range(bas_ep, epochs + 1):
        t0 = time.time()
        loss = egit_epoch_cnn(model, dl_tr, optimizer, kayip_fn, scaler)
        _, vf1, _, _ = degerlendir_cnn(model, dl_vl, kayip_fn)
        scheduler.step()
        print(f"  Epoch {ep:3d}/{epochs}  loss={loss:.4f}  val_f1={vf1:.4f}  "
              f"{time.time()-t0:.0f}s")
        tarih.append({"epoch": ep, "train_loss": round(loss,5), "val_f1": round(vf1,4)})
        dur = False
        if vf1 > en_iyi:
            en_iyi = vf1; pc = 0
            if durum_yolu is not None:
                _guvenli_kaydet({"model_state": model.state_dict(), "val_f1": vf1}, yol)
            else:
                torch.save({"model_state": model.state_dict(), "val_f1": vf1}, yol)
        else:
            pc += 1
            if pc >= patience:
                print(f"  Erken durdurma (epoch {ep})")
                dur = True
        gecen_sn += time.time() - t0
        if durum_yolu is not None:
            _guvenli_kaydet({
                "epoch": ep, "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "en_iyi": en_iyi, "pc": pc,
                "tarih": tarih, "gecen_sn": gecen_sn,
                "rng": {"torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "numpy": np.random.get_state(),
                        "sa": dl_tr.dataset._sa_rng.bit_generator.state},
            }, durum_yolu)
        if kesinti_epoch is not None and ep >= kesinti_epoch:
            print(f"  [TEST] Kesinti taklidi: epoch {ep} sonrasi cikiliyor (kod 3)")
            sys.exit(3)
        if durum_yolu is not None and DURDUR_DOSYASI.exists() and not dur and ep < epochs:
            print(f"  [DURDUR] Istek alindi: epoch {ep} durumu kaydedildi, cikiliyor (kod 4). "
                  f"Ayni komut kaldigi yerden devam eder.")
            sys.exit(4)
        if dur:
            break

    # Son epoch modeli (v2): checkpoint secimi val'e dayanmadigi icin val+test birlikte
    # temiz degerlendirme kumesi olur (kayit sayisi iki katina cikar). En iyi-val
    # checkpoint'inden ONCE hesaplanir; v1 ciktilari degismez.
    son_epoch = None
    if tahmin_dondur:
        _, s_te_f1, s_te_tahmin, _ = degerlendir_cnn(model, dl_te, kayip_fn)
        _, s_vl_f1, s_vl_tahmin, _ = degerlendir_cnn(model, dl_vl, kayip_fn)
        son_epoch = {"epoch": tarih[-1]["epoch"] if tarih else None,
                     "test_f1": round(s_te_f1, 4), "val_f1": round(s_vl_f1, 4),
                     "test_tahmin": np.array(s_te_tahmin, dtype=np.int16),
                     "val_tahmin": np.array(s_vl_tahmin, dtype=np.int16)}

    ckpt = torch.load(yol, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    _, test_f1, tahmin, gercek = degerlendir_cnn(model, dl_te, kayip_fn)
    rapor = classification_report(gercek, tahmin, target_names=SINIF_SIRASI,
                                  output_dict=True, zero_division=0)
    sinif_f1 = {SINIF_SIRASI[i]: round(rapor[SINIF_SIRASI[i]]["f1-score"], 4)
                for i in range(N_SINIF)}
    sure_dk = (time.time() - t0_toplam) / 60 if durum_yolu is None else gecen_sn / 60
    print(f"  [{tag}] val_F1={en_iyi:.4f}  test_F1={test_f1:.4f}  ({sure_dk:.1f}dk)")
    sonuc = {"val_f1": round(en_iyi,4), "test_f1": round(test_f1,4),
             "sinif_f1": sinif_f1, "tarih": tarih, "sure_dk": round(sure_dk,1)}
    if tahmin_dondur:
        _, _, v_tahmin, v_gercek = degerlendir_cnn(model, dl_vl, kayip_fn)
        rapor_v = classification_report(v_gercek, v_tahmin, target_names=SINIF_SIRASI,
                                         output_dict=True, zero_division=0)
        sonuc["val_sinif_f1"] = {s: round(rapor_v[s]["f1-score"], 4) for s in SINIF_SIRASI}
        sonuc["_tahminler"] = {
            "test_tahmin": np.array(tahmin, dtype=np.int16),
            "test_gercek": np.array(gercek, dtype=np.int16),
            "test_dosya":  np.array([w for w, _, _ in dl_te.dataset.indeks]),
            "val_tahmin":  np.array(v_tahmin, dtype=np.int16),
            "val_gercek":  np.array(v_gercek, dtype=np.int16),
            "val_dosya":   np.array([w for w, _, _ in dl_vl.dataset.indeks]),
            "son_test_tahmin": son_epoch["test_tahmin"],
            "son_val_tahmin":  son_epoch["val_tahmin"],
        }
        sonuc["son_epoch"] = {k: son_epoch[k] for k in ("epoch", "test_f1", "val_f1")}
    return sonuc


def baseline_cnn_yukle(df_test) -> dict:
    """Tutarli dongu-baseline'ini (cnn_baseline_iter_best.pt) test setinde degerle.
    Boylece TUM turlar AYNI baseline'a karsi karsilastirilir (#4 cozumu).
    Dongu baseline'i yoksa orijinal cnn_best.pt'ye geri doner."""
    kayip_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    model = SpectrogramCNN(n_sinif=N_SINIF, pretrained=False).to(DEVICE)
    yol = MODEL_DIR / "cnn_baseline_iter_best.pt"   # tutarli dongu baseline'i
    if not yol.exists():
        yol = MODEL_DIR / "cnn_best.pt"             # geri donus: orijinal model
    if not yol.exists():
        print(f"HATA: baseline ckpt bulunamadi."); sys.exit(1)
    print(f"  Baseline ckpt: {yol.name}")
    ckpt = torch.load(yol, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    dl = dataloader_olustur(df_test, ROOT, mod="cnn",
                            batch_size=64, shuffle=False, num_workers=0)
    _, test_f1, tahmin, gercek = degerlendir_cnn(model, dl, kayip_fn)
    rapor = classification_report(gercek, tahmin, target_names=SINIF_SIRASI,
                                  output_dict=True, zero_division=0)
    sinif_f1 = {SINIF_SIRASI[i]: round(rapor[SINIF_SIRASI[i]]["f1-score"], 4)
                for i in range(N_SINIF)}
    print(f"  Baseline CNN test_F1={test_f1:.4f}")
    return {"val_f1": ckpt.get("val_f1", 0.0), "test_f1": round(test_f1,4),
            "sinif_f1": sinif_f1, "tarih": [], "sure_dk": 0.0}


# ════════════════════════════════════════════════════════════════════════════
# LIGHTGBM BOLUMU
# ════════════════════════════════════════════════════════════════════════════

def sentetik_ozellik_cikar(siniflar: list[str]) -> pd.DataFrame:
    """
    Sentetik WAV'lardan 467-dim ozellik cikar, lgbm_ozellikler formatiyla dondur.
    Mevcut cache varsa atlar.
    """
    from src.data.features import duz_ozellik_cikar, SEGMENT_SN
    from src.data.preprocessing import onisle

    cache_yolu = ROOT / "data" / "processed" / "sentetik_lgbm_ozellikler.parquet"
    if cache_yolu.exists():
        df = pd.read_parquet(cache_yolu)
        # Sadece hedef siniflarin satirlarini kontrol et
        eksik = [s for s in siniflar if s not in df["sinif"].values]
        if not eksik:
            print(f"  Sentetik ozellik cache bulundu: {len(df)} satir")
            return df[df["sinif"].isin(siniflar)].reset_index(drop=True)

    print("  Sentetik WAV'lardan ozellik cikariliyor...")
    satirlar = []
    liste = sentetik_wav_listesi(siniflar)
    for i, (wav, sinif, idx) in enumerate(liste):
        try:
            y, sr = onisle(str(wav), sr=16000)
            ozellikler = duz_ozellik_cikar(y, sr)
            satir = {"sinif": sinif, "sinif_idx": idx,
                     "dosya": str(wav.relative_to(ROOT)),
                     "segment_idx": 0}
            for j, v in enumerate(ozellikler):
                satir[f"f{j}"] = float(v)
            satirlar.append(satir)
        except Exception as e:
            print(f"  [!] {wav.name}: {e}")
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(liste)} ozellik cikarildi")

    df = pd.DataFrame(satirlar)
    df.to_parquet(cache_yolu, index=False)
    print(f"  {len(df)} sentetik ozellik -> {cache_yolu}")
    return df


def egit_lgbm_aug(df_train_aug, df_val, df_test, tag: str) -> dict:
    """LightGBM'i augmented egitim seti ile egit."""
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler

    # Ozellik sutunlarini bul
    ozellik_sut = sorted([c for c in df_train_aug.columns
                          if c.startswith("f") and c[1:].isdigit()],
                         key=lambda c: int(c[1:]))

    def xy(df):
        return df[ozellik_sut].values.astype(np.float32), df["sinif_idx"].values

    X_tr, y_tr = xy(df_train_aug)
    X_vl, y_vl = xy(df_val)
    X_te, y_te = xy(df_test)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_vl = scaler.transform(X_vl)
    X_te = scaler.transform(X_te)

    # Mevcut best params varsa yukle
    params_yolu = ROOT / "checkpoints" / "lgbm_best_params.json"
    if params_yolu.exists():
        with open(params_yolu) as f:
            params = json.load(f)
        print(f"  Mevcut HPO parametreleri yuklendi")
    else:
        params = dict(n_estimators=500, num_leaves=127, max_depth=8,
                      learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, min_child_samples=20,
                      reg_alpha=0.1, reg_lambda=1.0)
        print("  Varsayilan parametreler kullaniliyor")

    params.update({"device": "gpu", "verbose": -1,
                   "class_weight": "balanced",
                   "objective": "multiclass", "num_class": N_SINIF,
                   "metric": "multi_logloss", "random_state": SEED})

    t0 = time.time()
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_vl, y_vl)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=50)])

    val_pred  = model.predict(X_vl)
    test_pred = model.predict(X_te)
    val_f1  = f1_score(y_vl, val_pred,  average="macro", zero_division=0)
    test_f1 = f1_score(y_te, test_pred, average="macro", zero_division=0)

    rapor = classification_report(y_te, test_pred, target_names=SINIF_SIRASI,
                                  output_dict=True, zero_division=0)
    sinif_f1 = {SINIF_SIRASI[i]: round(rapor[SINIF_SIRASI[i]]["f1-score"], 4)
                for i in range(N_SINIF)}
    sure_dk = (time.time() - t0) / 60
    print(f"  [{tag}] val_F1={val_f1:.4f}  test_F1={test_f1:.4f}  ({sure_dk:.1f}dk)")

    # Model kaydet
    CHECKPTS_DIR.mkdir(exist_ok=True)
    import joblib
    joblib.dump(model,  CHECKPTS_DIR / f"lgbm_{tag}.pkl")
    joblib.dump(scaler, CHECKPTS_DIR / f"lgbm_{tag}_scaler.pkl")
    return {"val_f1": round(val_f1,4), "test_f1": round(test_f1,4),
            "sinif_f1": sinif_f1, "sure_dk": round(sure_dk,1)}


def baseline_lgbm_yukle(df_test) -> dict:
    """
    lgbm_metrics.json'dan baseline F1 degerlerini oku.
    pkl numpy versiyonu uyumsuzlugu nedeniyle dogrudan json kullanilir.
    """
    m = ROOT / "outputs" / "lgbm_metrics.json"
    if not m.exists():
        print(f"  UYARI: lgbm_metrics.json bulunamadi, LGBM baseline atlanıyor.")
        return None
    with open(m, encoding="utf-8") as f:
        d = json.load(f)

    # Per-class F1 icin classification_report gerekli — yoksa None don, yeniden egit
    rapor = d.get("classification_report")
    if not rapor:
        print("  lgbm_metrics.json'da classification_report yok — yeniden egitiyor...")
        return None

    sinif_f1 = {}
    for sinif in SINIF_SIRASI:
        if sinif in rapor:
            sinif_f1[sinif] = round(rapor[sinif]["f1-score"], 4)
        else:
            sinif_f1[sinif] = 0.0

    test_f1 = d.get("test_macro_f1", 0.0)
    val_f1  = d.get("val_macro_f1",  0.0)
    print(f"  Baseline LightGBM test_F1={test_f1:.4f}  (lgbm_metrics.json'dan)")
    return {"val_f1": val_f1, "test_f1": test_f1,
            "sinif_f1": sinif_f1, "sure_dk": 0.0}


# ════════════════════════════════════════════════════════════════════════════
# KARSILASTIRMA YAZDIR
# ════════════════════════════════════════════════════════════════════════════

def karsilastir_yazdir(model_adi, hedef_siniflar, baseline, augmented):
    b_f1 = baseline["test_f1"]
    a_f1 = augmented["test_f1"]
    print(f"\n{'='*62}")
    print(f"  {model_adi} Karsilastirma:  Baseline={b_f1:.4f}  "
          f"Augmented={a_f1:.4f}  Delta={a_f1-b_f1:+.4f}")
    print(f"  {'-'*58}")
    for sinif in hedef_siniflar:
        b = baseline["sinif_f1"].get(sinif, 0.0)
        a = augmented["sinif_f1"].get(sinif, 0.0)
        d = a - b
        ikon = "+" if d > 0.01 else ("-" if d < -0.01 else "~")
        print(f"  [{ikon}] {sinif:<30} {b:.4f} -> {a:.4f}  ({d:+.4f})")
    print(f"{'='*62}")


# ════════════════════════════════════════════════════════════════════════════
# ANA
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--siniflar",   nargs="+", default=None)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--patience",   type=int,   default=7)
    parser.add_argument("--sadece_aug", action="store_true",
                        help="Baseline ckpt'den yukle, sadece augmented egit")
    parser.add_argument("--sadece_cnn",  action="store_true")
    parser.add_argument("--sadece_lgbm", action="store_true")
    parser.add_argument("--sentetik_dir", type=str, default=None,
                        help="Ozel sentetik dizin (orn: data/synthetic_tur2)")
    parser.add_argument("--spec_augment", action="store_true",
                        help="SpecAugment temeli (M2): egitimde zaman/frekans "
                             "maskeleme. Ayri tag kullanir, kanonik baseline'i ezmez.")
    parser.add_argument("--tur_etiketi", type=str, default=None,
                        help="Sonuc dosyasi etiketi (orn: tur2, tur3)")
    args = parser.parse_args()

    global SENTETIK_DIR
    if args.sentetik_dir:
        SENTETIK_DIR = ROOT / args.sentetik_dir

    # Hedef siniflar
    if args.siniflar:
        hedef = args.siniflar
    else:
        hedef = [d.name for d in SENTETIK_DIR.iterdir()
                 if d.is_dir() and
                 any(f.stat().st_size > 1000 for f in d.glob("*.wav"))]
    hedef = sorted(hedef)

    if not hedef:
        print("HATA: data/synthetic/ altinda gecerli WAV bulunamadi.")
        sys.exit(1)

    cnn_yapilsin  = not args.sadece_lgbm
    lgbm_yapilsin = not args.sadece_cnn

    print("=" * 62)
    print("  Iteratif Veri Artirma Dongusu — Tur 1")
    print(f"  Cihaz    : {DEVICE}")
    print(f"  Siniflar : {hedef}")
    print(f"  CNN      : {'EVET' if cnn_yapilsin else 'HAYIR'}  "
          f"LightGBM: {'EVET' if lgbm_yapilsin else 'HAYIR'}")
    print("=" * 62)

    # ── Ortak test bolunmesi (CNN icin) ───────────────────────────────────────
    print("\nVeri bolunuyor (seed=42)...")
    df_train_org, df_val_cnn, df_test_cnn = veri_bol(PARQUET_LISTE, ROOT)
    print(f"  CNN — Train={len(df_train_org):,} dosya  "
          f"Val={len(df_val_cnn):,}  Test={len(df_test_cnn):,}")

    tur_etiketi = args.tur_etiketi or ("1" if not args.sentetik_dir else args.sentetik_dir.split("_")[-1])
    sonuclar = {"tarih": datetime.now().strftime("%Y%m%d_%H%M"),
                "tur": tur_etiketi, "hedef_siniflar": hedef,
                "cnn": {}, "lgbm": {}}

    # ══════════════════════════════════════════════════════════════════════════
    # CNN
    # ══════════════════════════════════════════════════════════════════════════
    if cnn_yapilsin:
        print("\n" + "="*62)
        print("  CNN BOLUMU")
        print("="*62)

        # SpecAugment (M2) ayri tag kullanir; kanonik baseline ezilmez.
        bl_tag  = "baseline_specaug"  if args.spec_augment else "baseline_iter"
        aug_tag = "augmented_specaug" if args.spec_augment else "augmented_iter"

        # Baseline
        if args.sadece_aug and not args.spec_augment:
            print("\n[CNN 1/2] Baseline ckpt yukleniyor...")
            bl_cnn = baseline_cnn_yukle(df_test_cnn)
        else:
            # SpecAugment'te baseline = gercek veri + maskeleme (sentetik YOK):
            # bu, 'SpecAugment-only' temelidir (generative ile kiyas noktasi).
            etiket = "SpecAugment-only" if args.spec_augment else "Baseline"
            print(f"\n[CNN 1/2] {etiket} CNN egitiliyor...")
            bl_cnn = egit_cnn(df_train_org, df_val_cnn, df_test_cnn,
                              bl_tag, args.epochs, args.batch_size,
                              args.lr, args.patience,
                              spec_augment=args.spec_augment)

        # Augmented
        not_ = "SpecAugment+generative" if args.spec_augment else "Augmented"
        print(f"\n[CNN 2/2] {not_} CNN egitiliyor...")
        df_synt_cnn = sentetik_cnn_df(hedef)
        df_train_aug_cnn = pd.concat(
            [df_train_org, df_synt_cnn[["dosya","sinif","sinif_idx","n_segs"]]],
            ignore_index=True)
        aug_cnn = egit_cnn(df_train_aug_cnn, df_val_cnn, df_test_cnn,
                           aug_tag, args.epochs, args.batch_size,
                           args.lr, args.patience,
                           spec_augment=args.spec_augment)

        karsilastir_yazdir("CNN", hedef, bl_cnn, aug_cnn)
        sonuclar["cnn"] = {"baseline": bl_cnn, "augmented": aug_cnn,
                           "delta": round(aug_cnn["test_f1"]-bl_cnn["test_f1"],4)}

    # ══════════════════════════════════════════════════════════════════════════
    # LIGHTGBM
    # ══════════════════════════════════════════════════════════════════════════
    if lgbm_yapilsin:
        print("\n" + "="*62)
        print("  LIGHTGBM BOLUMU")
        print("="*62)

        if not PARQUET_LGBM.exists():
            print(f"  UYARI: {PARQUET_LGBM} bulunamadi, LightGBM atlaniyor.")
        else:
            df_lgbm = pd.read_parquet(PARQUET_LGBM)
            ozellik_sut = sorted([c for c in df_lgbm.columns
                                  if c.startswith("f") and c[1:].isdigit()],
                                 key=lambda c: int(c[1:]))
            print(f"  LGBM veri: {len(df_lgbm):,} segment, {len(ozellik_sut)}-dim")

            # Dosya duzeyinde ayni test bolunmesi
            from sklearn.model_selection import train_test_split
            dosya_sinif = (df_lgbm.groupby("dosya")["sinif_idx"]
                           .first().reset_index())
            d_tv, d_te = train_test_split(
                dosya_sinif, test_size=0.15, stratify=dosya_sinif["sinif_idx"],
                random_state=SEED)
            val_oran = 0.15 / (1 - 0.15)
            d_tr, d_vl = train_test_split(
                d_tv, test_size=val_oran, stratify=d_tv["sinif_idx"],
                random_state=SEED)
            def filtre(d):
                return df_lgbm[df_lgbm["dosya"].isin(d["dosya"])].reset_index(drop=True)
            df_tr_lgbm, df_vl_lgbm, df_te_lgbm = filtre(d_tr), filtre(d_vl), filtre(d_te)
            print(f"  LGBM — Train={len(df_tr_lgbm):,}  Val={len(df_vl_lgbm):,}  "
                  f"Test={len(df_te_lgbm):,}")

            # Baseline — TUTARLILIK: augmented ile AYNI protokol ve test
            # bolunmesiyle yeniden egit (eski lgbm_metrics.json kisayolu farkli
            # split kullanmis olabilir -> yanli delta riski). Boylece baseline
            # ve augmented yalnizca sentetik veriyle ayrisir (temiz delta).
            print("\n[LGBM 1/2] Baseline LightGBM egitiliyor (augmented ile ayni protokol)...")
            bl_lgbm = egit_lgbm_aug(df_tr_lgbm, df_vl_lgbm, df_te_lgbm, "baseline_iter")

            # Augmented
            print("\n[LGBM 2/2] Sentetik ozellikler cikariliyor + augmented egitim...")
            df_synt_lgbm = sentetik_ozellik_cikar(hedef)

            # Sadece ozellik sutunlarini hizala
            eksik_sut = [c for c in ozellik_sut if c not in df_synt_lgbm.columns]
            if eksik_sut:
                print(f"  [!] {len(eksik_sut)} sutun eksik, 0 ile dolduruluyor")
                for c in eksik_sut:
                    df_synt_lgbm[c] = 0.0

            df_tr_aug_lgbm = pd.concat(
                [df_tr_lgbm, df_synt_lgbm[["sinif","sinif_idx","dosya",
                                            "segment_idx"] + ozellik_sut]],
                ignore_index=True)
            print(f"  Augmented LGBM train: {len(df_tr_lgbm):,} + "
                  f"{len(df_synt_lgbm):,} = {len(df_tr_aug_lgbm):,}")

            aug_lgbm = egit_lgbm_aug(df_tr_aug_lgbm, df_vl_lgbm, df_te_lgbm,
                                     "augmented_iter")
            karsilastir_yazdir("LightGBM", hedef, bl_lgbm, aug_lgbm)
            sonuclar["lgbm"] = {"baseline": bl_lgbm, "augmented": aug_lgbm,
                                "delta": round(aug_lgbm["test_f1"]-bl_lgbm["test_f1"],4)}

    # ── Genel ozet ────────────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  GENEL OZET")
    if sonuclar["cnn"]:
        c = sonuclar["cnn"]
        print(f"  CNN      : {c['baseline']['test_f1']:.4f} -> "
              f"{c['augmented']['test_f1']:.4f}  ({c['delta']:+.4f})")
    if sonuclar["lgbm"]:
        l = sonuclar["lgbm"]
        print(f"  LightGBM : {l['baseline']['test_f1']:.4f} -> "
              f"{l['augmented']['test_f1']:.4f}  ({l['delta']:+.4f})")
    print("="*62)

    # ── Kaydet ────────────────────────────────────────────────────────────────
    tarih_str = datetime.now().strftime("%Y%m%d_%H%M")
    cikti = RESULT_DIR / f"iteratif_loop_{tur_etiketi}_{tarih_str}.json"
    RESULT_DIR.mkdir(exist_ok=True)
    with open(cikti, "w", encoding="utf-8") as f:
        json.dump(sonuclar, f, ensure_ascii=False, indent=2)
    print(f"\nSonuclar kaydedildi: {cikti}")


if __name__ == "__main__":
    main()
