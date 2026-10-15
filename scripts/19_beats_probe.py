"""
Ikinci siniflandirici (R3: "yalniz ResNet-18"; R5.3: "guclu on-egitimli / oz-denetimli modeller"):
dondurulmus BEATs gommeleri + MLP sinif kafasi. ON KAYITTA YOKTUR; hakem istegiyle eklenen
saglamlik analizidir ve makalede oyle raporlanir.

Model: BEATs_iter3+ (AS2M), AudioSet-2M uzerinde oz-denetimli on-egitim (Chen ve ark., ICML 2023;
github.com/microsoft/unilm/tree/master/beats). Resmi OneDrive baglantisi programla indirilemedigi
icin agirliklar Hugging Face aynasindan alindi (lpepino/beats_ckpts); SHA-256 iki bagimsiz aynada
(lpepino/beats_ckpts, mooneyko/BEATs) ayni: d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34.
AudioSet BBC Sound Effects arsivinden derlenmis bir veri seti degildir.

Segmentler ResNet-18 ile birebir ayni: veri_bol (seed 42) bolmeleri; kayit basina n_segs adet 5 sn'lik
ardisik dilim (mel onbellegiyle ayni hizalama: dilim s, 16 kHz'de s*313*256 orneginden baslar);
sentetik kliplerde ilk 5 sn (n_segs = 1). Dalga bicimi on islemesi mel onbellegiyle ayni: librosa
soxr_hq ile 16 kHz mono, ortalama cikarma, tepe 0,99 (sentetikte segment basina, ResNet-18 yolu gibi).

Gomme: BEATs son katman ciktisinin zaman ortalamasi (768). Kafa hiperparametreleri ONCEDEN SABIT,
val/test ile ayarlanmadi: 768-512-26 MLP (ReLU, dropout 0,3), AdamW (lr 1e-3, wd 1e-4), 30 epoch,
cosine, label smoothing 0,1, batch 256. Ozellikler gercek egitim gommelerinin ortalama/SS'si ile
standartlastirilir (tum konfigurasyonlarda ayni donusum). Birincil degerlendirme ResNet-18 ile ayni:
son epoch modeli, val+test; ayrica en iyi-val epoch'un test tahmini (sureklilik).

Asamalar:
  python scripts/19_beats_probe.py gomme                       # gercek + sentetik gommeler
  python scripts/19_beats_probe.py gomme --maks_dosya 40       # hizli deneme (ayri klasore)
  python scripts/19_beats_probe.py kafa --konfig C4p --seed 42 --kaynak data/synthetic_v2/C4p:... --n_klip 50
Ciktilar:
  data/beats_gomme/<bolum>/blok_XXXX.npz (git disi), data/beats_gomme/sentetik.npz
  results/v2/siniflandirma/beats_<konfig>_s<seed>.json, results/v2/tahminler/beats_<konfig>_s<seed>.npz
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BEATS_DIR = ROOT / "checkpoints" / "beats"
BEATS_CKPT = BEATS_DIR / "BEATs_iter3_plus_AS2M.pt"
BEATS_SHA256 = "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"
GOMME_DIR = ROOT / "data" / "beats_gomme"
SONUC_DIR = ROOT / "results" / "v2" / "siniflandirma"
TAHMIN_DIR = ROOT / "results" / "v2" / "tahminler"
DURDUR_DOSYASI = ROOT / "outputs" / "v2_loglar" / "kuyruk" / "DURDUR"

SR = 16_000
SEG_ORNEK = 80_000            # 5 sn
SEG_ADIM = 313 * 256          # mel onbellegi hizalamasi (T_MAX * HOP_LENGTH)
BLOK = 500                    # blok basina kayit (kesinti/devam birimi)
SENTETIK_KLASORLER = ["data/synthetic_turcoh", "data/synthetic", "data/synthetic_v2/H1ek", "data/synthetic_v2/C4p"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _modul(ad, yol):
    s = importlib.util.spec_from_file_location(ad, yol)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


# ───────────────────────────── dalga bicimi ─────────────────────────────
def _hazirla(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    y = y - y.mean()
    tepe = np.abs(y).max()
    if tepe > 0:
        y = y * (0.99 / tepe)
    return y.astype(np.float32)


def kayit_segmentleri(yol: str, n_segs: int) -> np.ndarray:
    """Gercek kayit: tum dosya islenir (mel onbellegiyle ayni), n_segs dilim kesilir."""
    import soundfile as sf
    try:
        y, sr = sf.read(yol, dtype="float32", always_2d=False)
        y = _hazirla(y, sr)
    except Exception as e:
        print(f"  [UYARI] okunamadi ({e}): {yol}", flush=True)
        y = np.zeros(SEG_ORNEK, dtype=np.float32)
    out = np.zeros((n_segs, SEG_ORNEK), dtype=np.float32)
    for s in range(n_segs):
        parca = y[s * SEG_ADIM: s * SEG_ADIM + SEG_ORNEK]
        out[s, :len(parca)] = parca
    return out


def sentetik_segment(yol: str) -> np.ndarray:
    """Sentetik klip: ResNet-18 yolu gibi ilk 5 sn, segment basina on isleme."""
    import soundfile as sf
    y, sr = sf.read(yol, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y[: int(5.0 * sr)]
    y = _hazirla(y, sr)
    out = np.zeros(SEG_ORNEK, dtype=np.float32)
    out[: min(len(y), SEG_ORNEK)] = y[:SEG_ORNEK]
    return out


class _KayitVeri(torch.utils.data.Dataset):
    def __init__(self, satirlar):
        self.satirlar = satirlar

    def __len__(self):
        return len(self.satirlar)

    def __getitem__(self, i):
        yol, n = self.satirlar[i]
        return kayit_segmentleri(yol, n)


# ───────────────────────────── BEATs ─────────────────────────────
def beats_yukle():
    import hashlib
    h = hashlib.sha256()
    with open(BEATS_CKPT, "rb") as f:
        for parca in iter(lambda: f.read(1 << 24), b""):
            h.update(parca)
    if h.hexdigest() != BEATS_SHA256:
        raise SystemExit(f"HATA: BEATs agirlik ozeti uyusmuyor: {h.hexdigest()}")
    sys.path.insert(0, str(BEATS_DIR))
    from BEATs import BEATs, BEATsConfig  # resmi kod (unilm/beats)
    ck = torch.load(BEATS_CKPT, map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.predictor = None          # yalniz gomme
    return model.eval().to(DEVICE)


@torch.no_grad()
def _ileri(model, x: torch.Tensor, yari: bool) -> torch.Tensor:
    """BEATs.extract_features ile ayni adimlar; fbank fp32'de (x*2^15 gucu fp16'da tasar),
    transformer istege bagli fp16 autocast ile."""
    fbank = model.preprocess(x.float()).unsqueeze(1)
    with torch.autocast("cuda", dtype=torch.float16, enabled=(yari and DEVICE == "cuda")):
        f = model.patch_embedding(fbank)
        f = f.reshape(f.shape[0], f.shape[1], -1).transpose(1, 2)
        f = model.layer_norm(f)
        if model.post_extract_proj is not None:
            f = model.post_extract_proj(f)
        h, _ = model.encoder(model.dropout_input(f), padding_mask=None)
    return h.float().mean(dim=1)


@torch.no_grad()
def gomme_hesapla(model, segler: np.ndarray, batch: int = 48, yari: bool = True) -> np.ndarray:
    out = []
    for i in range(0, len(segler), batch):
        x = torch.from_numpy(np.ascontiguousarray(segler[i:i + batch])).to(DEVICE)
        h = _ileri(model, x, yari)
        if not torch.isfinite(h).all():
            raise RuntimeError("BEATs gommesinde NaN/Inf")
        out.append(h.cpu().numpy().astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, 768), np.float16)


def _atomik_npz(yol: Path, **diziler):
    gecici = yol.with_name(yol.name + ".tmp.npz")
    np.savez(gecici, **diziler)
    with open(gecici, "r+b") as f:      # Windows'ta fsync yazma erisimi ister
        os.fsync(f.fileno())
    os.replace(gecici, yol)


def gomme(a):
    IL = _modul("iterloop", ROOT / "scripts" / "07_iterative_loop.py")
    hedef = GOMME_DIR if a.maks_dosya is None else GOMME_DIR.parent / "beats_gomme_deneme"
    hedef.mkdir(parents=True, exist_ok=True)
    model = beats_yukle()
    df_tr, df_vl, df_te = IL.veri_bol(IL.PARQUET_LISTE, ROOT)
    t0 = time.time()
    for bolum, df in (("val", df_vl), ("test", df_te), ("train", df_tr)):
        if a.maks_dosya is not None:
            df = df.iloc[: a.maks_dosya]
        d = hedef / bolum
        d.mkdir(exist_ok=True)
        n_blok = (len(df) + BLOK - 1) // BLOK
        for b in range(n_blok):
            yol = d / f"blok_{b:04d}.npz"
            if yol.exists():
                continue
            if DURDUR_DOSYASI.exists():
                print("  [DURDUR] Istek alindi; tamamlanan bloklar kayitli (kod 4).", flush=True)
                sys.exit(4)
            parca = df.iloc[b * BLOK:(b + 1) * BLOK]
            satirlar = [(str(ROOT / r.dosya), max(1, int(r.n_segs))) for r in parca.itertuples()]
            yukleyici = torch.utils.data.DataLoader(_KayitVeri(satirlar), batch_size=None, shuffle=False,
                                                    num_workers=a.isci, prefetch_factor=4 if a.isci else None)
            gommeler, dosyalar, etiketler, segno = [], [], [], []
            for (wav, n), segler in zip(satirlar, yukleyici):
                gommeler.append(gomme_hesapla(model, segler.numpy() if torch.is_tensor(segler) else segler))
                dosyalar += [wav] * n
                segno += list(range(n))
            for r in parca.itertuples():
                etiketler += [int(r.sinif_idx)] * max(1, int(r.n_segs))
            _atomik_npz(yol, gomme=np.concatenate(gommeler), dosya=np.array(dosyalar),
                        etiket=np.array(etiketler, dtype=np.int16), seg=np.array(segno, dtype=np.int32))
            gecen = time.time() - t0
            print(f"  {bolum} blok {b + 1}/{n_blok}  ({len(dosyalar)} segment)  {gecen / 60:.1f} dk", flush=True)
    # Sentetik klipler (tum ilgili klasorler; secim kafa asamasinda 15_siniflandirma_v2 ile ayni)
    yol = hedef / "sentetik.npz"
    if not yol.exists():
        wavlar = sorted(str(w.relative_to(ROOT)) for k in SENTETIK_KLASORLER
                        for w in (ROOT / k).glob("*/*.wav") if w.stat().st_size > 1000)
        if a.maks_dosya is not None:
            wavlar = wavlar[: a.maks_dosya]
        segler = np.stack([sentetik_segment(str(ROOT / w)) for w in wavlar])
        _atomik_npz(yol, gomme=gomme_hesapla(model, segler), dosya=np.array(wavlar))
        print(f"  sentetik: {len(wavlar)} klip", flush=True)
    (hedef / "TAMAM").write_text(datetime.now().isoformat(), encoding="utf-8")
    print(f"Gommeler tamam: {hedef}  ({(time.time() - t0) / 60:.1f} dk)", flush=True)


# ───────────────────────────── MLP kafasi ─────────────────────────────
def _bolum_yukle(bolum):
    bloklar = sorted((GOMME_DIR / bolum).glob("blok_*.npz"))
    z = [np.load(b) for b in bloklar]
    return (np.concatenate([x["gomme"] for x in z]).astype(np.float32),
            np.concatenate([x["etiket"] for x in z]).astype(np.int64),
            np.concatenate([x["dosya"] for x in z]))


def kafa(a):
    from sklearn.metrics import f1_score
    if not (GOMME_DIR / "TAMAM").exists():
        raise SystemExit("HATA: once 'gomme' asamasi tamamlanmali")
    S15 = _modul("siniflandirma_v2", ROOT / "scripts" / "15_siniflandirma_v2.py")
    ad = f"beats_{a.konfig}_s{a.seed}"
    cikti = SONUC_DIR / f"{ad}.json"
    if cikti.exists() and not a.zorla:
        print(f"Atlaniyor: {cikti.name} zaten var")
        return
    X_tr, y_tr, _ = _bolum_yukle("train")
    X_vl, y_vl, d_vl = _bolum_yukle("val")
    X_te, y_te, d_te = _bolum_yukle("test")
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6           # gercek egitim istatistikleri (tum konfiglerde ayni)
    df_synt = S15.sentetik_sec(a.kaynak, a.n_klip)
    if len(df_synt):
        zs = np.load(GOMME_DIR / "sentetik.npz")
        satir = {d: i for i, d in enumerate(zs["dosya"])}
        eksik = [d for d in df_synt["dosya"] if d not in satir]
        if eksik:
            raise SystemExit(f"HATA: {len(eksik)} sentetik klibin gommesi yok, ornek: {eksik[0]}")
        X_tr = np.concatenate([X_tr, zs["gomme"][[satir[d] for d in df_synt["dosya"]]].astype(np.float32)])
        y_tr = np.concatenate([y_tr, df_synt["sinif_idx"].to_numpy(np.int64)])
        print("  Sentetik: " + "  ".join(f"{s}={n}" for s, n in df_synt["sinif"].value_counts().sort_index().items()))
    st = lambda X: torch.from_numpy((X - mu) / sd)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = torch.nn.Sequential(torch.nn.Linear(768, 512), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                                torch.nn.Linear(512, 26)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    EPOCH, BATCH = 30, 256
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCH)
    kf = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    Xtr, Ytr = st(X_tr).to(DEVICE), torch.from_numpy(y_tr).to(DEVICE)
    Xvl, Xte = st(X_vl).to(DEVICE), st(X_te).to(DEVICE)
    g = torch.Generator(device="cpu").manual_seed(a.seed)

    @torch.no_grad()
    def tahmin(X):
        model.eval()
        return torch.cat([model(X[i:i + 4096]).argmax(1) for i in range(0, len(X), 4096)]).cpu().numpy()

    en_iyi, en_iyi_durum, tarih = -1.0, None, []
    t0 = time.time()
    for ep in range(1, EPOCH + 1):
        model.train()
        sira = torch.randperm(len(Xtr), generator=g).to(DEVICE)
        toplam = 0.0
        for i in range(0, len(sira) - BATCH + 1, BATCH):
            ix = sira[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            kayip = kf(model(Xtr[ix]), Ytr[ix])
            kayip.backward()
            opt.step()
            toplam += kayip.item()
        sch.step()
        p_vl = tahmin(Xvl)
        f_vl = f1_score(y_vl, p_vl, average="macro", zero_division=0)
        tarih.append({"epoch": ep, "kayip": toplam / max(1, len(sira) // BATCH), "val_f1": float(f_vl)})
        if f_vl > en_iyi:
            en_iyi, en_iyi_durum = f_vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  Epoch {ep:3d}/{EPOCH}  loss={tarih[-1]['kayip']:.4f}  val_f1={f_vl:.4f}", flush=True)
    son_vl, son_te = tahmin(Xvl), tahmin(Xte)
    model.load_state_dict(en_iyi_durum)
    iyi_vl, iyi_te = tahmin(Xvl), tahmin(Xte)

    TAHMIN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TAHMIN_DIR / f"{ad}.npz",
                        test_tahmin=iyi_te.astype(np.int16), test_gercek=y_te.astype(np.int16), test_dosya=d_te,
                        val_tahmin=iyi_vl.astype(np.int16), val_gercek=y_vl.astype(np.int16), val_dosya=d_vl,
                        son_test_tahmin=son_te.astype(np.int16), son_val_tahmin=son_vl.astype(np.int16))
    f = lambda y, p: float(f1_score(y, p, average="macro", zero_division=0))
    kayit = {"konfig": f"beats_{a.konfig}", "seed": a.seed, "tarih": datetime.now().strftime("%Y%m%d_%H%M"),
             "git": S15.git_surumu(), "argumanlar": vars(a), "model": "BEATs_iter3+ (AS2M) dondurulmus + MLP",
             "beats_sha256": BEATS_SHA256,
             "sentetik_sayisi": df_synt["sinif"].value_counts().sort_index().to_dict(),
             "sentetik_dosyalar": df_synt["dosya"].tolist(),
             "val_f1": f(y_vl, iyi_vl), "test_f1": f(y_te, iyi_te),
             "son_epoch": {"epoch": EPOCH, "val_f1": f(y_vl, son_vl), "test_f1": f(y_te, son_te)},
             "tarih_epoch": tarih, "sure_dk": round((time.time() - t0) / 60, 2)}
    SONUC_DIR.mkdir(parents=True, exist_ok=True)
    with open(cikti, "w", encoding="utf-8") as fp:
        json.dump(kayit, fp, ensure_ascii=False, indent=1)
    print(f"\nKaydedildi: {cikti}  (son epoch val+test: val={kayit['son_epoch']['val_f1']:.4f}, "
          f"test={kayit['son_epoch']['test_f1']:.4f})", flush=True)


def main():
    p = argparse.ArgumentParser()
    alt = p.add_subparsers(dest="asama", required=True)
    g = alt.add_parser("gomme")
    g.add_argument("--maks_dosya", type=int, default=None, help="Deneme: bolum basina en fazla kayit")
    g.add_argument("--isci", type=int, default=4, help="Ses okuma/yeniden ornekleme surec sayisi")
    k = alt.add_parser("kafa")
    k.add_argument("--konfig", required=True)
    k.add_argument("--seed", type=int, default=42)
    k.add_argument("--kaynak", action="append", default=[])
    k.add_argument("--n_klip", type=int, default=None)
    k.add_argument("--zorla", action="store_true")
    a = p.parse_args()
    gomme(a) if a.asama == "gomme" else kafa(a)


if __name__ == "__main__":
    main()
