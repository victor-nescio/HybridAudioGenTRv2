"""
Sentetik ses kalite metrikleri (revizyon K8; R1.6, R2.6, R5.8). Konfigurasyonlar H1, C2', C3', C4';
her biri icin egitime giren 50 klip/sinif (15_siniflandirma_v2.sentetik_sec ile ayni secim).

 1) FAD (VGGish; fadtk 1.1.0 ile ayni gomme, kare duzeyi). Referans: sinifin siniflandirici-EGITIM
    bolumundeki tum benzersiz kayitlar, kayit basina en fazla 6 esit aralikli 5 sn segment (ilk gonderimdeki
    kopya kayit sorunu yok; F16). Sentetik: klibin tamami (kayitli klipler 5 sn). Kalibrasyon: ayni
    referansa karsi, egitimde olmayan (val+test) 50 gercek kaydin (her birinin ilk 5 sn'si) FAD'i; 5 rastgele
    cekim ortalamasi. Ayni orneklem buyuklugu (50 x 5 sn) FAD'in buyukluk yanliligini esitler.
 2) CLAP skoru: LAION-CLAP 630k-audioset-best (fadtk'nin CLAP checkpoint'i; ureticinin kosullandirma
    kodlayicisindan farkli). 5 sn'lik sesin (48 kHz; model 10 sn'ye 'repeatpad' ile tamamlar) gommesi ile
    sinifin 5 Ingilizce promptunun metin gommeleri arasindaki ortalama kosinus. Referans: gercek val+test
    kayitlarinin ilk 5 sn'si (sentetikle ayni uzunluk).
 3) Siniflandirici gecerliligi: v2 baseline ResNet-18'ler (5 seed, kayitli en iyi-val checkpoint)
    sentetik kliplerin egitime giren ilk 5 sn'sini yuzde kac hedef sinifa atiyor. Referans: ayni
    modellerin gercek test segmentlerindeki sinif duyarliligi (recall).

Kullanim: python scripts/21_kalite_metrikleri.py [--maks_kayit 5]   # deneme
Cikti   : results/v2/kalite_metrikleri.json   (gomme onbellegi: data/kalite_onbellek/, git disi)
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ONBELLEK = ROOT / "data" / "kalite_onbellek"
SR, SEG = 16000, 5 * 16000
SINIFLAR = ["kopek_sesleri", "kus_sesleri", "hava_olaylari", "insaat_sesleri", "dini_mekan", "tarim_ciftlik",
            "acil_servis", "diger_hayvanlar"]
TUM = ",".join(SINIFLAR)
KONFIG = {
    "H1": ["data/synthetic_turcoh:kopek_sesleri,kus_sesleri,hava_olaylari,diger_hayvanlar",
           "data/synthetic:insaat_sesleri,dini_mekan,acil_servis", "data/synthetic_v2/H1ek:tarim_ciftlik"],
    "C2p": [f"data/synthetic_v2/C2p:{TUM}"],
    "C3p": [f"data/synthetic_v2/C3p:{TUM}"],
    "C4p": [f"data/synthetic_v2/C4p:{TUM}"],
}


def _modul(ad, yol):
    s = importlib.util.spec_from_file_location(ad, yol)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


EZ = _modul("ezber", ROOT / "scripts" / "17_ezber_analizi.py")      # ses_oku (resampy, 16 kHz mono), vggish_yukle


def ilk_sn(yol, sn: float, sr_hedef: int) -> np.ndarray:
    """Dosyanin yalnizca ilk `sn` saniyesini okur; mono, sr_hedef'e yeniden ornekler (16 kHz: resampy,
    ezber analiziyle ayni; 48 kHz: soxr_hq)."""
    import soundfile as sf
    sr = sf.info(str(yol)).samplerate
    x, sr = sf.read(str(yol), stop=int(sn * sr), dtype="float32", always_2d=False)
    x = x.mean(1) if x.ndim > 1 else x
    if sr != sr_hedef:
        if sr_hedef == SR:
            import resampy
            x = resampy.resample(x, sr, SR)
        else:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=sr_hedef, res_type="soxr_hq")
    return x.astype(np.float32)


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    from fadtk.fad import calc_frechet_distance
    return float(calc_frechet_distance(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False)))


def vggish_kareler(model, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.get_embedding(x.astype(np.float32)), dtype=np.float32)


def referans_kareleri(model, df_tr, sinif, maks_kayit, maks_segment=6):
    yol = ONBELLEK / f"vggish_egitim_{sinif}{'_deneme' if maks_kayit else ''}.npy"
    if yol.exists():
        return np.load(yol)
    kayitlar = sorted(df_tr[df_tr.sinif == sinif]["dosya"].unique())[:maks_kayit]
    K = []
    for d in kayitlar:
        x = EZ.ses_oku(ROOT / d)
        n = max(1, len(x) // SEG)
        for j in np.unique(np.linspace(0, n - 1, min(n, maks_segment)).astype(int)):
            seg = x[j * SEG:(j + 1) * SEG]
            if len(seg) >= SR:
                K.append(vggish_kareler(model, seg))
    K = np.concatenate(K)
    ONBELLEK.mkdir(parents=True, exist_ok=True)
    np.save(yol, K)
    return K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maks_kayit", type=int, default=None, help="Deneme: sinif basina gercek kayit siniri")
    a = p.parse_args()
    import torch
    IL = _modul("iterloop", ROOT / "scripts" / "07_iterative_loop.py")
    S15 = _modul("s15", ROOT / "scripts" / "15_siniflandirma_v2.py")
    df_tr, df_vl, df_te = IL.veri_bol(IL.PARQUET_LISTE, ROOT)
    import pandas as pd
    df_dis = pd.concat([df_vl, df_te], ignore_index=True)          # referansta olmayan gercek kayitlar
    secim = {k: S15.sentetik_sec(v, 50) for k, v in KONFIG.items()}
    if a.maks_kayit:
        secim = {k: v.groupby("sinif").head(5) for k, v in secim.items()}
    sonuc = {"tarih": datetime.now().strftime("%Y%m%d_%H%M"), "tanim": __doc__.strip().splitlines()[0],
             "fad_vggish": {}, "clap_skoru": {}, "siniflandirici_gecerliligi": {}}

    # ── 1) FAD (VGGish) ───────────────────────────────────────────────────────
    vg = EZ.vggish_yukle()
    rng = np.random.default_rng(2026)
    for sinif in SINIFLAR:
        R = referans_kareleri(vg, df_tr, sinif, a.maks_kayit)
        dis = sorted(df_dis[df_dis.sinif == sinif]["dosya"].unique())[:a.maks_kayit]
        dis_k = [vggish_kareler(vg, ilk_sn(ROOT / d, 5.0, SR)) for d in dis]
        n_cek = min(50, len(dis_k))
        kal = [frechet(R, np.concatenate([dis_k[i] for i in rng.choice(len(dis_k), n_cek, replace=False)]))
               for _ in range(5)]
        sonuc["fad_vggish"][sinif] = {"n_referans_kare": int(len(R)),
                                      "gercek_kalibrasyon": {"n_kayit": n_cek, "ort": round(float(np.mean(kal)), 3),
                                                             "ss": round(float(np.std(kal, ddof=1)), 3)}}
        for k, df in secim.items():
            S = np.concatenate([vggish_kareler(vg, EZ.ses_oku(ROOT / d)) for d in df[df.sinif == sinif]["dosya"]])
            sonuc["fad_vggish"][sinif][k] = round(frechet(R, S), 3)
        print("FAD", sinif, sonuc["fad_vggish"][sinif], flush=True)
    del vg

    # ── 2) CLAP skoru (LAION-CLAP 630k-audioset-best) ─────────────────────────
    from fadtk.model_loader import CLAPLaionModel
    cm = CLAPLaionModel("audio")
    cm.load_model()
    promptlar = json.loads((ROOT / "data" / "processed" / "turkce_promptlar.json").read_text(encoding="utf-8"))

    def ses48(yol, sn=5.0):
        """Ilk 5 sn, 48 kHz; tam 240000 ornek (kisa kayitta sifir dolgu). 10 sn'ye tamamlama modelin kendi
        'repeatpad' yoluyla yapilir (gercek ve sentetik icin ayni)."""
        x = np.clip(ilk_sn(yol, sn, 48000), -1, 1)
        x = (x * 32767).astype(np.int16).astype(np.float32) / 32767       # laion_clap ornek kodundaki nicemleme
        out = np.zeros(int(sn * 48000), np.float32)
        out[: min(len(x), len(out))] = x[: len(out)]
        return out

    @torch.no_grad()
    def ses_gomme(yollar):
        E = []
        for i in range(0, len(yollar), 16):
            x = np.stack([ses48(y) for y in yollar[i:i + 16]])
            E.append(cm.model.get_audio_embedding_from_data(x=x, use_tensor=False))
        E = np.concatenate(E)
        return E / np.linalg.norm(E, axis=1, keepdims=True)

    for sinif in SINIFLAR:
        T = np.asarray(cm.model.get_text_embedding(promptlar[sinif]["ingilizce"]))
        T = T / np.linalg.norm(T, axis=1, keepdims=True)
        dis = sorted(df_dis[df_dis.sinif == sinif]["dosya"].unique())[:a.maks_kayit]
        g = (ses_gomme([ROOT / d for d in dis]) @ T.T).mean(1)
        sonuc["clap_skoru"][sinif] = {"gercek_valtest": {"n": int(len(g)), "ort": round(float(g.mean()), 4)}}
        for k, df in secim.items():
            s = (ses_gomme([ROOT / d for d in df[df.sinif == sinif]["dosya"]]) @ T.T).mean(1)
            sonuc["clap_skoru"][sinif][k] = round(float(s.mean()), 4)
        print("CLAP", sinif, sonuc["clap_skoru"][sinif], flush=True)
    del cm

    # ── 3) Siniflandirici gecerliligi (v2 baseline ResNet-18, 5 seed) ─────────
    from src.data.dataset import dataloader_olustur
    from src.models.cnn import SpectrogramCNN
    BA = _modul("ba", ROOT / "scripts" / "16_birincil_analiz.py")
    seedler = [42, 43, 44, 45, 46]
    recall = {s: [] for s in SINIFLAR}
    atama = {k: {s: [] for s in SINIFLAR} for k in secim}
    for sd in seedler:
        model = SpectrogramCNN(n_sinif=26, pretrained=False).to(IL.DEVICE)
        model.load_state_dict(torch.load(ROOT / "models" / "v2" / f"cnn_baseline_s{sd}.pt", map_location=IL.DEVICE,
                                         weights_only=False)["model_state"])
        model.eval()
        z = np.load(ROOT / "results" / "v2" / "tahminler" / f"baseline_s{sd}.npz")
        for s in SINIFLAR:
            c = BA.SINIF_SIRASI.index(s)
            m = z["test_gercek"] == c
            recall[s].append(float((z["test_tahmin"][m] == c).mean()))
        for k, df in secim.items():
            dl = dataloader_olustur(df, ROOT, mod="cnn", batch_size=64, shuffle=False, preload=False)
            tah = []
            with torch.no_grad():
                for x, _ in dl:
                    tah.append(model(x.to(IL.DEVICE)).argmax(1).cpu().numpy())
            tah = np.concatenate(tah)
            for s in SINIFLAR:
                ix = (df["sinif"] == s).to_numpy()
                atama[k][s].append(float((tah[ix] == BA.SINIF_SIRASI.index(s)).mean()))
    for s in SINIFLAR:
        sonuc["siniflandirici_gecerliligi"][s] = {"gercek_test_recall": round(float(np.mean(recall[s])), 4),
                                                  **{k: round(float(np.mean(atama[k][s])), 4) for k in secim}}
        print("GECERLILIK", s, sonuc["siniflandirici_gecerliligi"][s], flush=True)

    cikti = ROOT / "results" / "v2" / f"kalite_metrikleri{'_deneme' if a.maks_kayit else ''}.json"
    cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Kaydedildi:", cikti)


if __name__ == "__main__":
    main()
