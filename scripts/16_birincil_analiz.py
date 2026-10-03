"""
Birincil istatistik analizi (revizyon K5; R1.2, R2.7, R3.4, R5).

1) BIRINCIL — kayit duzeyi, sinifa gore katmanli KUME BOOTSTRAP (F1 farki):
   Test kayitlari (dosyalar) kendi siniflari icinde yerine koyarak yeniden orneklenir;
   her tekrarda iki modelin sinif F1'leri ayni yeniden orneklenmis kayitlar uzerinde
   hesaplanir (eslestirilmis). v2'de fark seed'ler uzerinden ortalanir (her iki model
   ayni seed ile eslestirilir). Grup F1 farki = gruptaki siniflarin F1 farklarinin
   ortalamasi. Iki yonlu bootstrap p; grup duzeyi testlere Holm duzeltmesi.
2) IKINCIL — karma etkili lojistik model (R/lme4):
   dogru ~ model [+ seed] + (1|kayit) + (1|segment)   (yalniz grup sinifi segmentleri;
   her segment iki model altinda gozlenir -> eslestirilmis tasarim).

Girdi bicimleri:
  --v1 : results/v1/preds_test_seg.npz (anahtarlar: baseline, H1, C2, C3, C4, gt, dosya_id; tek seed)
  (v2) : results/v2/tahminler/<konfig>_s<seed>.npz (test_tahmin, test_gercek, test_dosya)

Kullanim:
  python scripts/16_birincil_analiz.py --v1                 # v1 tahminlerinin yeniden analizi
  python scripts/16_birincil_analiz.py --konfigler baseline H1 C2p C3p C4p --ciftler C4p:H1 C2p:H1
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SINIF_SIRASI = [
    "acil_servis", "at_sesleri", "ayak_sesleri", "cocuk_bebek_sesleri",
    "diger_hayvanlar", "dini_mekan", "doga_atmosferi", "endustri_fabrika",
    "ev_ici_sesler", "gemi_tekne_sesleri", "hava_olaylari", "insaat_sesleri",
    "kalabalik_sesleri", "kapi_sesleri", "kara_tasitlari", "kopek_sesleri",
    "kus_sesleri", "ofis_iletisim", "saat_can_sesleri", "sehir_atmosferi",
    "silah_patlama", "spor_eglence", "su_sesleri", "tarim_ciftlik",
    "tren_sesleri", "ucak_sesleri",
]
K = len(SINIF_SIRASI)
GRUPLAR = {
    "A_belirgin": ["kopek_sesleri", "kus_sesleri", "hava_olaylari"],
    "B_dusukF1":  ["insaat_sesleri", "dini_mekan", "acil_servis"],
    "kontrol":    ["diger_hayvanlar"],
}
RSCRIPT = Path(r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe")


# ── Veri yukleme ──────────────────────────────────────────────────────────────
def v1_yukle():
    z = np.load(ROOT / "results" / "v1" / "preds_test_seg.npz")
    gt, kayit = z["gt"].astype(int), z["dosya_id"].astype(int)
    tah = {k: {42: z[k].astype(int)} for k in ("baseline", "H1", "C2", "C3", "C4")}
    return gt, kayit, tah


def v2_yukle(konfigler, degerlendirme="test"):
    """degerlendirme: 'test'       -> en iyi-val checkpoint, test kumesi (v1 ile ayni tasarim)
                      'son_valtest' -> son epoch modeli, val+test birlikte (checkpoint secimi
                                       val'e dayanmadigi icin; degerlendirme kaydi 2 kat)"""
    tah, gt, dosya = {}, None, None
    for k in konfigler:
        tah[k] = {}
        for f in sorted((ROOT / "results" / "v2" / "tahminler").glob(f"{k}_s*.npz")):
            seed = int(f.stem.rsplit("_s", 1)[1])
            z = np.load(f)
            if degerlendirme == "son_valtest":
                g = np.concatenate([z["val_gercek"], z["test_gercek"]])
                d = np.concatenate([z["val_dosya"], z["test_dosya"]])
                t = np.concatenate([z["son_val_tahmin"], z["son_test_tahmin"]])
            else:
                g, d, t = z["test_gercek"], z["test_dosya"], z["test_tahmin"]
            if gt is None:
                gt, dosya = g.astype(int), d
            assert np.array_equal(gt, g) and np.array_equal(dosya, d), f"{f.name}: segment sirasi farkli"
            tah[k][seed] = t.astype(int)
        if not tah[k]:
            raise SystemExit(f"HATA: {k} icin tahmin dosyasi yok")
    _, kayit = np.unique(dosya, return_inverse=True)
    return gt, kayit, tah


# ── Kayit duzeyi sayimlar ve F1 ───────────────────────────────────────────────
def kayit_sayimlari(gt, pred, kayit, n_kayit):
    """[n_kayit, K] TP, FP, FN sayilari."""
    tp = np.zeros((n_kayit, K)); fp = np.zeros((n_kayit, K)); fn = np.zeros((n_kayit, K))
    dogru = gt == pred
    np.add.at(tp, (kayit[dogru], gt[dogru]), 1)
    np.add.at(fp, (kayit[~dogru], pred[~dogru]), 1)
    np.add.at(fn, (kayit[~dogru], gt[~dogru]), 1)
    return tp, fp, fn


def f1(tp, fp, fn):
    pay, payda = 2 * tp, 2 * tp + fp + fn
    return np.divide(pay, payda, out=np.zeros_like(pay, dtype=float), where=payda > 0)


def bootstrap_agirliklari(kayit_sinif, B, rng):
    """Sinif icinde katmanli kume bootstrap: [B, n_kayit] tekrar sayilari."""
    n = len(kayit_sinif)
    W = np.zeros((B, n), dtype=np.float32)
    for c in np.unique(kayit_sinif):
        idx = np.where(kayit_sinif == c)[0]
        W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=B)
    return W


def karsilastir(gt, kayit, kayit_sinif, tah_a, tah_b, W):
    """tah_b - tah_a farki (seed'lerde eslestirilmis). Doner: nokta [K], bootstrap [B, K]."""
    n_kayit = len(kayit_sinif)
    ortak = sorted(set(tah_a) & set(tah_b))
    if not ortak:
        raise SystemExit("HATA: ortak seed yok")
    nokta, boot = np.zeros(K), np.zeros((W.shape[0], K))
    for s in ortak:
        ta, fa, na = kayit_sayimlari(gt, tah_a[s], kayit, n_kayit)
        tb, fb, nb = kayit_sayimlari(gt, tah_b[s], kayit, n_kayit)
        nokta += f1(tb.sum(0), fb.sum(0), nb.sum(0)) - f1(ta.sum(0), fa.sum(0), na.sum(0))
        boot += f1(W @ tb, W @ fb, W @ nb) - f1(W @ ta, W @ fa, W @ na)
    return nokta / len(ortak), boot / len(ortak), ortak


def p_iki_yonlu(dagilim):
    return float(min(1.0, 2 * min((dagilim <= 0).mean(), (dagilim >= 0).mean())))


def holm(pler):
    sira = np.argsort(pler)
    m, duz, onceki = len(pler), np.zeros(len(pler)), 0.0
    for r, i in enumerate(sira):
        onceki = max(onceki, min(1.0, (m - r) * pler[i]))
        duz[i] = onceki
    return duz


# ── GLMM (R/lme4) ─────────────────────────────────────────────────────────────
def glmm(gt, kayit, tah_a, tah_b, siniflar, ad_a, ad_b):
    if not RSCRIPT.exists():
        return {"hata": "Rscript yok"}
    idx = np.where(np.isin(gt, [SINIF_SIRASI.index(s) for s in siniflar]))[0]
    ortak = sorted(set(tah_a) & set(tah_b))
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "d.csv"
        with open(csv, "w", encoding="utf-8") as f:
            f.write("dogru,model,seed,kayit,segment\n")
            for s in ortak:
                for m, t in ((0, tah_a[s]), (1, tah_b[s])):
                    for i in idx:
                        f.write(f"{int(t[i] == gt[i])},{m},{s},{kayit[i]},{i}\n")
        seed_terimi = "+ factor(seed) " if len(ortak) > 1 else ""
        r = f"""
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages(library(lme4))
d <- read.csv("{csv.as_posix()}")
# Rastgele egim (1 + model | kayit): konfigurasyon etkisinin kayittan kayda degismesine izin verir;
# olmazsa etki kayit icindeki segmentlerde sozde-tekrarlanir ve SE kuculur (segment McNemar sorunu).
m <- glmer(dogru ~ model {seed_terimi}+ (1 + model | kayit) + (1|segment), data=d, family=binomial,
           control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
s <- summary(m)$coefficients
vc <- VarCorr(m)
uyari <- paste(c(m@optinfo$conv$lme4$messages, if (isSingular(m)) "singular"), collapse="; ")
cat(sprintf('{{"log_or": %.6f, "se": %.6f, "z": %.4f, "p": %.6g, "n_gozlem": %d, "n_kayit": %d, "var_kayit": %.4f, "var_egim": %.4f, "var_segment": %.4f, "uyari": "%s"}}',
    s["model","Estimate"], s["model","Std. Error"], s["model","z value"], s["model","Pr(>|z|)"],
    nrow(d), length(unique(d$kayit)), vc$kayit[1,1], vc$kayit[2,2], vc$segment[1], gsub('"', "'", uyari)))
"""
        rf = Path(td) / "m.R"
        rf.write_text(r, encoding="utf-8")
        cikti = subprocess.run([str(RSCRIPT), str(rf)], capture_output=True, text=True)
        try:
            sonuc = json.loads(cikti.stdout.strip().splitlines()[-1])
            sonuc["odds_ratio"] = round(float(np.exp(sonuc["log_or"])), 4)
            sonuc["or_ga95"] = [round(float(np.exp(sonuc["log_or"] - 1.96 * sonuc["se"])), 4),
                                round(float(np.exp(sonuc["log_or"] + 1.96 * sonuc["se"])), 4)]
            sonuc["karsilastirma"] = f"{ad_b} vs {ad_a}"
            return sonuc
        except Exception:
            return {"hata": (cikti.stderr or cikti.stdout)[-800:]}


# ── Ana ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1", action="store_true", help="v1 tahminlerini analiz et")
    p.add_argument("--konfigler", nargs="+", default=None, help="v2: baseline + konfigler")
    p.add_argument("--ciftler", nargs="*", default=[], help="Konfiglerarasi: 'B:A' (B - A)")
    p.add_argument("--gruplar", nargs="+", default=None)
    p.add_argument("--B", type=int, default=10000)
    p.add_argument("--glmm", action="store_true", help="Ikincil GLMM analizini de calistir (R/lme4)")
    p.add_argument("--degerlendirme", choices=["test", "son_valtest"], default="test",
                   help="v2: 'test' (en iyi-val ckpt, test) | 'son_valtest' (son epoch, val+test)")
    p.add_argument("--cikti", default=None)
    a = p.parse_args()

    if a.v1:
        gt, kayit, tah = v1_yukle()
        gruplar = a.gruplar or ["A_belirgin", "kontrol"]
        ciftler = a.ciftler or ["C4:H1", "C2:H1", "C3:C2"]
        cikti = Path(a.cikti or ROOT / "results" / "v1" / "birincil_analiz_v1.json")
    else:
        gt, kayit, tah = v2_yukle(a.konfigler, a.degerlendirme)
        gruplar = a.gruplar or list(GRUPLAR)
        ciftler = a.ciftler
        cikti = Path(a.cikti or ROOT / "results" / "v2" / f"birincil_analiz_{a.degerlendirme}.json")

    n_kayit = kayit.max() + 1
    kayit_sinif = np.zeros(n_kayit, dtype=int)
    kayit_sinif[kayit] = gt
    rng = np.random.default_rng(2026)
    W = bootstrap_agirliklari(kayit_sinif, a.B, rng)
    print(f"Test: {len(gt):,} segment, {n_kayit:,} kayit; bootstrap B={a.B} (sinif icinde katmanli)")

    karsilastirmalar = [("baseline", k) for k in tah if k != "baseline"] + \
                       [tuple(c.split(":")[::-1]) for c in ciftler]
    sonuc = {"tarih": datetime.now().strftime("%Y%m%d_%H%M"), "kaynak": "v1" if a.v1 else "v2",
             "B": a.B, "n_segment": int(len(gt)), "n_kayit": int(n_kayit), "karsilastirmalar": []}
    aile = []
    for ref, test in karsilastirmalar:
        nokta, boot, seedler = karsilastir(gt, kayit, kayit_sinif, tah[ref], tah[test], W)
        k = {"test": test, "referans": ref, "seedler": seedler, "gruplar": {}}
        for g in gruplar:
            ix = [SINIF_SIRASI.index(s) for s in GRUPLAR[g]]
            gd = boot[:, ix].mean(1)
            k["gruplar"][g] = {
                "delta_f1": round(float(nokta[ix].mean()), 4),
                "ga95": [round(float(np.percentile(gd, 2.5)), 4), round(float(np.percentile(gd, 97.5)), 4)],
                "p_bootstrap": round(p_iki_yonlu(gd), 5),
                "sinif": {s: {"delta_f1": round(float(nokta[SINIF_SIRASI.index(s)]), 4),
                              "ga95": [round(float(np.percentile(boot[:, SINIF_SIRASI.index(s)], q)), 4) for q in (2.5, 97.5)]}
                          for s in GRUPLAR[g]},
            }
            aile.append((len(sonuc["karsilastirmalar"]), g, k["gruplar"][g]["p_bootstrap"]))
            if a.glmm and g != "kontrol":
                print(f"  GLMM: {test} vs {ref} [{g}] ...", flush=True)
                k["gruplar"][g]["glmm"] = glmm(gt, kayit, tah[ref], tah[test], GRUPLAR[g], ref, test)
        sonuc["karsilastirmalar"].append(k)

    duz = holm(np.array([x[2] for x in aile]))
    for (ki, g, _), ph in zip(aile, duz):
        sonuc["karsilastirmalar"][ki]["gruplar"][g]["p_holm"] = round(float(ph), 5)

    print(f"\n{'karsilastirma':22s} {'grup':12s} {'dF1':>8s}  {'%95 GA':>18s}  {'p':>8s}  {'p_Holm':>8s}  GLMM OR [GA] p")
    for k in sonuc["karsilastirmalar"]:
        for g, v in k["gruplar"].items():
            gl = v.get("glmm", {})
            gls = f"{gl['odds_ratio']} {gl['or_ga95']} {gl['p']:.3g}" if "odds_ratio" in gl else gl.get("hata", "")[:40]
            print(f"{k['test'] + ' vs ' + k['referans']:22s} {g:12s} {v['delta_f1']:+8.4f}  "
                  f"[{v['ga95'][0]:+.4f}, {v['ga95'][1]:+.4f}]  {v['p_bootstrap']:8.4f}  {v['p_holm']:8.4f}  {gls}")
    cikti.parent.mkdir(parents=True, exist_ok=True)
    cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nKaydedildi: {cikti}")


if __name__ == "__main__":
    main()
