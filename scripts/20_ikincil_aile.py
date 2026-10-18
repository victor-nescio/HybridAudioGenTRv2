"""
On kayitli ikincil analiz ailesi (docs/v2_onkayit.md) ve "ayrica raporlanacaklar".

16_birincil_analiz.py'nin bootstrap fonksiyonlari aynen kullanilir (sinif icinde katmanli kayit
kume bootstrap'i, B = 10.000, uretec seed 2026, eslestirilmis seed'ler). Fark: Holm duzeltmesi
yalnizca on kayittaki 11 testlik aileye uygulanir (16 numarali script hesapladigi tum
karsilastirma x grup ciftlerine uyguluyordu).

  Aile (Holm): 1-2 C4'-baseline (Grup B, kontrol); 3 H1-baseline; 4 C4'-H1; 5 C2'-baseline;
               6 C3'-baseline; 7 C3'-C2'; 8 C4'kaydir-baseline; 9 C4'-C4'kaydir;
               10 SpecAug-baseline; 11 SpecAug+C4'-SpecAug   (3-11: Grup A)
  Aile disi : P1 (hatirlatma), doz-cevap (25/50/100/200; 50 = C4', seed 42-44), acil servis,
              sureklilik (en iyi-val checkpoint, yalniz test), BEATs (on kayitta yok; R3, R5.3),
              ezber analizi ozeti.

Kullanim: python scripts/20_ikincil_aile.py
Cikti   : results/v2/ikincil_analiz_tam.json
"""
import json
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("ba", ROOT / "scripts" / "16_birincil_analiz.py")
BA = importlib.util.module_from_spec(_s)
_s.loader.exec_module(BA)

AILE = [  # (no, test, referans, grup, aciklama)
    (1, "C4p", "baseline", "B_dusukF1", "C4' vs baseline - Grup B"),
    (2, "C4p", "baseline", "kontrol", "C4' vs baseline - kontrol"),
    (3, "H1", "baseline", "A_belirgin", "H1 vs baseline"),
    (4, "C4p", "H1", "A_belirgin", "C4' vs H1 (ince-ayarin katkisi)"),
    (5, "C2p", "baseline", "A_belirgin", "C2' vs baseline"),
    (6, "C3p", "baseline", "A_belirgin", "C3' vs baseline"),
    (7, "C3p", "C2p", "A_belirgin", "C3' vs C2' (Turkce vs Ingilizce)"),
    (8, "C4p_kaydir", "baseline", "A_belirgin", "C4'-kaydir vs baseline (icerik kontrolu)"),
    (9, "C4p", "C4p_kaydir", "A_belirgin", "C4' vs C4'-kaydir (eslesen icerigin katkisi)"),
    (10, "specaug", "baseline", "A_belirgin", "SpecAugment vs baseline"),
    (11, "specaugC4p", "specaug", "A_belirgin", "SpecAugment+C4' vs SpecAugment"),
]
CNN_KONFIG = ["baseline", "C4p", "H1", "C2p", "C3p", "C4p_kaydir", "specaug", "specaugC4p",
              "doz25", "doz100", "doz200"]


def ozet(nokta, boot, grup, seedler):
    ix = [BA.SINIF_SIRASI.index(s) for s in BA.GRUPLAR[grup]]
    gd = boot[:, ix].mean(1)
    return {"delta_f1": round(float(nokta[ix].mean()), 6),
            "ga95": [round(float(np.percentile(gd, 2.5)), 6), round(float(np.percentile(gd, 97.5)), 6)],
            "p_bootstrap": round(BA.p_iki_yonlu(gd), 5), "seedler": seedler,
            "sinif": {s: round(float(nokta[BA.SINIF_SIRASI.index(s)]), 6) for s in BA.GRUPLAR[grup]}}


def hazirla(konfigler, degerlendirme):
    gt, kayit, tah = BA.v2_yukle(konfigler, degerlendirme)
    n_kayit = kayit.max() + 1
    kayit_sinif = np.zeros(n_kayit, dtype=int)
    kayit_sinif[kayit] = gt
    W = BA.bootstrap_agirliklari(kayit_sinif, 10000, np.random.default_rng(2026))
    return gt, kayit, kayit_sinif, tah, W


def kars(ctx, test, ref, grup, seedler=None):
    gt, kayit, kayit_sinif, tah, W = ctx
    ta, tb = tah[ref], tah[test]
    if seedler is not None:
        ta = {s: v for s, v in ta.items() if s in seedler}
        tb = {s: v for s, v in tb.items() if s in seedler}
    nokta, boot, ortak = BA.karsilastir(gt, kayit, kayit_sinif, ta, tb, W)
    return ozet(nokta, boot, grup, ortak)


def yazdir(baslik, satirlar):
    print(f"\n{baslik}")
    for ad, v in satirlar:
        ph = f"  p_Holm={v['p_holm']:.4f}" if "p_holm" in v else ""
        print(f"  {ad:48s} {v['delta_f1']:+.4f} [{v['ga95'][0]:+.4f}, {v['ga95'][1]:+.4f}]  "
              f"p={v['p_bootstrap']:.4f}{ph}  seed={v['seedler']}")


def main():
    sonuc = {"tarih": datetime.now().strftime("%Y%m%d_%H%M"), "B": 10000, "uretec_seed": 2026,
             "degerlendirme": "son epoch, val+test (on kayit)"}

    # ── On kayitli aile (son epoch, val+test) ─────────────────────────────────
    ctx = hazirla(CNN_KONFIG, "son_valtest")
    print(f"val+test: {len(ctx[0]):,} segment, {len(ctx[2]):,} kayit")
    aile = []
    for no, test, ref, grup, ack in AILE:
        v = kars(ctx, test, ref, grup)
        aile.append({"no": no, "test": test, "referans": ref, "grup": grup, "aciklama": ack, **v})
    duz = BA.holm(np.array([x["p_bootstrap"] for x in aile]))
    for x, ph in zip(aile, duz):
        x["p_holm"] = round(float(ph), 5)
    sonuc["aile"] = aile
    yazdir("ON KAYITLI IKINCIL AILE (Holm, m = 11)", [(f"{x['no']:2d}. {x['aciklama']}", x) for x in aile])

    # ── Aile disi ─────────────────────────────────────────────────────────────
    p1 = kars(ctx, "C4p", "baseline", "A_belirgin")
    acil = {k: kars(ctx, k, "baseline", "B_faz1_acil") for k in ("C4p", "H1", "C2p", "C3p")}
    sonuc["P1_hatirlatma"] = p1
    sonuc["acil_servis"] = acil
    yazdir("P1 (hatirlatma) ve acil servis", [("P1 C4' vs baseline - Grup A", p1)] +
           [(f"{k} vs baseline - acil servis", v) for k, v in acil.items()])

    doz = {}
    for d, k in ((25, "doz25"), (50, "C4p"), (100, "doz100"), (200, "doz200")):
        doz[d] = {g: kars(ctx, k, "baseline", g, seedler={42, 43, 44}) for g in ("A_belirgin", "kontrol", "B_dusukF1")}
    sonuc["doz_cevap"] = {"not": "seed 42-44; doz 50 = C4' (ayni seed'ler); Grup B tum dozlarda 50", "dozlar": doz}
    yazdir("DOZ-CEVAP (seed 42-44; Grup A ve kontrol doz alir, Grup B 50'de sabit)",
           [(f"doz {d:3d} - {g}", v) for d, gg in doz.items() for g, v in gg.items()])

    # Sureklilik: en iyi-val checkpoint, yalniz test
    ctx_t = hazirla(CNN_KONFIG[:8], "test")
    sur = [{"test": t, "referans": r, "grup": g, **kars(ctx_t, t, r, g)}
           for t, r, g in [("C4p", "baseline", "A_belirgin")] + [(t, r, g) for _, t, r, g, _ in AILE]]
    sonuc["sureklilik_enIyiVal_test"] = sur
    yazdir("SUREKLILIK (en iyi-val checkpoint, yalniz test; bilgi amacli)",
           [(f"{x['test']} vs {x['referans']} - {x['grup']}", x) for x in sur])
    del ctx, ctx_t

    # ── BEATs (on kayitta yok; R3, R5.3) ──────────────────────────────────────
    bk = ["beats_baseline", "beats_H1", "beats_C4p"]
    ctx_b = hazirla(bk + ["baseline"], "son_valtest")      # assert: segmentler ResNet-18 ile ayni
    beats = {f"{t} vs {r} - {g}": kars(ctx_b, t, r, g)
             for t, r in (("beats_C4p", "beats_baseline"), ("beats_H1", "beats_baseline"), ("beats_C4p", "beats_H1"))
             for g in ("A_belirgin", "B_dusukF1", "kontrol")}
    sonuc["beats"] = {"not": "On kayitta yok; hakem istegiyle saglamlik analizi. Seed 42-44.", "karsilastirmalar": beats}
    yazdir("BEATs (dondurulmus gomme + MLP; son epoch, val+test)", list(beats.items()))

    # ── Ezber analizi ozeti ───────────────────────────────────────────────────
    ezber = {}
    for et in ("H1", "C2p", "C3p", "C4p"):
        f = ROOT / "results" / "v2" / f"ezber_analizi_{et}.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text(encoding="utf-8"))["siniflar"]
        n = sum(v["n_sentetik"] for v in s.values())
        ezber[et] = {"n_sentetik": n,
                     "esigi_asan_toplam_oran": round(sum(v["esigi_asan_sentetik_orani"] * v["n_sentetik"] for v in s.values()) / n, 4),
                     "sinif": {k: {"sent_gercek_medyan": v["sentetik_gercek_nn_medyan"],
                                   "gercek_gercek_medyan": v["gercek_gercek_nn_medyan"],
                                   "esigi_asan": v["esigi_asan_sentetik_orani"]} for k, v in s.items()}}
    sonuc["ezber"] = {"not": "Esik: gercek->gercek (ayni kayit disi) NN benzerliginin 99. yuzdeligi; "
                             "gercek segmentlerin tanim geregi %1'i asar.", "konfig": ezber}
    print("\nEZBER (yakin-kopya esigini asan sentetik orani; gercek icin beklenen ~%1)")
    for et, v in ezber.items():
        ust = max(v["sinif"].items(), key=lambda kv: kv[1]["esigi_asan"])
        print(f"  {et:5s} n={v['n_sentetik']:4d}  toplam {100 * v['esigi_asan_toplam_oran']:.1f}%  "
              f"(en yuksek: {ust[0]} {100 * ust[1]['esigi_asan']:.1f}%)")

    cikti = ROOT / "results" / "v2" / "ikincil_analiz_tam.json"
    cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nKaydedildi: {cikti}")


if __name__ == "__main__":
    main()
