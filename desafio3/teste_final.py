#!/usr/bin/env python3
"""Avalia no TESTE (uma única vez) os checkpoints do vencedor e da baseline listados em resumo.json."""
import csv, json, os, sys
import torch
import treino
from dados import carregar

if os.path.exists("testes.csv") and "--forcar" not in sys.argv:
    sys.exit("testes.csv já existe: o teste só pode ser usado uma vez (use --forcar apenas se souber o que faz).")
R = json.load(open("resumo.json"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
linhas = []
for grupo, ids in (("vencedor", R["vencedor_ids"]), ("baseline", R["baseline_ids"])):
    for rid in ids:
        ck = torch.load(os.path.join("ckpt", rid + ".pt"), map_location=dev)
        _, _, _, _, Xte, yte, *_ = carregar(ck["dados"], 1.0)
        m = treino.MLP(ck["din"], ck["ncls"], ck["width"], ck["depth"]).to(dev)
        m.load_state_dict(ck["sd"])
        tl, ta, tm = treino.avaliar(m, Xte.to(dev), yte.to(dev), ck.get('metrica', 'acc'))
        linhas.append(dict(grupo=grupo, id=rid, test_loss=tl, test_acc=ta, test_metrica=tm))
        print(grupo, rid, f"teste={tm:.4f}")
with open("testes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["grupo", "id", "test_loss", "test_acc", "test_metrica"]); w.writeheader(); w.writerows(linhas)
