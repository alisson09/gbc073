#!/usr/bin/env python3
"""Lê experimentos.csv + logs/ e gera painel.pdf, slide.pdf, resumo.md e resumo.json."""
import json, math, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

D = sys.argv[1] if len(sys.argv) > 1 else "."
P = lambda *x: os.path.join(D, *x)
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
COR = {"sgd": "#8c564b", "sgdn": "#7f7f7f", "adam": "#1f77b4", "adamw": "#2ca02c",
       "muon": "#d62728", "rmsprop": "#9467bd", "nadam": "#ff7f0e"}

df = pd.read_csv(P("experimentos.csv"))
for c in ("lr", "lr_adamw", "val_acc", "val_metrica", "val_loss", "frac", "seed"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["divergiu"] = df["divergiu"].astype(int).astype(bool)
df["score"] = np.where(df["divergiu"], np.nan, df["val_metrica"])
df["lr_adamw"] = df["lr_adamw"].where(df["opt"] == "muon", np.nan)
df = df.drop_duplicates("id", keep="last")
M = {"pr_auc": "PR-AUC", "acc": "acurácia"}.get(str(df["metrica"].iloc[0]), "métrica")
f_busca, f_full = df["frac"].min(), df["frac"].max()
cfg = ["opt", "lr", "lr_adamw", "sched"]
rot = lambda r: f"{r['opt']} lr={r['lr']:g}" + (f" (+AdamW {r['lr_adamw']:g})" if r["opt"] == "muon" else "") + f" {r['sched']}"


def log_de(rid):
    f = P("logs", rid + ".csv")
    return pd.read_csv(f) if os.path.exists(f) else None


# ---------- confirmação (dados completos, >=2 sementes)
conf = df[(df["frac"] == f_full) & ~df["divergiu"]].copy()
g = conf.groupby(cfg, dropna=False).agg(n=("score", "size"), media=("score", "mean"),
                                         dp=("score", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
                                         ids=("id", list), seeds=("seed", list)).reset_index()
g = g[g["n"] >= 2].sort_values("media", ascending=False)
base_cfg = json.load(open(P("config_baseline.json"))) if os.path.exists(P("config_baseline.json")) else None
eh_base = lambda r: base_cfg and r["opt"] == base_cfg["opt"] and math.isclose(r["lr"], base_cfg["lr"]) \
    and r["sched"] == base_cfg["sched"]
gb = g[g.apply(eh_base, axis=1)] if base_cfg and len(g) else g.iloc[0:0]
gn = g[~g.apply(eh_base, axis=1)] if len(g) else g
base = gb.iloc[0] if len(gb) else None
venc = gn.iloc[0] if len(gn) else None
muons = g[g["opt"] == "muon"]
melhor_muon = muons.iloc[0] if len(muons) else None

resumo = dict(f_busca=float(f_busca), f_full=float(f_full))
if venc is not None and base is not None:
    ganho = venc["media"] - base["media"]
    sig = ganho > max(venc["dp"] if not np.isnan(venc["dp"]) else 0, base["dp"] if not np.isnan(base["dp"]) else 0)
    resumo.update(vencedor=rot(venc), vencedor_ids=venc["ids"], baseline=rot(base), baseline_ids=base["ids"],
                  media_venc=venc["media"], dp_venc=venc["dp"], media_base=base["media"], dp_base=base["dp"],
                  ganho=ganho, ganho_significativo=bool(sig))
json.dump(resumo, open(P("resumo.json"), "w"), indent=1, default=float)

# ---------- instável: divergida; senão pior execução de lr alto na busca
busca = df[(df["frac"] == f_busca) & (df["sched"] == "warmcos") & (df["seed"] == 0)]
inst = df[df["divergiu"]]
inst_row = inst.iloc[0] if len(inst) else (busca.sort_values("score").iloc[0] if len(busca) else None)


def sinais(l):
    """Resumos numéricos lidos dos sinais (usados no diagnóstico)."""
    if l is None or len(l) < 5:
        return {}
    gr = l["grad"].values
    med = pd.Series(gr).rolling(9, center=True, min_periods=3).median().values
    pic = int(np.sum(gr[len(gr) // 10:] > 3 * med[len(gr) // 10:]))
    out = dict(picos_g=pic, theta_ini=l["theta"].iloc[0], theta_fim=l["theta"].dropna().iloc[-1],
               g_ini=gr[0], g_fim=gr[-1])
    for c in l.columns:
        if c.startswith("rho/"):
            out[c] = float(np.nanmedian(l[c].values[1:]))
    pm = l["perda_media"].values
    out["queda_final"] = float(pm[int(.8 * len(pm))] - pm[-1]) if len(pm) > 10 else np.nan
    return out


def primeira(ids, seed=0):
    return ids[0] if len(ids) else None


def row_id(df_, r):
    q = df_[(df_["opt"] == r["opt"]) & np.isclose(df_["lr"], r["lr"]) & (df_["sched"] == r["sched"])]
    if r["opt"] == "muon":
        q = q[np.isclose(q["lr_adamw"], r["lr_adamw"])]
    q = q.sort_values("seed")
    return q["id"].iloc[0] if len(q) else None


colunas = []   # (titulo, id)
if venc is not None: colunas.append(("MELHOR\n" + rot(venc), venc["ids"][int(np.argmin(venc["seeds"]))]))
if base is not None: colunas.append(("BASELINE\n" + rot(base), base["ids"][int(np.argmin(base["seeds"]))]))
if melhor_muon is not None and not (venc is not None and venc["opt"] == "muon"):
    colunas.append(("MELHOR COM MUON\n" + rot(melhor_muon), melhor_muon["ids"][int(np.argmin(melhor_muon["seeds"]))]))
if inst_row is not None:
    colunas.append(("INSTÁVEL" + (" (divergiu)" if inst_row["divergiu"] else "") + "\n" + rot(inst_row), inst_row["id"]))
if venc is not None and venc["opt"] == "muon":
    pass

sig_log = {}
with PdfPages(P("painel.pdf")) as pdf:
    # --- pág 1: quatro sinais alinhados
    if colunas:
        fig, ax = plt.subplots(4, len(colunas), figsize=(3.6 * len(colunas), 9), sharex="col", squeeze=False)
        for j, (tit, rid) in enumerate(colunas):
            l = log_de(rid)
            ax[0, j].set_title(tit, fontsize=7.5)
            if l is None:
                continue
            sig_log[tit.split("\n")[0]] = sinais(l)
            ax[0, j].semilogy(l["passo"], l["perda_media"], label="treino (média)")
            if "val_loss" in l:
                v = l.dropna(subset=["val_loss"]); ax[0, j].semilogy(v["passo"], v["val_loss"], "o-", ms=3, label="validação")
            ax[0, j].legend(fontsize=6)
            ax[1, j].semilogy(l["passo"], l["grad"], color="tab:orange")
            ax[2, j].plot(l["passo"], l["theta"], color="tab:purple")
            ax[3, j].plot(l["passo"], l["lr"], color="k")
            if "lr_adamw" in l:
                ax[3, j].plot(l["passo"], l["lr_adamw"], "k--", lw=.8, label="lr AdamW"); ax[3, j].legend(fontsize=6)
            ax[3, j].set_xlabel("passo")
        for i, n in enumerate(["perda", "‖∇L‖₂ (antes do clip)", "‖θ‖₂", "taxa efetiva η_t"]):
            ax[i, 0].set_ylabel(n)
        fig.suptitle("Os quatro sinais alinhados no mesmo eixo de passos", y=.995)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # --- pág 2: métrica × lr por otimizador
    if len(busca):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for o, q in busca.groupby("opt"):
            q = q.sort_values("lr")
            ax.plot(q["lr"], q["score"], "o-", color=COR.get(o), label=o)
            d = q[q["divergiu"]]
            ax.plot(d["lr"], [ax.get_ylim()[0]] * len(d), "x", color=COR.get(o), ms=9)
        ax.set_xscale("log"); ax.set_xlabel("η (matrizes, no Muon)"); ax.set_ylabel(f"{M} de validação")
        ax.set_title(f"Varredura de η — agenda warmup+cosseno, 1 semente, {f_busca:g} do treino (× = divergiu)")
        ax.legend(); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # --- pág 3: razão de atualização por camada, Muon × AdamW
    def melhor_busca(o):
        q = busca[busca["opt"] == o]
        return q.sort_values("score", ascending=False).iloc[0]["id"] if len(q.dropna(subset=["score"])) else None
    ra, rm = melhor_busca("adamw"), melhor_busca("muon")
    if ra and rm:
        la, lm = log_de(ra), log_de(rm)
        cams = [c for c in la.columns if c.startswith("rho/")]
        fig, ax = plt.subplots(1, len(cams), figsize=(3.6 * len(cams), 3.6), sharey=True)
        ax = np.atleast_1d(ax)
        for a_, c in zip(ax, cams):
            a_.semilogy(la["passo"], la[c], color=COR["adamw"], label="AdamW")
            if c in lm: a_.semilogy(lm["passo"], lm[c], color=COR["muon"], label="Muon")
            a_.axhline(1e-3, color="gray", ls=":", lw=1); a_.set_title(c[4:]); a_.set_xlabel("passo")
        ax[0].set_ylabel("‖θ₊₁−θ‖/‖θ‖"); ax[0].legend()
        fig.suptitle("Razão de atualização por camada (linha pontilhada: referência 1e-3)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # --- pág 4: tabela de confirmação
    if len(g):
        fig, ax = plt.subplots(figsize=(9, 1.2 + .4 * len(g))); ax.axis("off")
        cell = [[rot(r), int(r["n"]), f"{r['media']:.4f}", "—" if np.isnan(r["dp"]) else f"{r['dp']:.4f}"]
                for _, r in g.iterrows()]
        t = ax.table(cellText=cell, colLabels=["configuração", "n sementes", f"{M} val (média)", "desvio"], loc="center")
        t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.4)
        ax.set_title(f"Confirmação — dados completos (frac={f_full:g})"); pdf.savefig(fig); plt.close(fig)

# ---------- diagnóstico lido nos sinais
diag = []
sv = sig_log.get("MELHOR", {}); sb = sig_log.get("BASELINE", {})
if sv and sb:
    if sb["picos_g"] > sv["picos_g"]:
        diag.append(f"‖g‖ da baseline tem {sb['picos_g']} picos (>3× mediana) vs {sv['picos_g']} na melhor: passo/estatísticas instáveis.")
    elif sb["picos_g"] or sv["picos_g"]:
        diag.append(f"‖g‖: {sb['picos_g']} picos na baseline, {sv['picos_g']} na melhor.")
    rb = sb["theta_fim"] / sb["theta_ini"]; rv = sv["theta_fim"] / sv["theta_ini"]
    diag.append(f"‖θ‖ final/inicial: baseline {rb:.2f}×, melhor {rv:.2f}× (η, weight decay e agenda).")
    if not np.isnan(sv.get("queda_final", np.nan)):
        diag.append(f"Decaimento final da agenda derruba a perda em {sv['queda_final']:.3f} nos últimos 20% dos passos (melhor) vs {sb['queda_final']:.3f} (baseline).")
ra_s = sinais(log_de(ra)) if 'ra' in dir() and ra else {}
rm_s = sinais(log_de(rm)) if 'rm' in dir() and rm else {}
for c in [k for k in rm_s if k.startswith("rho/")]:
    if c in ra_s:
        diag.append(f"Razão de atualização mediana em {c[4:]}: Muon {rm_s[c]:.1e} vs AdamW {ra_s[c]:.1e}.")
resumo["diagnostico_auto"] = diag
json.dump(resumo, open(P("resumo.json"), "w"), indent=1, default=float)

# ---------- teste (se teste_final.py já rodou)
teste = pd.read_csv(P("testes.csv")) if os.path.exists(P("testes.csv")) else None

# ---------- slide
fig = plt.figure(figsize=(13.33, 7.5))
fig.text(.04, .93, "Desafio 3 — O melhor treino para o nosso problema", fontsize=22, weight="bold")
if venc is not None and base is not None:
    fig.text(.04, .84, "Receita vencedora: " + rot(venc), fontsize=15, color="#b00020", weight="bold")
    ganho_txt = f"{resumo['ganho']:+.4f} de {M} de validação (3 sementes)"
    fig.text(.04, .78, f"Ganho sobre a baseline: {ganho_txt}", fontsize=14)
    fig.text(.04, .735, f"{venc['media']:.4f} ± {venc['dp']:.4f}  vs  baseline {base['media']:.4f} ± {base['dp']:.4f}"
             + ("   → maior que a variação entre sementes" if resumo["ganho_significativo"]
                else "   → NÃO supera a variação entre sementes"), fontsize=11)
    if teste is not None and len(teste):
        tv = teste[teste["grupo"] == "vencedor"]["test_metrica"]; tb = teste[teste["grupo"] == "baseline"]["test_metrica"]
        fig.text(.04, .70, f"Teste (uma única avaliação): vencedor {tv.mean():.4f} ± {tv.std(ddof=1):.4f} | baseline {tb.mean():.4f} ± {tb.std(ddof=1):.4f}", fontsize=10)
    ax = fig.add_axes([.06, .12, .38, .5])
    gg = g.head(5)
    ax.bar(range(len(gg)), gg["media"], yerr=gg["dp"].fillna(0), color=[COR.get(o) for o in gg["opt"]], capsize=4)
    ax.set_xticks(range(len(gg))); ax.set_xticklabels([f"{r['opt']}\n{r['sched']}\nη={r['lr']:g}" for _, r in gg.iterrows()], fontsize=7)
    lo = min(gg["media"].min(), base["media"]) - .02; ax.set_ylim(max(0, lo), 1.0)
    ax.axhline(base["media"], color="k", ls="--", lw=1); ax.set_ylabel(f"{M} de validação (média ± dp)")
    fig.text(.50, .62, "Diagnóstico lido nos sinais (revise e ajuste ao seu problema)", fontsize=13, weight="bold")
    import textwrap
    y = .57
    for d in diag[:5]:
        w = textwrap.fill("• " + d, 66); fig.text(.50, y, w, fontsize=10.5, va="top"); y -= .033 * (w.count("\n") + 1) + .03
    fig.text(.04, .04, f"Protocolo: η ajustada por otimizador em {f_busca:g} do treino; confirmação com dados completos; seleção na validação.", fontsize=8, color="gray")
else:
    fig.text(.04, .5, "Rode o protocolo completo para preencher este slide.", fontsize=14)
fig.savefig(P("slide.pdf")); plt.close(fig)

# ---------- resumo.md
with open(P("resumo.md"), "w") as f:
    f.write("# Resumo\n\n")
    f.write(g.drop(columns=["ids", "seeds"]).to_markdown(index=False) if len(g) else "sem confirmação ainda")
    f.write("\n\n" + json.dumps({k: v for k, v in resumo.items() if k not in ('vencedor_ids', 'baseline_ids')}, indent=1, default=str))
print(open(P("resumo.md")).read())
