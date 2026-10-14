#!/usr/bin/env python3
"""Desafio 3 (GBC073) — treino configurável com instrumentação completa.

Exemplo:
  python treino.py --opt muon --lr 0.02 --lr_adamw 3e-3 --sched warmcos --seed 0
Cada execução: grava logs/<id>.csv (sinais) e acrescenta UMA linha em experimentos.csv.
"""
import argparse, csv, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OPTS = ["sgd", "sgdn", "adam", "adamw", "muon", "rmsprop", "nadam"]
SCHEDS = ["const", "cos", "warmcos", "wsd", "onecycle"]
CAMPOS = ["id", "etapa", "opt", "lr", "lr_adamw", "sched", "seed", "steps", "epocas", "bs", "frac",
          "dados", "metrica", "val_metrica", "val_acc", "val_loss", "divergiu", "tempo_s", "comando"]


# ----------------------------------------------------------------- Muon ----
def ortogonaliza(G, passos=5):
    """Newton–Schulz (coeficientes do enunciado): ~ U V^T de G = U S V^T."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    T = X.size(0) > X.size(1)
    if T:
        X = X.T
    for _ in range(passos):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if T else X


class Muon(torch.optim.Optimizer):
    """Muon próprio: momento (Nesterov) -> Newton–Schulz -> passo; weight decay desacoplado.
    Aceita tensores >2-D (conv) achatando para (C_out, resto).
    rms_match=True usa o ajuste de RMS do Moonshot (0.2*sqrt(max(m,n))), que deixa
    o Muon comparável ao AdamW; False usa max(1, m/n)**0.5 (Jordan)."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.01, ns_steps=5, rms_match=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps,
                                      rms_match=rms_match))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(p.grad)
                upd = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                M = upd.reshape(upd.shape[0], -1).float()
                O = ortogonaliza(M, g["ns_steps"])
                r, c = M.shape
                esc = 0.2 * math.sqrt(max(r, c)) if g["rms_match"] else max(1.0, r / c) ** 0.5
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(O.reshape(p.shape).to(p.dtype), alpha=-g["lr"] * esc)


# --------------------------------------------------------------- modelo ----
class MLP(nn.Module):
    """Troque por o modelo do seu projeto. Nomes: 'layers.*' (ocultas -> Muon) e 'head' (saída -> AdamW)."""

    def __init__(self, din, ncls, width=256, depth=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(din if i == 0 else width, width) for i in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.head = nn.Linear(width, ncls)

    def forward(self, x):
        for l, n in zip(self.layers, self.norms):
            x = F.gelu(n(l(x)))
        return self.head(x)


def camadas_log(model, depth):
    """Primeira, uma do meio e a última (mínimo exigido)."""
    nomes = ["layers.0.weight", f"layers.{depth // 2}.weight", "head.weight"]
    return list(dict.fromkeys(nomes))


# ------------------------------------------------------ otimizador/agenda ----
def construir_otimizadores(model, a):
    P = dict(model.named_parameters())
    if a.opt == "muon":
        muon_p = [p for n, p in P.items() if p.ndim >= 2 and "head" not in n and "emb" not in n]
        ids = {id(p) for p in muon_p}
        adamw_p = [p for p in model.parameters() if id(p) not in ids]
        return [Muon(muon_p, lr=a.lr, momentum=0.95, weight_decay=a.wd, rms_match=a.rms_match),
                torch.optim.AdamW(adamw_p, lr=a.lr_adamw, weight_decay=a.wd)]
    ps = model.parameters()
    if a.opt == "sgd":
        return [torch.optim.SGD(ps, lr=a.lr, momentum=0.9, nesterov=False, weight_decay=5e-4)]
    if a.opt == "sgdn":
        return [torch.optim.SGD(ps, lr=a.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)]
    if a.opt == "adam":
        return [torch.optim.Adam(ps, lr=a.lr)]
    if a.opt == "adamw":
        return [torch.optim.AdamW(ps, lr=a.lr, weight_decay=a.wd)]
    if a.opt == "rmsprop":
        return [torch.optim.RMSprop(ps, lr=a.lr, momentum=0.9)]
    if a.opt == "nadam":
        return [torch.optim.NAdam(ps, lr=a.lr)]
    raise ValueError(a.opt)


def fator(kind, t, T, warm=0.05):
    """Multiplicador do lr base no passo t (0..T-1)."""
    w = max(1, int(warm * T))
    if kind == "const":
        return 1.0
    if kind == "cos":
        return 0.5 * (1 + math.cos(math.pi * t / T))
    if kind == "warmcos":
        return (t + 1) / w if t < w else 0.5 * (1 + math.cos(math.pi * (t - w) / max(1, T - w)))
    if kind == "wsd":                       # warmup–estável–decaimento linear nos últimos 20%
        d0 = int(0.8 * T)
        if t < w:
            return (t + 1) / w
        return 1.0 if t < d0 else max(0.0, (T - t) / max(1, T - d0))
    if kind == "onecycle":                  # sobe linear em 30%, desce em cosseno
        u = int(0.3 * T)
        if t < u:
            return 0.04 + 0.96 * t / max(1, u)
        return 0.5 * (1 + math.cos(math.pi * (t - u) / max(1, T - u)))
    raise ValueError(kind)


# ------------------------------------------------------------ utilidades ----
def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--opt", default="adamw", choices=OPTS)
    p.add_argument("--lr", type=float, default=1e-3, help="lr (no Muon: das matrizes)")
    p.add_argument("--lr_adamw", type=float, default=3e-3, help="só Muon: lr da parte AdamW")
    p.add_argument("--sched", default="const", choices=SCHEDS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--frac", type=float, default=1.0, help="fração do treino (busca barata)")
    p.add_argument("--dados", default="digits")
    p.add_argument("--k", type=int, default=25, help="registrar a cada k passos (10..50)")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=0.0, help=">0 liga clipping (norma global)")
    p.add_argument("--rms_match", action="store_true", help="Muon: ajuste de RMS (Moonshot)")
    p.add_argument("--metrica", default="auto", choices=["auto", "acc", "pr_auc"],
                   help="métrica de seleção na validação (auto: pr_auc se 2 classes, senão acc)")
    p.add_argument("--peso_classe", default="sqrt", choices=["none", "sqrt", "inv"],
                   help="peso da perda por classe (desbalanceamento): 1/sqrt(freq), 1/freq ou nenhum")
    p.add_argument("--etapa", default="manual")
    p.add_argument("--saida", default=".")
    p.add_argument("--ckpt", action="store_true", help="salva pesos finais (p/ teste final)")
    a = p.parse_args(argv)
    assert 10 <= a.k <= 50, "k deve estar em [10, 50]"
    a._argv = list(sys.argv[1:] if argv is None else argv)
    return a


def run_id(a):
    s = f"{a.opt}_lr{a.lr:g}_{a.sched}_s{a.seed}_f{a.frac:g}_n{a.steps}_b{a.bs}_{os.path.basename(a.dados)}"
    s += f"_w{a.peso_classe}_m{a.metrica}"
    if a.opt == "muon":
        s += f"_la{a.lr_adamw:g}" + ("_rms" if a.rms_match else "")
    return s


def comando(a):
    partes = ["python treino.py", f"--opt {a.opt}", f"--lr {a.lr:g}", f"--sched {a.sched}",
              f"--seed {a.seed}", f"--steps {a.steps}", f"--bs {a.bs}", f"--frac {a.frac:g}",
              f"--dados {a.dados}", f"--k {a.k}", f"--width {a.width}", f"--depth {a.depth}",
              f"--wd {a.wd:g}", f"--clip {a.clip:g}", f"--metrica {a.metrica}",
              f"--peso_classe {a.peso_classe}"]
    if a.opt == "muon":
        partes.append(f"--lr_adamw {a.lr_adamw:g}")
        if a.rms_match:
            partes.append("--rms_match")
    return " ".join(partes)


def ap_score(y, p):
    """Average precision (área sob a curva precisão×revocação) da classe positiva."""
    o = np.argsort(-p, kind="stable"); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    n = y.sum()
    return float((prec * y).sum() / n) if n > 0 else float("nan")


@torch.no_grad()
def avaliar(model, X, y, metrica="acc"):
    """Devolve (perda CE sem pesos, acurácia, métrica de seleção)."""
    model.eval()
    L = A = 0.0; ps = []
    for i in range(0, len(X), 2048):
        out = model(X[i:i + 2048])
        L += F.cross_entropy(out, y[i:i + 2048], reduction="sum").item()
        A += (out.argmax(1) == y[i:i + 2048]).sum().item()
        if metrica == "pr_auc":
            ps.append(torch.softmax(out.float(), 1)[:, 1].cpu())
    model.train()
    met = ap_score(y.cpu().numpy(), torch.cat(ps).numpy()) if metrica == "pr_auc" else A / len(X)
    return L / len(X), A / len(X), met


def anexar_csv(caminho, linha):
    novo = not os.path.exists(caminho)
    with open(caminho, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if novo:
            w.writeheader()
        w.writerow({k: linha.get(k, "") for k in CAMPOS})


# ------------------------------------------------------------- execução ----
def executar(a):
    from dados import carregar
    t0 = time.time()
    rid = run_id(a)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    Xtr, ytr, Xva, yva, Xte, yte, din, ncls = [
        v.to(dev) if torch.is_tensor(v) else v for v in carregar(a.dados, a.frac)]
    epocas = a.steps * a.bs / len(Xtr)
    if epocas > 100:
        print(f"AVISO: {epocas:.0f} épocas sobre {len(Xtr)} amostras -> risco alto de sobreajuste; "
              f"reduza --steps/--epocas ou aumente --frac.", file=sys.stderr)
    model = MLP(din, ncls, a.width, a.depth).to(dev)
    metrica = a.metrica if a.metrica != "auto" else ("pr_auc" if ncls == 2 else "acc")
    cnt = torch.bincount(ytr, minlength=ncls).float().clamp(min=1)
    W = None
    if a.peso_classe != "none":                       # classe rara pesa mais na perda de TREINO
        W = cnt.sum() / (ncls * cnt)
        W = (W.sqrt() if a.peso_classe == "sqrt" else W).to(dev)
    opts = construir_otimizadores(model, a)
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lambda t: fator(a.sched, t, a.steps)) for o in opts]
    P = dict(model.named_parameters())
    CAM = camadas_log(model, a.depth)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    ev_cada = a.k * max(1, a.steps // (30 * a.k))
    max_norm = a.clip if a.clip > 0 else float("inf")

    log, divergiu, loss0, soma, n_soma = [], False, None, 0.0, 0
    for t in range(a.steps):
        idx = torch.randint(0, len(Xtr), (a.bs,), generator=gen).to(dev)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx], weight=W)
        loss.backward()
        lv = loss.item()
        if loss0 is None:
            loss0 = lv
        if not math.isfinite(lv) or lv > 100 * max(loss0, 1.0):
            divergiu = True
        soma += lv; n_soma += 1
        registrar = (t % a.k == 0) or t == a.steps - 1 or divergiu
        if registrar:
            gcam = {n: P[n].grad.norm().item() for n in CAM if P[n].grad is not None}
            th = torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())).item()
            thcam = {n: P[n].detach().norm().item() for n in CAM}
            antes = {n: P[n].detach().clone() for n in CAM}
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # norma ANTES do corte
        if divergiu:
            log.append(dict(passo=t, perda=lv, perda_media=soma / n_soma, grad=float(g), theta=float("nan"),
                            lr=opts[0].param_groups[0]["lr"]))
            break
        for o in opts:
            o.step()
        lr_atual = opts[0].param_groups[0]["lr"]            # lr usado NESTE passo
        for s in scheds:
            s.step()
        if registrar:
            linha = dict(passo=t, perda=lv, perda_media=soma / n_soma, grad=float(g), theta=th, lr=lr_atual)
            if len(opts) > 1:
                linha["lr_adamw"] = opts[1].param_groups[0]["lr"]
            for n in CAM:
                linha[f"grad/{n}"] = gcam.get(n, float("nan"))
                linha[f"theta/{n}"] = thcam[n]
                linha[f"rho/{n}"] = ((P[n].detach() - antes[n]).norm() / (antes[n].norm() + 1e-12)).item()
            if t % ev_cada == 0 or t == a.steps - 1:
                linha["val_loss"], linha["val_acc"], linha["val_metrica"] = avaliar(model, Xva, yva, metrica)
            log.append(linha)
            soma, n_soma = 0.0, 0

    if divergiu:
        vl, va, vm = float("nan"), float("nan"), float("nan")
    else:
        vl, va, vm = avaliar(model, Xva, yva, metrica)
    # grava log
    pasta = os.path.join(a.saida, "logs"); os.makedirs(pasta, exist_ok=True)
    cols = list(dict.fromkeys(k for r in log for k in r))
    with open(os.path.join(pasta, rid + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(log)
    if a.ckpt and not divergiu:
        os.makedirs(os.path.join(a.saida, "ckpt"), exist_ok=True)
        torch.save(dict(sd=model.state_dict(), din=din, ncls=ncls, width=a.width, depth=a.depth,
                        dados=a.dados, metrica=metrica), os.path.join(a.saida, "ckpt", rid + ".pt"))
    linha = dict(id=rid, etapa=a.etapa, opt=a.opt, lr=a.lr, lr_adamw=a.lr_adamw if a.opt == "muon" else "",
                 sched=a.sched, seed=a.seed, steps=a.steps, epocas=round(epocas, 1), bs=a.bs, frac=a.frac, dados=a.dados,
                 metrica=metrica, val_metrica=vm, val_acc=va, val_loss=vl, divergiu=int(divergiu), tempo_s=round(time.time() - t0, 2),
                 comando=comando(a))
    anexar_csv(os.path.join(a.saida, "experimentos.csv"), linha)
    return linha


if __name__ == "__main__":
    r = executar(get_args())
    print({k: r[k] for k in ("id", "metrica", "val_metrica", "val_acc", "divergiu", "tempo_s")})
