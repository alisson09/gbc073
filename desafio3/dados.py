"""Carregamento de dados do Desafio 3.

Contrato: carregar(nome, frac, seed_split=0) devolve
    (Xtr, ytr, Xva, yva, Xte, yte, din, nclasses)
X float32 (N, din) padronizado, y int64.

- `frac` reduz SÓ o treino, de forma ESTRATIFICADA (a classe rara não some).
  Validação e teste nunca mudam. Split fixo (seed_split), independente da semente do treino.
- nome:  "creditcard.csv" (Kaggle ULB, fraude) | "digits" | "mnist" | "fashion" | "cifar10" | "arquivo.npz"
"""
import numpy as np
import torch

_RAW = {}   # cache: o CSV grande é lido uma única vez por processo


def _padroniza(Xtr, Xva, Xte):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    f = lambda X: ((X - mu) / sd).astype(np.float32)
    return f(Xtr), f(Xva), f(Xte)


def _split(X, y, rng, pva=0.2, pte=0.2, estratificar=False):
    if not estratificar:
        idx = rng.permutation(len(X))
        nva, nte = int(pva * len(X)), int(pte * len(X))
        va, te, tr = idx[:nva], idx[nva:nva + nte], idx[nva + nte:]
    else:                                        # proporção de cada classe mantida nos 3 conjuntos
        va, te, tr = [], [], []
        for c in np.unique(y):
            ic = rng.permutation(np.where(y == c)[0])
            nva, nte = int(pva * len(ic)), int(pte * len(ic))
            va += list(ic[:nva]); te += list(ic[nva:nva + nte]); tr += list(ic[nva + nte:])
        va, te, tr = (rng.permutation(np.array(v)) for v in (va, te, tr))
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def _bruto(nome, seed_split):
    rng = np.random.RandomState(seed_split)
    if nome == "digits":
        from sklearn.datasets import load_digits
        d = load_digits()
        return _split(d.data.astype(np.float32), d.target, rng)
    if nome.endswith(".csv"):                    # creditcard.csv: Time, V1..V28, Amount, Class
        import pandas as pd
        d = pd.read_csv(nome)
        y = d["Class"].values.astype(np.int64)
        X = d.drop(columns=[c for c in ("Time", "Class") if c in d.columns])
        if "Amount" in X:
            X["Amount"] = np.log1p(X["Amount"].clip(lower=0))   # valor tem cauda muito pesada
        return _split(X.values.astype(np.float32), y, rng, estratificar=True)
    if nome in ("mnist", "fashion", "cifar10"):
        import torchvision as tv
        cls = {"mnist": tv.datasets.MNIST, "fashion": tv.datasets.FashionMNIST,
               "cifar10": tv.datasets.CIFAR10}[nome]
        ds, dt = cls("./data", train=True, download=True), cls("./data", train=False, download=True)
        X = np.asarray(ds.data, dtype=np.float32).reshape(len(ds), -1) / 255.0
        y = np.asarray(ds.targets, dtype=np.int64)
        idx = rng.permutation(len(X)); nva = len(X) // 6
        Xte = np.asarray(dt.data, dtype=np.float32).reshape(len(dt), -1) / 255.0
        return (X[idx[nva:]], y[idx[nva:]], X[idx[:nva]], y[idx[:nva]], Xte,
                np.asarray(dt.targets, dtype=np.int64))
    if nome.endswith(".npz"):
        z = np.load(nome)
        a = [z[k] for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte")]
        for i in (0, 2, 4):
            a[i] = a[i].reshape(len(a[i]), -1).astype(np.float32)
        return tuple(a)
    raise ValueError(f"dados desconhecidos: {nome}")


def carregar(nome="digits", frac=1.0, seed_split=0):
    if (nome, seed_split) not in _RAW:
        _RAW[(nome, seed_split)] = _bruto(nome, seed_split)
    Xtr, ytr, Xva, yva, Xte, yte = _RAW[(nome, seed_split)]
    if frac < 1.0:                               # subamostra estratificada e fixa do treino
        rs, sub = np.random.RandomState(seed_split + 1), []
        for c in np.unique(ytr):
            ic = rs.permutation(np.where(ytr == c)[0])
            sub += list(ic[:max(2, int(round(frac * len(ic))))])
        sub = np.array(sorted(sub)); Xtr, ytr = Xtr[sub], ytr[sub]
    Xtr, Xva, Xte = _padroniza(Xtr, Xva, Xte)
    t = lambda a, d: torch.from_numpy(np.ascontiguousarray(a)).to(d)
    L, F = torch.long, torch.float32
    return (t(Xtr, F), t(ytr, L), t(Xva, F), t(yva, L), t(Xte, F), t(yte, L),
            Xtr.shape[1], int(max(ytr.max(), yva.max(), yte.max())) + 1)
