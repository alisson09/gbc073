#!/usr/bin/env python3
"""Executa o protocolo do Desafio 3 (retomável: execuções já presentes em experimentos.csv são puladas).

  etapa 1  baseline (receita atual) .......... 3 sementes, dados COMPLETOS
  etapa 2  varredura de lr por otimizador .... >=5 lrs, warmcos, 1 semente, dados REDUZIDOS (--frac_busca)
  etapa 3  agendas (2 melhores + Muon) ....... 4 agendas, melhor lr, 1 semente, dados REDUZIDOS
  etapa 4  confirmação (3 melhores + Muon) ... 3 sementes, dados COMPLETOS  (dica do professor)
Mesmo orçamento (steps, bs) em tudo. Teste NÃO é tocado aqui (ver teste_final.py).

  python run_protocolo.py                       # tudo
  python run_protocolo.py --rapido              # teste de fumaça (poucos passos)
  python run_protocolo.py --dados cifar10 --epocas 20 --frac_busca 0.2
"""
import argparse, csv, json, math, os
import math as _m
import treino
from dados import carregar

AGENDAS = ["const", "cos", "warmcos", "wsd"]           # 4ª à escolha: WSD
GRADES = {                                              # grade logarítmica (passo ~ 10^0.5)
    "adamw": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "adam": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "sgd": [3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    "sgdn": [3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    "muon": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    "rmsprop": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "nadam": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
}
FATOR = 10 ** 0.5
MAX_EXT = 4


def ler_csv(caminho):
    if not os.path.exists(caminho):
        return {}
    with open(caminho, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def score(r):
    """Métrica de seleção (maior = melhor); divergência vale -inf."""
    try:
        v = float(r["val_metrica"])
    except (TypeError, ValueError):
        return -math.inf
    return -math.inf if (int(r["divergiu"]) or math.isnan(v)) else v


class Protocolo:
    def __init__(self, A):
        self.A = A
        self.csv = os.path.join(A.saida, "experimentos.csv")
        self.N = len(carregar(A.dados, 1.0)[0])          # tamanho do treino completo

    def passos(self, frac):
        """Orçamento em ÉPOCAS constante: com menos dados, menos passos (evita sobreajuste na busca)."""
        if self.A.steps:                                  # --steps força um valor fixo
            return self.A.steps
        n = self.N if frac >= 1 else max(32, int(frac * self.N))
        return self.A.epocas * _m.ceil(n / self.A.bs)

    def rodar(self, etapa, opt, lr, sched, seed, frac, lr_adamw=3e-3, ckpt=False):
        A = self.A
        argv = ["--opt", opt, "--lr", f"{lr:g}", "--sched", sched, "--seed", str(seed),
                "--steps", str(self.passos(frac)), "--bs", str(A.bs), "--frac", f"{frac:g}", "--dados", A.dados,
                "--k", str(A.k), "--width", str(A.width), "--depth", str(A.depth),
                "--clip", str(A.clip), "--etapa", etapa, "--saida", A.saida,
                "--lr_adamw", f"{lr_adamw:g}", "--metrica", A.metrica, "--peso_classe", A.peso_classe] + (["--ckpt"] if ckpt else [])
        if A.rms_match and opt == "muon":
            argv.append("--rms_match")
        a = treino.get_args(argv)
        rid = treino.run_id(a)
        existentes = ler_csv(self.csv)
        if rid in existentes and (not ckpt or os.path.exists(os.path.join(A.saida, "ckpt", rid + ".pt"))):
            return existentes[rid]
        r = treino.executar(a)
        print(f"[{etapa}] {rid}: val={float(r['val_metrica']):.4f} div={r['divergiu']}", flush=True)
        return {k: str(v) for k, v in r.items()}

    # ---- etapa 1
    def baseline(self):
        A = self.A
        json.dump(dict(opt=A.base_opt, lr=A.base_lr, sched=A.base_sched),
                  open(os.path.join(A.saida, "config_baseline.json"), "w"))
        for s in range(3):
            self.rodar("1-baseline", A.base_opt, A.base_lr, A.base_sched, s, 1.0, ckpt=True)

    # ---- etapa 2
    def varredura(self):
        A, melhor_lr, tabela = self.A, {}, {}
        ordem = [o for o in ["adamw", "adam", "sgd", "sgdn", "muon"] + A.extras]
        for opt in ordem:                                     # adamw primeiro -> lr_adamw do Muon
            grade = list(GRADES[opt])
            lr_aw = melhor_lr.get("adamw", 3e-3)
            res, ext_lo, ext_hi = {}, 0, 0
            while True:
                for lr in grade:
                    if lr not in res:
                        res[lr] = score(self.rodar("2-lr", opt, lr, "warmcos", 0, A.frac_busca, lr_aw))
                b = max(res, key=lambda k: (res[k], -k))
                if b == min(res) and ext_lo < MAX_EXT:
                    ext_lo += 1; grade.append(float(f"{min(res) / FATOR:.2g}"))
                elif b == max(res) and ext_hi < MAX_EXT:
                    ext_hi += 1; grade.append(float(f"{max(res) * FATOR:.2g}"))
                else:
                    break
            melhor_lr[opt] = max(res, key=lambda k: (res[k], -k)); tabela[opt] = res
            print(f"   melhor lr {opt}: {melhor_lr[opt]:g} (val {res[melhor_lr[opt]]:.4f})", flush=True)
        self.melhor_lr, self.tab_lr = melhor_lr, tabela

    # ---- etapa 3
    def agendas(self):
        A = self.A
        rank = sorted(self.melhor_lr, key=lambda o: -self.tab_lr[o][self.melhor_lr[o]])
        alvo = rank[:2] + (["muon"] if "muon" not in rank[:2] else [])
        lr_aw = self.melhor_lr["adamw"]
        self.combos = []                                      # (score, opt, lr, sched)
        for opt in alvo:
            for ag in AGENDAS:
                r = self.rodar("3-agenda", opt, self.melhor_lr[opt], ag, 0, A.frac_busca, lr_aw)
                self.combos.append((score(r), opt, self.melhor_lr[opt], ag))
        self.lr_aw = lr_aw

    # ---- etapa 4
    def confirmar(self):
        A = self.A
        ord_ = sorted(self.combos, key=lambda c: -c[0])
        top = ord_[:3]
        if not any(c[1] == "muon" for c in top):
            top.append(next(c for c in ord_ if c[1] == "muon"))
        for _, opt, lr, ag in top:
            for s in range(3):
                self.rodar("4-confirma", opt, lr, ag, s, 1.0, self.lr_aw, ckpt=True)
        print("confirmação concluída:", [(o, f"{l:g}", g) for _, o, l, g in top])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dados", default="digits")
    p.add_argument("--epocas", type=int, default=30, help="orçamento em épocas (vale p/ dados reduzidos e completos)")
    p.add_argument("--steps", type=int, default=None, help="opcional: número fixo de passos (ignora --epocas)")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--metrica", default="auto", choices=["auto", "acc", "pr_auc"])
    p.add_argument("--peso_classe", default="sqrt", choices=["none", "sqrt", "inv"])
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.0)
    p.add_argument("--frac_busca", type=float, default=0.25, help="fração do treino nas etapas 2-3")
    p.add_argument("--saida", default=".")
    p.add_argument("--base_opt", default="adamw")
    p.add_argument("--base_lr", type=float, default=1e-3)
    p.add_argument("--base_sched", default="const")
    p.add_argument("--extras", nargs="*", default=[], help="ex.: rmsprop nadam (5º otimizador)")
    p.add_argument("--rms_match", action="store_true")
    p.add_argument("--rapido", action="store_true", help="smoke test: 100 passos")
    A = p.parse_args()
    if A.rapido:
        A.steps, A.k = 100, 10
    os.makedirs(A.saida, exist_ok=True)
    P = Protocolo(A)
    P.baseline(); P.varredura(); P.agendas(); P.confirmar()
    print("\nProximo passo: python analisar.py && python teste_final.py && python analisar.py")


if __name__ == "__main__":
    main()
