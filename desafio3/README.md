# Desafio 3 — GBC073 (otimizadores e agendas, observados por dentro)

## Como rodar (ordem)
```bash
pip install torch numpy pandas matplotlib tabulate scikit-learn      # PyTorch >= 2.x
python treino.py --opt muon --lr 0.02 --sched warmcos --frac 0.25    # 1 execução (teste de fumaça)
python run_protocolo.py --rapido                                     # protocolo inteiro em 100 passos (verifica tudo)
rm -rf experimentos.csv logs ckpt config_baseline.json               # limpa o teste de fumaça
python run_protocolo.py --frac_busca 0.25 --epocas 30              # protocolo real (retomável)
python analisar.py                                                   # painel.pdf, slide.pdf, resumo.md/json
python teste_final.py                                                # teste usado UMA vez (vencedor + baseline)
python analisar.py                                                   # reexecute para pôr o teste no slide
```
## Antes de tudo: ligue ao SEU projeto
1. `dados.py` → `carregar(...)` (use `--dados caminho.npz` ou edite). 2. `treino.py` → classe `MLP`
(troque pelo seu modelo; mantenha os nomes `layers.*` e `head`). 3. Ajuste `--base_opt/--base_lr/--base_sched`
de `run_protocolo.py` para a receita atual do grupo.

## Estratégia do professor (poucos dados → conjunto completo)
Etapas 2–3 (varredura de η e agendas): `--frac_busca` do treino. Etapa 1 (baseline) e 4 (confirmação, 3 sementes): treino completo.
Mesmo `--steps`/`--bs` em todas as execuções; validação e teste nunca são reduzidos.

## Arquivos gerados
experimentos.csv (1 linha/execução, com coluna `comando` para reproduzir) • logs/ • painel.pdf • slide.pdf • resumo.md

## Fraude (creditcard.csv)
`--dados creditcard.csv --metrica pr_auc --bs 512 --epocas 10`. Split 60/20/20 estratificado; `--frac_busca` também estratificado;
perda de treino ponderada (`--peso_classe sqrt|inv|none`); seleção por PR-AUC (average precision). Colunas `val_metrica`/`test_metrica` no CSV.
