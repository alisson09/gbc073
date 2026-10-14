# Tutorial: rodar o Desafio 3 no Google Colab

1. **Suba o zip:** envie `desafio3.zip` para o *Meu Drive* (raiz).
2. **Abra o notebook:** envie `desafio3_colab.ipynb` ao Drive e abra com o Colab (ou colab.research.google.com → Upload).
3. **Ative a GPU:** Ambiente de execução → Alterar tipo de ambiente de execução → **T4 GPU**.
4. **Rode as células em ordem:** montar Drive → descompactar → instalar `tabulate` → **teste de fumaça** (`--rapido`) → limpar → **protocolo real** → `analisar.py` → `teste_final.py` → `analisar.py` → baixar PDFs.
5. **Se a sessão cair:** remonte o Drive e rode de novo a célula do protocolo. Tudo é gravado em `MyDrive/desafio3/` e execuções prontas são puladas.
6. **Seus dados:** crie um `.npz` com `Xtr,ytr,Xva,yva,Xte,yte`, coloque em `MyDrive/desafio3/` e use `--dados meus_dados.npz`. Modelo próprio: edite a classe `MLP` em `treino.py` (mantenha os nomes `layers.*` e `head`).
7. **Entrega:** copie `treino.py`, `experimentos.csv`, `logs/`, `painel.pdf`, `slide.pdf` (e os demais scripts) para a pasta `desafio3/` do repositório e faça o commit **antes das 12h de 14/10**.

Dicas: se estiver lento, reduza `--epocas` ou `--frac_busca`; para só 4 otimizadores, tire `sgd` ou `sgdn` da lista em `varredura()` (o enunciado pede as duas variantes de SGD, então combine com o professor). Aviso de "épocas > 100" = risco de sobreajuste.

## Projeto de fraude (creditcard.csv, Kaggle ULB)
- Baixe o dataset no Kaggle (precisa de login), suba o `archive.zip` ou o `creditcard.csv` para a raiz do Drive e rode a célula "Dados" do notebook.
- Linha do protocolo (célula 6):
  `!python run_protocolo.py --dados creditcard.csv --metrica pr_auc --bs 512 --epocas 10 --frac_busca 0.25 --base_opt adamw --base_lr 1e-3 --base_sched const`
- A seleção usa **PR-AUC** (não acurácia: com 0,17% de fraudes, prever "tudo normal" já dá 99,8%). Perda de treino ponderada por classe (`--peso_classe sqrt`) e divisões/subamostras estratificadas.
- Troque `--base_*` pela receita que o grupo já usa.
