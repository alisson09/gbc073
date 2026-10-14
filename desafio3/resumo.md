# Resumo

| opt   |    lr |   lr_adamw | sched   |   n |    media |        dp |
|:------|------:|-----------:|:--------|----:|---------:|----------:|
| muon  | 0.001 |      0.001 | cos     |   3 | 0.797822 | 0.0113738 |
| muon  | 0.001 |      0.001 | warmcos |   3 | 0.794962 | 0.0104207 |
| muon  | 0.001 |      0.001 | wsd     |   3 | 0.778799 | 0.0115712 |
| adamw | 0.001 |    nan     | const   |   3 | 0.728372 | 0.0602512 |

{
 "f_busca": 0.25,
 "f_full": 1.0,
 "vencedor": "muon lr=0.001 (+AdamW 0.001) cos",
 "baseline": "adamw lr=0.001 const",
 "media_venc": 0.7978221496696184,
 "dp_venc": 0.011373836640149744,
 "media_base": 0.7283719709896083,
 "dp_base": 0.060251197546771745,
 "ganho": 0.06945017868001013,
 "ganho_significativo": true,
 "diagnostico_auto": [
  "\u2016g\u2016: 13 picos na baseline, 24 na melhor.",
  "\u2016\u03b8\u2016 final/inicial: baseline 1.08\u00d7, melhor 1.02\u00d7 (\u03b7, weight decay e agenda).",
  "Decaimento final da agenda derruba a perda em -0.001 nos \u00faltimos 20% dos passos (melhor) vs 0.018 (baseline).",
  "Raz\u00e3o de atualiza\u00e7\u00e3o mediana em layers.0.weight: Muon 6.9e-04 vs AdamW 4.4e-04.",
  "Raz\u00e3o de atualiza\u00e7\u00e3o mediana em layers.2.weight: Muon 3.0e-04 vs AdamW 7.0e-04.",
  "Raz\u00e3o de atualiza\u00e7\u00e3o mediana em head.weight: Muon 3.9e-04 vs AdamW 8.0e-04."
 ]
}