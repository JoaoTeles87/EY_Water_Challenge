# Guia Técnico: Previsão de Qualidade da Água (Explicação para Estudantes de Computação)

Este documento traduz e explica os conceitos técnicos e o código do "Baseline Model" para o contexto acadêmico brasileiro de Ciência da Computação.

---

## 1. O Problema (Data Science Context)

Estamos tentando prever a qualidade da água em rios baseando-nos **apenas** no que satélites veem do espaço e em dados climáticos. É um problema de **Regressão Multi-Saída** (Multi-Output Regression), pois queremos prever 3 valores contínuos ao mesmo tempo.

### Os 3 Alvos (Targets)
1.  **Alcalinidade Total (Buffering Capacity):**
    *   *O que é:* Capacidade da água de resistir a mudanças de pH (como um "escudo" químico).
    *   *Visto do Espaço:* Alta alcalinidade geralmente vem de rochas dissolvidas, que alteram a cor da água ou estão associadas a certas vegetações nas margens.
2.  **Condutância Elétrica (Electrical Conductance):**
    *   *O que é:* Capacidade de conduzir eletricidade. Água pura não conduz; água com **sais** (poluição ou mar) conduz muito.
    *   *Visto do Espaço:* O satélite Landsat tem bandas infravermelhas (SWIR) que detectam minerais e salinidade no solo/água.
3.  **Fósforo Reativo Dissolvido (Phosphorus):**
    *   *O que é:* Nutriente que alimenta algas. Em excesso, causa "eutrofização" (água verde e tóxica).
    *   *Visto do Espaço:* Detectamos o "verde" das algas usando infravermelho próximo (NIR) e índices de vegetação (NDWI).

---

## 2. O Código `day1_baseline.py` Explicado

O script Python automatiza o processo de "ETL" (Extract, Transform, Load) e Modelagem. Aqui está o que cada função faz:

### A. `load_and_preprocess_data()`
*   **Desafio:** O Python é chato com números decimais. `-28.76083333` não é igual a `-28.7608`.
*   **Solução:** Arredondamos tudo para 4 casas decimais (~11 metros). Isso cria uma "chave primária" artificial para juntar (merge) os dados do satélite com os dados da água.
*   **Merge:** Usamos `pd.merge` (parecido com SQL JOIN) para combinar as tabelas.

### B. `create_features()`
*   **Feature Engineering:** O modelo não entende datas como "Janeiro".
*   **Seno/Cosseno:** Transformamos o mês em um ciclo.
    *   Janeiro (1) fica perto de Dezembro (12) no círculo trigonométrico. Se usássemos apenas números (1 a 12), Dezembro estaria longe de Janeiro, o que é errado climaticamente.

### C. `get_spatial_groups()`
*   **O "Pulo do Gato":** Em dados geográficos, não podemos usar `train_test_split` aleatório.
*   **Por quê?** Se eu tenho duas amostras do mesmo rio, uma no dia 1 e outra no dia 2, e coloco uma no treino e outra no teste, o modelo "decora" o rio em vez de aprender a ver a água pelo satélite.
*   **Solução (KMeans):** Agrupamos os pontos próximos em 10 "bairros" (clusters). O modelo treina em 8 bairros e testa em 2 bairros inéditos. Isso simula a vida real (generalização).

### D. `train_and_evaluate()`
*   **Modelo (XGBoost):** Usamos "Gradient Boosting". É uma técnica que cria centenas de "árvores de decisão" (if-else), onde cada nova árvore tenta corrigir o erro da anterior. É o estado da arte para dados tabulares.
*   **MultiOutputRegressor:** O sklearn permite que treinemos um modelo para os 3 alvos de uma vez, simplificando o código.

---

## 3. Próximos Passos (Dicas para o Times)

1.  **Melhorar o Merge:** Se o arredondamento de 4 casas perder dados, tentem usar uma biblioteca chamada `cKDTree` para buscar o vizinho mais próximo geoespacialmente (Nearest Neighbor).
2.  **Validar o Erro:** O erro atual é "RMSE" (Raiz do Erro Quadrático Médio). Quanto menor, melhor.
3.  **Explorar o "implementation.md":** Lá tem a estratégia completa para vencer o desafio.

Este código é um "esqueleto". O trabalho de vocês agora é adicionar "músculos" (novas features de satélite) e "cérebro" (otimizar o modelo).
