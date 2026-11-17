# Relatório de Machine Learning — CitizenTech: Zonas de Risco de São Paulo

**Projeto:** CitizenTech · **Entrega:** clusterização de distritos para priorização de resgates · **Recorte temporal:** 2025 · **Ferramentas:** Python, Pandas, Scikit-Learn, Matplotlib

---

## 1. Conceito da atividade e objetivo

Em eventos de chuva intensa, o tempo disponível para evacuar populações vulneráveis é curto e os recursos de resgate são finitos. Decidir *por onde começar* com base apenas na intuição ou na pressão do momento é caro e desigual: os distritos com mais famílias em extrema pobreza costumam ser também os que têm menor capacidade de resposta própria e maior exposição a alagamentos.

O CitizenTech transforma esse dilema em um problema de **clusterização não supervisionada**. Em vez de prever um rótulo conhecido, o algoritmo agrupa os distritos segundo duas dimensões observadas: **vulnerabilidade social** (quantidade de famílias em extrema pobreza) e **exposição a alagamentos** (quantidade de chamados registrados no SP156). Cada grupo resultante é uma *Zona de Risco* com um perfil interpretável, permitindo à Defesa Civil hierarquizar ações **antes** de o alerta se converter em emergência.

O objetivo de negócio é direto: **responder, com evidência, quais distritos devem receber barcos, ambulâncias e equipes do SUS primeiro** quando um alerta meteorológico severo for emitido. O modelo não substitui o planejador — ele reduz o tempo entre o alerta e a decisão e torna o critério auditável.

## 2. Dados utilizados

Foram usados **exclusivamente dados reais e públicos da Prefeitura do Município de São Paulo**, sem qualquer geração de valores sintéticos:

| Fonte | Arquivo entregue | O que foi usado | Volume |
|---|---|---|---|
| ObservaSampa — Indicadores | `ObservaSampaDadosAbertosIndicadoresCSV.csv` | Indicador `01.01.01 Quantidade de famílias em situação de extrema pobreza (até 1/4 salário mínimo)`, `Período = 2025` e `Região` do tipo `(Distrito)` | 206.198 linhas brutas → **96 distritos** |
| SP156 — Central de Atendimento | `SP156-Database.csv` | Chamados cujo `Serviço` é `Alagamento` ou `Inundação`, com `Data de Abertura` em 2025 e `Distrito` preenchido | 2.619.941 chamados brutos → **1.326 chamados** em 92 distritos |

Três decisões de limpeza foram necessárias e estão documentadas no notebook:

1. O ObservaSampa contém centenas de indicadores e vários anos. Mantivemos apenas o indicador de extrema pobreza e o ano 2025; das 129 linhas de 2025, 33 são subprefeituras ou o total do município e foram descartadas — elas não são unidades de clusterização. O sufixo `" (Distrito)"` foi removido para obter o nome limpo do bairro. O indicador varia de 84 a 18.360 famílias por distrito (média 5.025).
2. O SP156 fornecido não é uma base "só de alagamentos": ele inclui todos os temas do 156. Agrupar sem filtrar incluiria capinação, tapa-buraco e outros serviços sem relação com enchentes. O filtro por `Serviço` isolou os 1.326 chamados de Alagamento/Inundação; a leitura foi feita em blocos de 200 mil linhas para evitar carregar os 485 MB do arquivo de uma vez.
3. Os dois sistemas escrevem os nomes de forma diferente (`São Miguel (Distrito)` versus `SAO MIGUEL`). A chave de merge foi padronizada para maiúsculas sem acentos e sem pontuação. Usamos `pd.merge(..., how='left')` para manter os 96 distritos: cinco não tiveram chamados em 2025 e receberam valor `0`. Um rótulo do SP156 (`M BOI MIRIM`, na verdade uma subprefeitura) ficou fora da união.

Os arquivos reais usam `|` como delimitador (e não `;`), com encoding `utf-8` no ObservaSampa e `cp1252` no SP156. O notebook detecta o delimitador automaticamente, aplica esses encodings e também aceita os nomes genéricos caso os arquivos sejam renomeados.

## 3. Metodologia e ciclo de Machine Learning

**Entendimento e preparação.** A unidade de análise é o distrito. Cada linha final contém `Extrema_Pobreza` e `Total_Alagamentos`, além do nome original do distrito. A qualidade da união foi verificada por cardinalidade (96 distritos únicos, um para um) e pela contagem de valores nulos preenchidos.

**Pré-processamento.** As variáveis têm grandezas incomparáveis: famílias em extrema pobreza ficam na casa dos milhares, enquanto os chamados vão de 0 a 250. O KMeans otimiza distâncias euclidianas; sem padronização, a feature de maior escala dominaria a formação dos grupos e o modelo "esqueceria" os alagamentos. Aplicamos `StandardScaler`, deixando cada variável com média 0 e desvio padrão 1. É um passo de equalização de unidades, não de alteração da informação.

**Modelagem.** Escolhemos `KMeans` porque o problema é de segmentação de unidades em grupos homogêneos, sem rótulos prévios, com apenas duas features contínuas e um conjunto pequeno (96 observações). O algoritmo é determinístico com `random_state=42` e `n_init=10`, o que garante reprodutibilidade da entrega acadêmica.

**Avaliação.** Como não existe "resposta certa" em clusterização, usamos dois critérios complementares: o **Método do Cotovelo** (inércia/WCSS) e o **Silhouette Score**. O cotovelo foi detectado por programação pela maior segunda diferença da curva de inércia, evitando interpretação visual subjetiva.

## 4. Teste de hiperparâmetros e avaliação

Testamos `k = 2` a `10`. Resultados obtidos na execução do notebook:

| k | Inércia (WCSS) | Queda em relação a k−1 | Silhouette |
|---:|---:|---:|---:|
| 2 | 111,48 | — | **0,6290** |
| **3** | **54,98** | **56,50** | **0,5487** |
| 4 | 35,29 | 19,69 | 0,5011 |
| 5 | 23,88 | 11,41 | 0,5018 |
| 6 | 18,09 | 5,80 | 0,4508 |
| 7 | 14,64 | 3,45 | 0,4605 |
| 8 | 11,62 | 3,02 | 0,4707 |
| 9 | 9,75 | 1,88 | 0,4691 |
| 10 | 8,19 | 1,56 | 0,4640 |

A inércia sempre cai ao aumentar `k`, então o ponto de interesse é onde o ganho marginal desaparece: a queda é de 56,50 entre `k=2` e `k=3`, cai para 19,69 e depois para 11,41. A curvatura máxima ocorre em **k = 3**, que é o cotovelo. Nesse ponto o Silhouette é **0,5487**, acima do obtido com `k=4` (0,5011), confirmando grupos bem separados e internamente coerentes.

`k=2` apresenta o maior Silhouette (0,6290), mas é uma solução operacionalmente pobre: ela resume a cidade a "baixo risco" e "alto risco" e mistura dois perfis que exigem respostas diferentes — distritos com muitas famílias vulneráveis e poucos chamados, e distritos com alagamentos extremos. Como o objetivo é *priorizar* recursos, a granularidade adicional de `k=3` justifica a pequena perda de Silhouette. A escolha final foi, portanto, **k = 3**.

## 5. Resultados: as três Zonas de Risco

| Cluster | Distritos | Média de famílias em extrema pobreza | Média de alagamentos | Índice combinado de prioridade | Interpretação |
|---:|---:|---:|---:|---:|---|
| 0 | 68 | 2.901,50 | 6,82 | −2,15 | Monitoramento padrão |
| 1 | 26 | 10.317,23 | 18,77 | 0,35 | Segundo escalão: alta vulnerabilidade social |
| **2** | **2** | **8.422,00** | **186,50** | **1,80** | **Prioridade máxima** |

O índice combinado é a soma dos z-scores das médias de cluster, de modo a destacar o grupo simultaneamente elevado nas duas dimensões. O **Cluster 2** é o **Cluster de Prioridade Máxima**: apesar de ter a segunda maior média de extrema pobreza, sua média de alagamentos é quase **dez vezes** a do cluster seguinte. São apenas dois distritos, mas concentram o pior cruzamento possível:

| Distrito | Famílias em extrema pobreza (2025) | Chamados de alagamento/inundação (2025) |
|---|---:|---:|
| Jardim Helena | 8.974 | 250 |
| Vila Jacuí | 7.870 | 123 |

O Cluster 1 (26 distritos, incluindo Grajaú, Jardim Ângela e Brasilândia) representa vulnerabilidade social elevada com exposição hídrica moderada; possui 18,77 chamados em média, mas não apresenta os picos de 250 e 123 observados no Cluster 2.

## 6. Significado para o cidadão (conclusão)

A pergunta de negócio tem resposta prática: **com um alerta meteorológico severo, a Defesa Civil deve posicionar preventivamente barcos, ambulâncias e equipes do SUS em Jardim Helena e Vila Jacuí antes que a água interrompa o acesso**. São os distritos onde a combinação de 16.844 famílias em extrema pobreza e 373 chamados de alagamento indica que a população mais vulnerável é também a mais exposta — e a que tem menos condições de evacuar sem apoio público.

O modelo permite ainda uma estratégia em camadas: enquanto o Cluster 2 recebe pré-posicionamento de recursos, o Cluster 1 entra em vigilância reforçada com abrigos e equipes de saúde prontas, e o Cluster 0 permanece em monitoramento padrão. Isso transforma um alerta genérico em uma ordem de prioridade **rastreável, reproduzível e revisável** — o cidadão vulnerável deixa de depender da ordem em que os chamados chegam à central.

**Limitações e cuidados.** Chamados do 156 medem a procura por serviço, não o risco hidrológico objetivo; distritos mais populosos tendem a gerar mais chamados. O recorte é um retrato de 2025 e não normaliza por população, o que impede comparar taxas entre distritos de tamanhos diferentes. Além disso, a clusterização descreve associação, não causalidade. Recomenda-se reexecutar o notebook a cada atualização das bases e cruzar as prioridades com os mapas oficiais de risco hidrológico e geológico da Defesa Civil antes de consolidar o plano operacional.

---

### Reprodutibilidade

Arquivos da entrega: `ml/notebook_citizentech.ipynb` (código executado, com tabelas e gráficos) e `ml/RELATORIO_ML.md` (este documento). Para gerar novamente as saídas:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace ml/notebook_citizentech.ipynb
```

Semente aleatória fixa em `42`, `n_init=10`, padronização com `StandardScaler` e invariante de 96 distritos após o merge.
