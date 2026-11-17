# CitizenTech — Zonas de Risco de São Paulo

Projeto acadêmico de Machine Learning que agrupa distritos de São Paulo em **Zonas de Risco**, cruzando vulnerabilidade social (famílias em extrema pobreza) e ocorrências de alagamento, para apoiar a Defesa Civil na priorização preventiva de resgates.

> **Importante:** o projeto não gera dados simulados. As duas bases são fornecidas separadamente no arquivo `databases.tar.gz`.

## Estrutura do projeto

```
.
├── databases.tar.gz                 # você recebe este arquivo
├── ObservaSampaDadosAbertosIndicadoresCSV.csv   # obtido após extrair o tar
├── SP156-Database.csv                            # obtido após extrair o tar
├── requirements.txt
├── README.md
└── ml/
    ├── notebook_citizentech.ipynb   # pipeline executado
    └── RELATORIO_ML.md              # relatório completo
```

## Passo a passo de execução

### 1. Extraia as bases de dados na raiz do projeto

Coloque `databases.tar.gz` na raiz do projeto (a mesma pasta deste README) e extraia:

```bash
tar -xzf databases.tar.gz
```

No Windows, o comando acima funciona no PowerShell ou no Terminal; também é possível usar o 7-Zip e escolher "Extrair aqui".

Após a extração, confirme que estes dois arquivos estão na raiz:

```bash
# Windows (PowerShell)
Get-ChildItem *.csv

# Linux / macOS
ls *.csv
```

Esperado:

- `ObservaSampaDadosAbertosIndicadoresCSV.csv`
- `SP156-Database.csv`

Se o tar trouxer os CSVs dentro de uma pasta, mova-os para a raiz. Caso você os renomeie para `pobreza_observasampa.csv` e `alagamentos_sp156.csv`, o notebook também os localiza automaticamente.

### 2. Instale as dependências

Recomenda-se um ambiente virtual:

```bash
python -m venv .venv
```

Ative-o:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (CMD)
.\.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

Instale os pacotes:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Execute o notebook

Abra o Jupyter e execute todas as células:

```bash
python -m jupyter notebook ml/notebook_citizentech.ipynb
```

Ou execute em lote pela linha de comando, atualizando o próprio arquivo com as saídas:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace ml/notebook_citizentech.ipynb
```

O notebook já vem executado; a reexecução serve para validar o ambiente ou atualizar as saídas após trocar as bases.

### 4. Leia os resultados

- `ml/notebook_citizentech.ipynb` — limpeza, merge, `StandardScaler`, KMeans com `k = 2..10`, gráfico do cotovelo, Silhouette Score, coluna `Cluster` e identificação do cluster prioritário.
- `ml/RELATORIO_ML.md` — análise completa, metodologia e significado para o cidadão.

Saídas esperadas (dados de 2025): 96 distritos após o merge, 1.326 chamados de alagamento/inundação, cotovelo em `k = 3` (Silhouette ≈ 0,549) e **Cluster de Prioridade Máxima** formado por Jardim Helena e Vila Jacuí.

## Solução de problemas

| Sintoma | Verificação |
|---|---|
| `FileNotFoundError` ao ler os CSVs | Os dois arquivos precisam estar na raiz do projeto (ou usar os nomes genéricos aceitos pelo notebook). |
| `ModuleNotFoundError` | O ambiente virtual não foi ativado; repita o passo 2. |
| Execução lenta | O `SP156-Database.csv` tem ~485 MB e ~2,6 milhões de linhas; o notebook o lê em blocos de 200 mil, mas a primeira execução pode levar alguns minutos. |
| Acentos estranhos no terminal | É apenas o encoding do console; o notebook e o relatório estão em UTF-8 e são exibidos corretamente no Jupyter. |
| `nbconvert` não encontrado | Confirme que `requirements.txt` foi instalado no ambiente ativo. |

## Requisitos

- Python 3.12 (as versões exatas das bibliotecas estão em `requirements.txt`).
- Espaço em disco suficiente para o SP156 descompactado (~485 MB).
