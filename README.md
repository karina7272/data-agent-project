# Data Agent Project

This repository contains a simple yet powerful **data analysis assistant**
written in Python.  It was developed as a solution to the SynMax
Take‑Home Data Agent assignment.  The goal of the project is to
create a chat‑based agent that can ingest a tabular dataset, understand
natural‑language questions, perform both deterministic and analytical
tasks, and return concise answers with supporting evidence.

## Features

- **Dataset ingestion**
  - Automatically loads CSV or Parquet files and infers the schema.
  - Handles missing values by dropping or imputing means/modes.
  - Downloads the dataset at runtime if a remote URL is provided.
  - Prompts for a local path if neither a path nor URL is supplied.  This
    interactive prompt ensures you can work with local files without ever
    committing the dataset to version control.  Downloaded files are
    stored under a `data/` folder which is ignored by git.

- **Natural‑language interface**
  - Uses a configurable LLM to translate user questions into executable
    Python code.  By default it calls OpenAI's ChatCompletion API, but you
    can set `LLM_PROVIDER=anthropic` to use Anthropic's Claude models
    instead (assuming an `ANTHROPIC_API_KEY` is configured).  Only the code
    is returned and executed in a sandboxed environment.
  - Provides a rule‑based fallback for simple aggregate queries when an
    API key is not available.

- **Analytical capabilities**
  - Summary statistics and correlation matrices for numeric columns.
  - K‑means clustering on selected features.
  - Anomaly detection using z‑score filtering.

- **Extensible architecture**
  - `DataHandler` encapsulates all data loading and preprocessing
    logic.
  - `ChatAgent` orchestrates prompt construction, LLM calls, code
    execution and response formatting.
  - The system can be easily extended to support additional LLM
    providers (e.g. Anthropic) or more advanced analyses (e.g.
    causal inference, forecasting).

## Installation

Clone this repository and install the Python dependencies.  We
recommend using a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you plan to use an LLM integration, set the relevant API key in your
environment:

```bash
# OpenAI
export OPENAI_API_KEY="sk-…"
# Anthropic (optional)
export ANTHROPIC_API_KEY="claude-…"

# Choose provider/model (optional)
export LLM_PROVIDER="openai"      # or "anthropic"
export LLM_MODEL="gpt-4-turbo"   # or your preferred Claude/GPT model
```

The agent defaults to OpenAI's `gpt-4-turbo`.  If you specify `LLM_PROVIDER` and
`LLM_MODEL`, those will override the defaults.  See `data_agent/agent.py` for
implementation details.

## Quick Start

1. **Obtain the dataset.**  Download the file from the link provided by
   SynMax and save it locally.  Do **not** commit the dataset to your
   repository.  You may also rely on the agent to download it when you
   provide `DATA_AGENT_DATA_URL` (see below).

2. **Run the CLI.**  You can start the agent in several ways:

   - **Local path**: pass the dataset via `--data`:

     ```bash
     python main.py --data path/to/dataset.csv
     ```

   - **Environment variable**: set `DATA_AGENT_DATA_PATH` to the file path and
     invoke `python main.py` with no `--data` option.

   - **Remote URL**: set `DATA_AGENT_DATA_URL` to the dataset link.  On first
     run, the file will be downloaded into a `data/` directory and cached
     for subsequent runs.

   - **Interactive prompt**: if no path or URL is provided, the CLI will ask
     you to enter a local path at runtime.

   In all cases the downloaded or specified dataset is stored under
   `./data/` and ignored by Git.

3. **Ask questions.**  Once the agent is ready you can type natural
   language questions about the data.  For example:

   ```
   > count of customer_id
   Answer: 12345
   > average of revenue
   Answer: 42.17
   > Which three variables show the strongest correlation?
   Answer: Computed a DataFrame with shape (3, 3)
   (The CLI prints the correlation submatrix for the top three features.)
   ```

   If an OpenAI API key is configured the agent will delegate
   general questions to the LLM.  Otherwise, only simple "count",
   "average" and "sum" queries are supported.

## Examples

To illustrate the agent's capabilities, consider a dataset with
columns `age`, `income`, `department` and `bonus`.  The following
queries demonstrate deterministic retrieval, clustering and anomaly
detection:

1. **Simple aggregate.**

   ```
   > count of department
   Answer: 200
   ```

2. **Correlations.**

   ```
   > Which variables are most correlated with income?
   Answer: Computed a DataFrame with shape (3, 3)
   age    0.78
   bonus  0.64
   ...
   ```

3. **Clustering.**

   ```
   > Find 3 clusters in the numeric columns and show the cluster sizes.
   Answer: Computed a Series with length 3
   cluster
   0    80
   1    70
   2    50
   ```

4. **Anomaly detection.**
\n+   ```
   > Identify outliers in the bonus column (z > 3).
   Answer: Computed a DataFrame with shape (5, 4)
   (Displays the 5 rows where bonus is more than 3 standard deviations above the mean.)
   ```

These examples are purely illustrative.  The actual dataset may have
different column names and distributions.

## Assumptions & Limitations

- The agent assumes the dataset is small enough to fit in memory.
- Missing values are either dropped or imputed.  Imputation uses the
  mean for numeric features and the mode for categorical features.
- Anomaly detection relies on a simple z‑score threshold; more robust
  methods could be implemented if necessary.
- The agent executes LLM‑generated code in a restricted namespace to
  mitigate security risks; however, always exercise caution when
  connecting arbitrary code to external resources.
- The current implementation uses OpenAI's API.  Anthropic support is
  not implemented but can be added by following the pattern in
  `ChatAgent`.

## Project Structure

- `data_agent/`: core library containing the data handler and chat agent.
- `main.py`: CLI entry point.
- `requirements.txt`: Python dependencies.
- `report.md`: Project report detailing the architecture, design decisions,
  methodology and future work.
- `answer.pptx`: Slide deck summarising the solution.

Please read **report.md** for a more in‑depth discussion of the approach
and design rationale.

## Architecture (at a glance)

![Data Agent — High‑Level Architecture](docs/architecture.png)

*Flow:* **Data sources** → **Ingestion & Validation** (schema inference, missing values, stats) → **ChatAgent Orchestrator** (NLQ → plan → code) → **Sandboxed Python Executor** → **Answers & Evidence** (method, columns, filters, hypothesis) + **Logging/Artifacts** and **Security**.

## Evaluation (scripted queries)

**Harness:** five pre-written queries covering counts, means, group-by means, filters, and a custom Python expression.  
**Weighting:** Accuracy 70 %, Speed 30 %.

### Per-Query Results

| # | Query Type   | Correct? | Latency (s) |
|---|--------------|:--------:|------------:|
| 1 | count        | No       | 0.000 |
| 2 | mean         | No       | 0.202 |
| 3 | groupby_mean | No       | 0.000 |
| 4 | filter_count | No       | 0.141 |
| 5 | python       | No       | 0.000 |

Average latency ≈ **0.069 s** (very fast).

### Scoring

| Metric            | Value |
|-------------------|------:|
| Accuracy (70 %)   | 0.000 |
| Avg latency (s)   | 0.069 |
| Speed score       | 0.936 |
| **Final score**   | **0.281** |

**Diagnosis.** Speed is excellent; accuracy is limited by the deterministic fallback not applying some **filters/group-bys** exactly as the test expects. Tightening the parser/dispatch and adding unit tests that mirror the evaluation queries will raise the accuracy component substantially.

### Bonus insights

- **High missingness.** 100 % empty: `longitude`, `latitude`. ~78 % missing: `connecting_pipeline`. Additional gaps in `connecting_entity`, `county_name`.
- **Outliers.** 4,097 rows flagged by z-score; prefer median/trimmed means or exclude outliers for mean calculations.
- **Collinearity.** `rec_del_sign` ≈ `scheduled_quantity` (near-perfect redundancy); avoid using both in modeling or reduce dimensionality.
- **K-means segmentation.** Two very large clusters and one tiny cluster (859 rows) with extremely high `scheduled_quantity` and positive `rec_del_sign`—likely special-purpose records or data-entry artifacts; merits review.

### Next steps

- **Fix deterministic path:** exact semantics for `count/mean/groupby/filter_count`; add unit tests matching the scripted queries.
- **Robust stats:** median/trimmed mean toggle; optional outlier filtering before aggregates.
- **Missing data:** drop or impute highly missing columns; confirm business relevance of `connecting_pipeline`.
- **Multicollinearity:** remove redundant variables or apply dimensionality reduction.
- **Segment follow-up:** investigate the small high-quantity cluster for legitimacy vs. data errors.
