# Evals

The evals measure how well the search finds the right document for a question.

```bash
make evals v1-baseline        # same as: uv run python -m evals.runner v1-baseline
```

A run with the default model `potion-multilingual-128m` needs nothing else: the model downloads on first use. A run with an Ollama model needs Ollama running with that model pulled (`ollama pull mxbai-embed-large`).

## Layout

Datasets and runs are kept apart. A **dataset** is the documents and the questions. A **run** is one set of settings
tried against one dataset, with its results.

```
evals/
  runner.py                     the code that runs a run folder
  datasets/
    wiki-v1/
      tasks.json                the questions
      data/                     the documents to index
  runs/
    v1-baseline/
      config.json               the dataset name and the settings
      ostra.db                  the database of the last run (written by the run, not committed)
      output.json               the results of the last run (written by the run)
```

**Do not edit a dataset once runs use it.** Runs compare only when they used the same data. To change the documents
or the questions, copy the dataset to a new name (`wiki-v2`) and change the copy. A new dataset starts a new
baseline: run your current best settings on it first, then compare new settings against that.

Every `output.json` records a `fingerprint` of its dataset: a SHA-256 hash of `tasks.json` and every document. Two
runs are comparable only when their fingerprints are the same. If they differ, the dataset changed between them.

## Trying a change

Copy a run, change one setting, run both, and compare their `output.json`:

```bash
cp -r evals/runs/v1-baseline evals/runs/v1-chunk256
# edit evals/runs/v1-chunk256/config.json: "chunk_size_tokens": 256
make evals v1-chunk256
```

Change one thing per run, so you know what moved the score.

### config.json

| Key | Meaning | Default |
|---|---|---|
| `dataset` | Folder name in `evals/datasets/`. Required | |
| `embedding_model` | `potion-multilingual-128m`, `mxbai-embed-large` or `bge-m3` | `potion-multilingual-128m` |
| `chunk_size_tokens` | Chunk size in tokens | default of the model (510) |
| `chunk_overlap_tokens` | Tokens shared by two neighbor chunks | 50 |
| `search_limit` | Chunks returned by each search. The expected file must be among them to get a rank | 50 |
| `ollama_base_url` | Address of the Ollama server, for the Ollama models | `OLLAMA_BASE_URL`, or `http://localhost:11434` |

A run always uses its own `ostra.db` and its dataset's `data/`, so it never touches your own index, even when
`OSTRA_` environment variables are set.

### tasks.json

A list of tasks. `expected` is the file name in `data/` that holds the answer. `answer` is not used for scoring yet;
it documents the task and is ready for answer-generation evals later.

```json
[
  {
    "id": 1,
    "question": "Which king signed the treaty of Saint-Clair-sur-Epte with Rollo?",
    "expected": "normans.txt",
    "answer": "King Charles III"
  }
]
```

## What a run does

1. `setup()` deletes the database of the previous run and creates an empty one in the run folder.
2. Every file in the dataset's `data/` is chunked, embedded and stored in that database.
3. Each question is embedded and searched. The chunks are grouped by file, so a file that has three matching
   chunks counts once, at the position of its best chunk.
4. The results go to `output.json` in the run folder, and a short report is printed.

### output.json

`dataset` holds the dataset's `name`, `fingerprint`, and its number of `documents` and `tasks`. `config` holds the
settings the run actually used. `indexing.seconds` is the time to embed and store all documents.

For each task:

| Field | Meaning |
|---|---|
| `seconds` | Time of the search: embedding the question plus the vector search |
| `first_result` | The file of the nearest chunk |
| `first_distance` | Cosine distance of the nearest chunk (0 is the same direction) |
| `expected_rank` | Position of the expected file among the found files, starting at 1. `null` when it is not in the results |
| `top_documents` | The first five files |

The `summary` totals the run:

| Field | Meaning |
|---|---|
| `hit_at_1`, `hit_at_3`, `hit_at_5` | Share of tasks whose expected file is in the first 1, 3 or 5 files |
| `mrr` | Mean reciprocal rank: the mean of 1/rank (0 when not found). 1.0 means every expected file came first |
| `not_found` | Tasks whose expected file was not in the results at all |
| `mean_seconds`, `median_seconds`, `max_seconds` | Search times |

## How big should a dataset be?

For a first eval, **10–20 documents and 30–50 questions** is enough to spot clear problems and compare settings.
The number of questions matters more than the number of documents: with 30 questions one task is 3.3 percentage
points, so a change of one or two tasks between runs is noise. Grow to 100+ questions when you need to tell apart
two settings that are close. Documents on related topics (two universities, two places in California, two
French-history articles) make the eval harder and more realistic, because a wrong file can look right.

If you tune settings against the same questions for a long time, you start tuning for those questions. Once a
dataset has enough questions, keep 20–30% of them in a separate dataset and check it only now and then, to confirm
that a gain is real.

## Dataset wiki-v1

15 documents and 30 questions, two per document.

The documents are the passages of 15 English Wikipedia articles as published in the development set of SQuAD 2.0
(Rajpurkar et al., 2018, <https://rajpurkar.github.io/SQuAD-explorer/>). The text is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); the authors are the Wikipedia contributors of
each article. The paragraphs of each article are joined into one file. The questions were written for this eval,
based on SQuAD questions, and reworded so that each one makes sense without the article in front of you.
