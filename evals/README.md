# Evals

The evals measure how well the search finds the right document for a question.

```bash
make evals v1                 # same as: uv run python -m evals.runner v1
```

Ollama must be running with the model of the version's `config.json` pulled (`ollama pull mxbai-embed-large`).

## Version folders

Each folder under `evals/` is one version of the eval: the documents, the questions, the settings and the results of
the last run. To try a change, copy a folder and change one thing, then run both and compare their `output.json`.

```
evals/
  runner.py            the code that runs a version folder
  v1/
    config.json        settings of the run
    tasks.json         the questions
    data/              the documents to index
    ostra.db           the database of the last run (written by the run, not committed)
    output.json        the results of the last run (written by the run)
```

```bash
cp -r evals/v1 evals/v2       # then edit evals/v2/config.json, e.g. "chunk_size_tokens": 256
make evals v2
```

### config.json

| Key | Meaning | Default |
|---|---|---|
| `embedding_model` | `mxbai-embed-large` or `bge-m3` | `mxbai-embed-large` |
| `chunk_size_tokens` | Chunk size in tokens | default of the model (510) |
| `chunk_overlap_tokens` | Tokens shared by two neighbor chunks | 50 |
| `search_limit` | Chunks returned by each search. The expected file must be among them to get a rank | 50 |
| `ollama_base_url` | Address of the Ollama server | `http://localhost:11434` |

The run always uses `<version>/ostra.db` and `<version>/data/`, so it never touches your own index.

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

1. `setup()` deletes the database of the previous run and creates an empty one in the version folder.
2. Every file in `data/` is chunked, embedded and stored in that database.
3. Each question is embedded and searched. The chunks are grouped by file, so a file that has three matching
   chunks counts once, at the position of its best chunk.
4. The results go to `output.json` and a short report is printed.

### output.json

Per task:

| Field | Meaning |
|---|---|
| `seconds` | Time of the search: embedding the question plus the vector search |
| `first_result` | The file of the nearest chunk |
| `first_distance` | Cosine distance of the nearest chunk (0 is the same direction) |
| `expected_rank` | Position of the expected file among the found files, starting at 1. `null` when it is not in the results |
| `top_documents` | The first five files found |

The `summary` totals the run:

| Field | Meaning |
|---|---|
| `hit_at_1`, `hit_at_3`, `hit_at_5` | Share of tasks whose expected file is in the first 1, 3 or 5 files |
| `mrr` | Mean reciprocal rank: the mean of 1/rank (0 when not found). 1.0 means every expected file came first |
| `not_found` | Tasks whose expected file was not in the results at all |
| `mean_seconds`, `median_seconds`, `max_seconds` | Search times |

`indexing.seconds` is the time to embed and store all documents.

## How big should the eval be?

For a first eval, **10–20 documents and 30–50 questions** is enough to spot clear problems and compare settings.
The number of questions matters more than the number of documents: with 30 questions one task is 3.3 percentage
points, so a change of one or two tasks between runs is noise. Grow to 100+ questions when you need to tell apart
two settings that are close. Documents on related topics (two universities, two places in California, two
French-history articles) make the eval harder and more realistic, because a wrong file can look right.

`v1` has 15 documents and 30 questions, two per document.

## Data of v1

The documents of `v1/data/` are the passages of 15 English Wikipedia articles as published in the development set of
SQuAD 2.0 (Rajpurkar et al., 2018, <https://rajpurkar.github.io/SQuAD-explorer/>). The text is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); the authors are the Wikipedia contributors of
each article. The paragraphs of each article are joined into one file. The questions were written for this eval,
based on SQuAD questions, and reworded so that each one makes sense without the article in front of you.
