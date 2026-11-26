# Changelog

All notable changes to ostra are listed in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). ostra has no tagged release yet, so every change is under Unreleased.

## [Unreleased]

### Added

- `ostra search` in a terminal lists each matching file once, nearest first, and opens the file you pick in its default application. `--interactive` and `--no-interactive` choose the mode. Piped output, `--json` and `--full` print the chunk list as before. (#8)
- `ostra init` sets up the embedding model, the Ollama server address, the folders, the subfolder setting, the chunk settings and the database step by step. It saves only when you confirm. `ostra init --defaults` saves without questions, for scripts. (#5)
- After a change of the model, the folders, the subfolder setting, the chunk settings or the database, `ostra init` lists the changes and offers to run `ostra index`. (#6)
- The `recursive` config key, and `ostra index --recursive` and `--no-recursive` to override it for one run. (#5)
- The `potion-multilingual-128m` embedding model. It runs in the ostra process on the CPU through FastEmbed, supports more than 100 languages and needs no GPU and no server. It downloads once from Hugging Face on first use. (#4)
- Versioned retrieval evals. Datasets in `evals/datasets/` stay fixed, and each run folder in `evals/runs/` holds its settings and results. Run one with `make evals <run>`. (#3)
- The `bge-m3` Ollama model, with one index for each embedding model in the database.
- The `--database` option of `ostra index` and `ostra search`, to use another database file for one run.

### Changed

- `potion-multilingual-128m` is the default embedding model. Config files that name `mxbai-embed-large` keep using it and its existing index. (#4)
- The model decides where it runs. The `embedding_provider` config key is ignored. (#4)

### Removed

- The `ostra-daemon` command, its systemd user service and the background indexing question of `ostra init`. Run `ostra index` after you change files or settings. (#6)

  If you installed the service, remove it, because systemd keeps restarting a command that no longer exists. Before you upgrade, run `ostra-daemon uninstall`. After an upgrade, run:

  ```bash
  systemctl --user disable --now ostra-daemon.service
  rm ~/.config/systemd/user/ostra-daemon.service
  systemctl --user daemon-reload
  ```

- The desktop GUI, the `ostra-app` command and the `nicegui` extra. (#2)
- The `ostra eval` command. The versioned evals replace it.
- The Chroma backend and the `database_backend` setting. SQLite is the only store.
