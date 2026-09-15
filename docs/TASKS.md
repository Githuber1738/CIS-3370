# Task Delegation Log

## Scope decision (2026-09-15, revised same day)

Originally started with NSL-KDD only. **Revised per professor's guidance:**
NSL-KDD is the dataset already covered in class, so the professor wants the
team to work a different, more challenging dataset instead. Primary target is
now **CIC-DoS2017**. NSL-KDD ingestion code already written stays in the repo
(useful reference / possible comparison baseline) but is no longer the
critical path. Kitsune remains an open stretch goal, undecided.

## Roles

### Student 1 — Data & System Architect
Owns dataset acquisition, directory layout, and environment setup.

Immediate tasks:
- [x] Run `scripts/download_nsl_kdd.sh` and confirm all four files land in `data/nsl_kdd/`
- [ ] Confirm `python -m venv .venv && pip install -r requirements.txt` works clean on a teammate's machine

Stretch goals (defer until NSL-KDD reproduction is working):
- [ ] Decide which Kitsune attack scenario to target (see `scripts/download_kitsune.sh` header — check the xNIDS repo at github.com/CactiLab/code-xNIDS for which one they used) and run the download
- [ ] Get a Kaggle API token and run `scripts/download_cic_dos2017.sh`, or email UNB CIC for direct access if the mirror is unsuitable

### Student 2 — Backend / Logic Developer
Owns ingestion, preprocessing, and baseline model/query code.

Immediate tasks:
- [ ] Write `src/ingestion/` loaders for each dataset (NSL-KDD is plain-text CSV-like with a 42-column schema + an undocumented `difficulty` 43rd column — watch for that)
- [ ] Write `src/preprocessing/` for encoding, scaling, class-imbalance handling
- [ ] Stand up a baseline model per target dataset once data is in place

### Student 3 — Documentation & Project Manager / Analyst
Owns `README.md`, task tracking, dependency management, and course alignment.

Immediate tasks:
- [ ] Keep this file current as tasks move
- [ ] Keep `requirements.txt` in sync with what's actually imported
- [ ] Cross-check deliverables against the course's capstone rubric (paper selection/proposal, reproduction, final presentation, GitHub repo + project board, Wiki report)
- [ ] Track the team's chosen commit approach (Individual vs. Delegated Commits) — this needs to be declared in the proposal per the course's GitHub requirements

## Status Log

| Date | Change |
|------|--------|
| 2026-09-15 | Project scaffold created: directory structure, download scripts, config, requirements |
| 2026-09-15 | NSL-KDD data downloaded and verified (all 4 files + ARFF zip) |
| 2026-09-15 | Scope decision: NSL-KDD only for now; Kitsune/CIC-DoS2017 deferred as stretch goals |
| 2026-09-15 | Scope revised per professor: switching primary target to CIC-DoS2017 (NSL-KDD is already covered in class) |
| 2026-09-15 | CIC-DoS2017 downloaded via Kaggle mirror (dhoogla/cicdos2017) and ingestion loader written/verified |
