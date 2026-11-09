# Phase 3.1 Confidence Verification Report - Five-Candidate Raw-Logit Integration

**Branch:** `member-2-retrieval`  
**Date:** 2026-11-07  
**Owner:** Member 2 - Data, Hybrid RAG & GraphRAG Lead  
**Status:** Phase 3.1 verification complete. No production defect was found. Phase 4 was not started.

---

## 1. Executive Summary

Phase 3.1 verified the integration between the Member 2 hybrid retrieval pipeline and Member 1's evidence confidence gate using five real indexed chunks, real BGE-M3 embeddings, and the real BGE cross-encoder reranker.

Result: **passed**.

| Verification | Result |
| :--- | :--- |
| Real five-candidate live test | `1 passed in 89.78s` |
| Existing confidence integration tests | `2 passed` |
| Member 2 retrieval suite with live PostgreSQL | `44 passed, 1 skipped` |
| Full project suite | `68 passed, 1 skipped, 1 warning` |
| Static compilation | `compile-ok` |
| Production defects found | None |
| Production code changed | None |

The live test confirmed:

- Five real chunks reached the confidence calculation.
- Every final evidence item carried its actual raw cross-encoder logit.
- The confidence gate selected the highest-ranked and fifth-ranked logits correctly.
- The reranking delta was calculated from raw logits, not sigmoid probabilities.
- Reversing the evidence input order produced the same confidence result.
- Workspace isolation remained intact.
- The composite confidence result was `c_evidence=0.95` with `sigma_delta_rerank=1.0`.

---

## 2. Actual Confidence-Gate Contract Discovered from Source Code

The contract below was verified directly in the source, not inferred from the prior report.

### 2.1 `core.schemas.ChunkEvidence`

`ChunkEvidence.cross_encoder_logit` is a required float field documented as:

```text
Raw logit score from cross-encoder (e.g. bge-reranker-v2-m3)
```

There is no probability field. The shared DTO does not transform the value.

### 2.2 `verifier.confidence_gate.compute_evidence_confidence`

For a non-empty evidence list, the gate:

1. Sorts chunks by `cross_encoder_logit` descending.
2. Selects:

```python
top1_logit = sorted_chunks[0].cross_encoder_logit
top5_logit = sorted_chunks[min(4, len(sorted_chunks) - 1)].cross_encoder_logit
```

3. Computes:

```python
delta_l = top1_logit - top5_logit
sigma_rerank = sigmoid(delta_l)
```

4. Builds the Jev context from the first five chunks in the sorted list.
5. Computes:

```text
C_evidence = 0.50 * P_Jev_Noul + 0.30 * sigma(Delta L_rerank) + 0.20 * Cov_entities
```

6. Triggers escalation when the unrounded `C_evidence` is below `policy.tau_escalate`.

Important details:

- With five or more chunks, index `4` is the fifth-ranked chunk.
- With fewer than five chunks, `min(4, len(chunks) - 1)` selects the last available chunk. The gate does not fabricate missing candidates.
- With zero chunks, `delta_l=0.0` and `sigma_rerank=0.0`.
- The final returned `sigma_delta_rerank` and `c_evidence` are rounded to four decimal places.
- The escalation decision uses the unrounded `C_evidence`.

### 2.3 `verifier.confidence_gate.EvidenceSufficiencyGate.evaluate`

`EvidenceSufficiencyGate.evaluate` builds the Jev Noul context from `chunks[:5]` in the caller's input order, then calls `compute_evidence_confidence` with the full list.

Because `RetrievalService.retrieve` returns evidence already sorted by reranker logit descending, the effective Jev context is the top five reranked chunks. The logit-delta calculation inside `compute_evidence_confidence` sorts independently, so it is not dependent on caller order.

### 2.4 Member 2 contract

- `BgeReranker.score_pairs_sync` calls the model with `normalize=False`.
- `BgeReranker.rerank` stores `float(score)` directly as `cross_encoder_logit`.
- `RetrievedChunk.to_chunk_evidence` requires a non-`None` logit and raises `MissingCrossEncoderLogit` otherwise.
- `RetrievedChunk.score_metadata` also exposes `retrieval_cross_encoder_logit` without transformation.
- `RetrievalService.retrieve` reranks and returns at most `top_k` evidence items. With default settings and at least five fused candidates, `top_k=5` returns five evidence items.

---

## 3. Five-Candidate Test Setup

New test:

```text
tests/test_retrieval_confidence_live.py
```

Test name:

```text
test_live_five_candidate_confidence_integration_with_real_models
```

Environment:

| Component | Value |
| :--- | :--- |
| Database | Live ParadeDB PostgreSQL 17.11 |
| pgvector | 0.8.6 |
| pg_search / ParadeDB | 0.26.5 |
| Schema | `0003` |
| Embedding model | Real `BAAI/bge-m3` |
| Reranker model | Real `BAAI/bge-reranker-v2-m3` |
| Device | CPU |
| Mocked models | None in this test |

Isolated workspaces:

- Workspace A: five distinct chunks.
- Workspace B: one distinct chunk with the same query term, used for isolation verification.

Workspace A contents:

1. `EV-Pack-V2 maximum defect tolerance is 0.5 percent for lithium cell batches.`
2. `EV-Pack-V2 inspection records are retained for five years.`
3. `EV-Pack-V2 firmware version 4.2 requires a controller reset after installation.`
4. `EV-Pack-V2 shipments use moisture-barrier packaging.`
5. `EV-Pack-V2 supplier audits occur every quarter.`

Workspace B content:

```text
EV-Pack-V2 tenant B confidential note: defect tolerance is 9.9 percent.
```

Query:

```text
What is the EV-Pack-V2 maximum defect tolerance?
```

The test indexed all chunks into the live database, ran `RetrievalService.retrieve(..., top_k=5)` with real models, and passed the resulting `ChunkEvidence` list to `compute_evidence_confidence`.

---

## 4. Actual Raw Logits from All Five Candidates

Actual test output, in final reranked order:

| Rerank rank | Chunk | Content summary | Raw cross-encoder logit |
| :--- | :--- | :--- | :--- |
| 1 | `...:c00000` | Maximum defect tolerance is 0.5 percent | `8.323678` |
| 2 | `...:c00001` | Inspection records retained for five years | `-0.382794` |
| 3 | `...:c00003` | Moisture-barrier packaging | `-3.603393` |
| 4 | `...:c00002` | Firmware 4.2 controller reset | `-3.812089` |
| 5 | `...:c00004` | Quarterly supplier audits | `-4.426744` |

Actual logit vector:

```text
[8.323678, -0.382794, -3.603393, -3.812089, -4.426744]
```

The negative values and values greater than 1 confirm these are raw cross-encoder logits, not sigmoid probabilities.

The test also independently called `reranker.score_pairs_sync` on the same five passages and asserted that its raw scores matched the final `ChunkEvidence.cross_encoder_logit` values within `rtol=1e-4, atol=1e-5`. It also asserted that the raw scores were not equal to `sigmoid(raw_score)`.

---

## 5. Highest-Ranked and Fifth-Ranked Candidate Selection

Actual selection from the five candidates:

| Selection | Value |
| :--- | :--- |
| Highest-ranked logit | `8.323678` |
| Fifth-ranked logit | `-4.426744` |
| Delta | `12.750422` |
| Unrounded sigmoid | approximately `0.999997` |
| Rounded `sigma_delta_rerank` | `1.0` |

The test verified selection in two ways:

1. It computed `sorted_logits = sorted(logits, reverse=True)` and asserted that the service output was already in that order.
2. It called `compute_evidence_confidence` again with the evidence list reversed. The reversed input produced the same `sigma_delta_rerank`, `c_evidence`, and `cov_entities`, proving that the gate selects by sorted logit rather than by input position.

With exactly five candidates, `min(4, len(chunks) - 1)` resolves to index `4`, which is the fifth-ranked candidate.

---

## 6. Actual Confidence Calculation Result

Inputs:

```text
P_Jev_Noul(Sufficient) = 0.9
sigma_delta_rerank = 1.0 (rounded; unrounded approximately 0.999997)
Cov_entities = 1.0
policy = balanced
```

Calculation:

```text
C_evidence = 0.50 * 0.9 + 0.30 * 0.999997 + 0.20 * 1.0
           = 0.45 + 0.299999 + 0.20
           = 0.949999
           -> rounded to 0.95
```

Actual result:

```text
c_evidence = 0.95
p_jev_noul_sufficient = 0.9
sigma_delta_rerank = 1.0
cov_entities = 1.0
escalate_triggered = False
policy_mode = balanced
tau_escalate = 0.65
```

The test asserted:

- `sigma_delta_rerank` equals `round(sigmoid(top1 - fifth), 4)`.
- `cov_entities` equals the independently computed entity coverage.
- `c_evidence` equals the independently computed composite formula.
- `p_jev_noul_sufficient` equals the supplied `0.9`.
- The escalation decision matches the unrounded composite compared with `tau_escalate`.

---

## 7. Workspace Isolation Results

The live test created two isolated workspaces and indexed both.

| Workspace | Chunks indexed | Query result |
| :--- | :--- | :--- |
| Workspace A | 5 | Exactly 5 workspace A chunks |
| Workspace B | 1 | Exactly 1 workspace B chunk |

Verified:

- Every evidence item returned for workspace A belonged to workspace A's document.
- No workspace B chunk appeared in workspace A results.
- Workspace B returned only its own chunk.
- Vector retrieval, BM25 retrieval, RRF, and reranking all operated under the workspace-scoped SQL predicates.
- Cleanup removed both test workspaces after the test.

Actual workspace A reranked chunk IDs ended with:

```text
c00000, c00001, c00003, c00002, c00004
```

Actual workspace B chunk ID ended with:

```text
c00000
```

The two workspaces used distinct document UUIDs.

---

## 8. Files Created or Modified

Created:

- `tests/test_retrieval_confidence_live.py` - new opt-in live five-candidate test using real PostgreSQL, real BGE-M3, and real BGE reranker.
- `retrieval/reports/PHASE_3_1_CONFIDENCE_VERIFICATION_REPORT.md` - this report.

Modified:

- `tests/test_retrieval_confidence_integration.py` - added an explicit mocked regression test for the fewer-than-five fallback behavior.

Production code:

- No production code was modified.
- `verifier/confidence_gate.py` was not modified.
- `core/schemas.py` was not modified.
- No Member 3 or Member 4 files were modified.
- No Phase 3 retrieval implementation was rewritten.

The new live test is opt-in because it requires the live database and real models:

```powershell
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:5432/calibrag'
$env:RETRIEVAL_RUN_REAL_MODELS='1'
python -m pytest tests/test_retrieval_confidence_live.py -q -s
```

---

## 9. Commands Executed and Actual Results

### 9.1 New live five-candidate test with real models

Command:

```powershell
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:5432/calibrag'
$env:RETRIEVAL_RUN_REAL_MODELS='1'
python -m pytest tests/test_retrieval_confidence_live.py -q -s
```

Actual result:

```text
five_candidate_logits= [8.323678, -0.382794, -3.603393, -3.812089, -4.426744]
top1= 8.323678 fifth= -4.426744 delta= 12.750422
sigma_delta_rerank= 1.0 c_evidence= 0.95
real_model_confidence_integration=ok
1 passed in 89.78s (0:01:29)
```

### 9.2 Existing confidence integration tests

Command:

```powershell
python -m pytest tests/test_retrieval_confidence_integration.py -q
```

Actual result:

```text
2 passed in 1.60s
```

### 9.3 Member 2 retrieval suite with live PostgreSQL

Command:

```powershell
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:5432/calibrag'
python -m pytest tests/test_retrieval_db.py tests/test_retrieval_ingestion.py tests/test_retrieval_workspace_isolation.py tests/test_retrieval_member2.py tests/test_retrieval_embeddings.py tests/test_retrieval_pgvector.py tests/test_retrieval_bm25.py tests/test_retrieval_hybrid.py tests/test_retrieval_reranker.py tests/test_retrieval_confidence_integration.py tests/test_retrieval_live_integration.py tests/test_retrieval_confidence_live.py -q
```

Actual result:

```text
44 passed, 1 skipped in 23.55s
```

The skipped test was the real-model live test because `RETRIEVAL_RUN_REAL_MODELS` was not set for this suite run. That test was executed separately in section 9.1.

### 9.4 Full project suite

Command:

```powershell
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:5432/calibrag'
python -m pytest -q
```

Actual result:

```text
68 passed, 1 skipped, 1 warning in 18.70s
```

The warning is the existing unrelated Starlette deprecation notice from `fastapi.testclient` about `httpx`. The skipped test is the opt-in real-model live test.

### 9.5 Static compilation

Command:

```powershell
python -m compileall -q retrieval tests/test_retrieval_confidence_integration.py tests/test_retrieval_confidence_live.py
```

Actual result:

```text
compile-ok
```

---

## 10. Mocked Versus Real Verification

| Verification | Database | Embeddings | Reranker | Classification |
| :--- | :--- | :--- | :--- | :--- |
| `test_retrieval_confidence_live.py` | Real PostgreSQL | Real BGE-M3 | Real BGE reranker | Fully real |
| `test_retrieval_confidence_integration.py` | None | Constructed evidence | Constructed logits | Mocked unit test |
| `test_retrieval_live_integration.py` | Real PostgreSQL | Deterministic fake | Fake | Live DB with mocked models |
| Existing Phase 3 unit tests | None or fake DB | Fake | Fake | Mocked unit tests |
| Member 2 ingestion/database tests | Real or fake as designed | Not used | Not used | Mixed per test |

No mocked score was presented as a real model output. The real-model live test printed the actual logits shown in section 4.

---

## 11. Remaining Limitations

- The five-candidate verification used one query and five short passages. It did not benchmark a large corpus.
- The test ran on CPU. GPU behavior and latency were not measured.
- The confidence gate's fewer-than-five behavior was verified with a constructed unit test. In production, the service returns fewer than five evidence items when fewer than five candidates survive fusion. The gate then uses the last available item as the fifth-ranked item; it does not fabricate missing candidates.
- `EvidenceSufficiencyGate.evaluate` uses `chunks[:5]` in caller order for the Jev Noul context. The service returns evidence sorted by reranker logit, so the effective context is top-five. A caller that supplies unsorted evidence could give Jev a different context even though the logit delta remains sorted. This is Member 1 behavior and was not modified.
- Real-model timing includes model loading and CPU inference; it is not a production latency claim.
- Hugging Face cache symlink warnings may appear on Windows. They did not affect model loading or inference.
- No commit or push was performed.

---

## 12. Git Branch and Status

- Branch: `member-2-retrieval`
- No branch switch, merge, rebase, commit, or push was performed.
- Production files modified in Phase 3.1: none.
- Test files added/modified: `tests/test_retrieval_confidence_live.py` and `tests/test_retrieval_confidence_integration.py`.
- Existing tracked modifications from earlier phases remain: `retrieval/__init__.py` and `retrieval/docling_parser.py`.
- Existing untracked Phase 2/3 files and `myenv/` remain untracked.
- No shared schema or Member 1 file was changed.

---

## 13. Phase 3 Readiness for Phase 4

Phase 3.1 found no genuine defect in the retrieval-to-confidence integration.

Verified readiness:

- Five real candidates reached the confidence calculation.
- Raw logits were preserved through reranking, `ChunkEvidence`, and the confidence gate.
- Top-1 and fifth-ranked selection matched the source contract.
- The confidence delta and composite score matched independent calculation.
- No sigmoid conversion or fabricated score was used.
- Workspace isolation remained intact.
- Existing and new tests passed.

Phase 3 is ready for Phase 4 review. Phase 4 was not started. Await explicit approval before beginning Phase 4 or LightRAG/GraphRAG work.
