# Constellation Use Review

This review asks where a canonical governance semantics could reduce drift across local constellation systems. It does not propose importing the Python research module into production.

## Repositories Inspected

| system | path | revision | status |
| --- | --- | --- | --- |
| AG-ng | `/home/jbeck/ag_ng` | `aab771b` | clean |
| NQ-ng | `/home/jbeck/git/skunkworks/nq-ng` | `59abd3b` | clean |
| NQ | `/home/jbeck/git/nq-root/nq` | `b50d8ae` | clean |
| nq-hatchet | `/home/jbeck/git/nq-root/nq-hatchet` | `772a535` | clean |
| nq-witness | `/home/jbeck/git/nq-root/nq-witness` | `44909b0` | clean |
| guvnah | `/home/jbeck/git/guvnah` | `7462814` | clean |
| gov-webui | `/home/jbeck/git/gov-webui` | `9bea9c2` | clean |
| epistemic_governor | `/home/jbeck/git/gov/epistemic_governor` | `f6d5a92` | clean |
| epistemic_governor_codex | `/home/jbeck/git/gov/epistemic_governor_codex` | `b69cd16` | clean |
| historical skunkworks | `/home/jbeck/git/skunkworks` | `9dcefc7` | dirty | lineage only, not doctrine source |

## Candidate Consumers

| system | current logic location | semantic overlap | risk of divergence | potential value | integration difficulty | shared runtime desirable? | shared spec/tests desirable? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AG-ng `ag-campaign` / `ag-store` / `ag-app` | `crates/ag-campaign/src/governed.rs`, `crates/ag-store/src/campaign.rs`, `crates/ag-app/src/governed_loop.rs` | Exact-work occurrence lifecycle, one-use burn, reconciliation, retry/probe/escalation budgets, typed refusals. | Medium: Rust is canonical, but independent docs/tests can drift from the transition law. | High: conformance vectors and semantic digest can make behavior reviewable across implementations. | Medium: needs Rust adapter/vector runner. | Not Python. Rust remains production runtime. | Yes, strongest near-term use. |
| AG-ng Docket issuance/custody surfaces | `docs/docket-issuance.md`, `crates/ag-app/src/governed_ports.rs`, `crates/ag-app/src/governed_loop.rs` | Burn-before-effect, issuance-is-not-authority, custody acceptance, settlement/reconciliation. | High if custody/settlement evidence is flattened into lifecycle status. | High: a shared conformance corpus can test no blind retry and no issuance on refusal. | Medium/high: Docket wire schemas are owned elsewhere. | No, unless implemented in the owning Rust/Docket code. | Yes. |
| NQ-ng semantic transport/refusal spine | `audit/SEMANTIC_TRANSPORT_MAP.md`, `audit/REFUSAL_PRESERVATION_CROSSWALK.md` | Typed refusal preservation, provider admission, raw custody, replay/idempotency, acknowledgment after commit. | High: the docs explicitly record historical erasure of refusal details. | High: semantic-digest and conformance-vector pattern fits this repo's repaired carrier doctrine. | Medium: domain differs, but the conformance pattern transfers. | Probably not. Domain-specific Rust semantics should remain local. | Yes. |
| NQ root database/read model | `crates/nq-db/src/publish.rs` | Admissibility view, suppression, maintenance expiry/overrun, evidence lineage, ambiguity preservation. | Medium/high: read-side projections can become mistaken for authoritative testimony. | Medium: use shared vocabulary for admissibility/refusal/expiry and test-vector discipline. | Medium: older and domain-specific. | No. | Yes, especially for projection-vs-testimony tests. |
| nq-witness | `SPEC.md`, `profiles/*.md` | Witness standing, coverage, freshness, partial collection, fail closed, evidence is not authority. | Medium: witness reports can be overinterpreted downstream as authority. | Medium: align evidence/standing vocabulary and versioned refusal/admissibility surfaces. | Low/medium: mostly specification-level. | No. | Yes. |
| nq-hatchet | `docs/POSITIONING.md`, candidate docs | Typed, authority-tagged event traces with declared source standing and blind spots. | Medium: projection catalogs can drift from governed transition semantics. | Medium: semantic graph digest/corpus is a useful example of an executable declared-controller catalog. | Medium. | No. | Yes. |
| guvnah | `ARCHITECTURE.md` | Untrusted cockpit, receipts inspector, commit/waive flow, challenge preview. | Medium: UI can flatten refusal/action state for users. | Medium: display semantic version/digest and refusal details from canonical source. | Low/medium. | No. UI must not enforce. | Yes, for presentation conformance. |
| gov-webui | `README.md` | Untrusted web presentation for governor, no signing/keys/scope/commit authority. | Medium: presentation summaries can drift from governor refusal/action meanings. | Medium: consume semantic version and refusal/action vocabulary for UI display. | Low. | No. | Yes. |
| epistemic_governor | `docs/CONSTRAINT_KERNELS.md`, `docs/REFUSALS.md` | Hostile admissibility oracle, typed refusals, boundary-first enforcement, immutable constitutional layer. | Low/medium: currently design-phase and not AG-ng runtime. | Low/medium: conceptual alignment, but not a direct consumer of AG-ng occurrence semantics. | Low for docs, high for runtime. | No. | Maybe, for research comparison only. |

## Packaging Assessment

| option | assessment |
| --- | --- |
| A. Python library | Good for research and quick conformance generation. Poor as production authority boundary for AG-ng/constellation. |
| B. Language-neutral transition tables / generated artifact | Strong near-term option. Deterministic, reviewable, easy to pin by digest, usable from Rust/Python/tests without runtime dependency. |
| C. Small declarative semantic IR with generated backends | Attractive later, but premature. Current fidelity gaps should be resolved before inventing syntax. |
| D. Rust implementation | Best eventual production/runtime backend for AG-ng-style enforcement. Should be derived from or checked against a shared corpus. |
| E. Lean-defined semantics with extracted/generated executable implementation | Valuable long-term if AG-ng wants formal correspondence. Not available from the current checkout and too expensive for this pass. |
| F. Specification + conformance-suite only | Best immediate constellation artifact. Avoids pretending this Python model is runtime authority while still preventing drift. |

Recommendation: use **F plus B** now. Publish a versioned finite semantics document plus `results/governance_conformance.json` and semantic digest. Use it as a conformance suite for Rust, Python, transformer, or other implementations. Do not make the Python module the shared runtime.

A Rust runtime backend becomes desirable only after the fidelity issues in `GOVERNANCE_FIDELITY.md` are intentionally scoped or modeled.

## Minimal Versioning / Provenance Scheme

Every future governance decision or conformance run should be able to name the semantic version it implements:

```json
{
  "governance_semantics": {
    "name": "schmittformer.governance.ag_ng_occurrence_core",
    "version": "0.1.0",
    "source": "ag-ng",
    "source_revision": "aab771b636d0e7f09b5e281fa2104d94dde7a595",
    "graph_schema": "schmittformer.governance.transition_graph.v1",
    "graph_sha256": "1926060d9a20ff1f19d4e67d31c0c0cf4725a11b1a3f401419fd6c033333cf8c",
    "conformance_schema": "schmittformer.governance.conformance.v1"
  }
}
```

Revision rules:

- Any reachable transition output/state change changes the graph digest and requires at least a patch version bump.
- Any event/state schema change requires a minor version bump unless it is purely metadata outside the digest.
- Any semantic reinterpretation of existing labels requires a major or explicitly incompatible version.
- Backends should fail closed if they claim a semantic digest but produce a different conformance result.
- Receipts should record semantic name, version, source revision, and graph digest, not just prose labels like "governance v3".

## Conformance Corpus

Generated artifact:

```text
results/governance_conformance.json
```

Schema summary:

```text
schema: schmittformer.governance.conformance.v1
semantic_name
semantic_version
semantic_digest_sha256
states: reachable state records
cases: one case per reachable state/event transition
```

Each case contains:

```text
case_id
initial_state_id
event / event_id
expected_next_state_id
expected_output / expected_output_id
expected_refusal / expected_refusal_id
expected_admitted_action / expected_admitted_action_id
```

The corpus currently contains 35,568 cases over 912 reachable states. It is meant for backend conformance testing. It does not prove that a backend is correct, but it precisely identifies the finite transition relation it claims to implement.

## Practical Recommendation

Near-term: keep the semantic core as a conformance/specification artifact, not as shared runtime code.

Medium-term: write a small Rust conformance runner against `results/governance_conformance.json` for AG-ng or a narrow extracted crate. Only after that passes should anyone consider a shared runtime kernel.

Do not recommend transformer machinery for production enforcement. A transformer backend may be research-useful; it should not be the constellation authority boundary merely because it can reproduce the finite table.
