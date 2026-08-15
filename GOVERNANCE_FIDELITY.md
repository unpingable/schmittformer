# Governance Fidelity Audit

Audit source: `/home/jbeck/ag_ng` at `aab771b636d0e7f09b5e281fa2104d94dde7a595`, clean.

The current schmittformer kernel is a finite abstraction of AG-ng's governed-loop exact-work occurrence lifecycle. It is not a complete AG-ng implementation and should not be described as one.

## Overall Verdict

The extraction preserves the main AG-ng occurrence-lifecycle discipline:

- proposal records do not create authority;
- current observation/standing/admission must precede a one-use burn;
- Docket custody follows burn;
- known settlement differs from indeterminate outcome;
- indeterminate outcome blocks blind retry;
- retry is bounded and must preserve prior proposal/preconditions;
- refusals are explicit and non-mutating;
- histories quotient through complete logical state.

The extraction flattens or omits several AG-ng doctrines that matter for production: exact digest/provenance binding, authenticated principals, real custody/settlement evidence, family-book authority reconstruction, residual obligations, human disposition detail, and scope/capability binding. These omissions do not invalidate the finite occurrence-lifecycle result, but they prevent the current kernel from being runtime-ready shared governance infrastructure.

## Classification Table

| Schmittformer concept | AG-ng source concept | source file / symbol | classification | semantic correspondence | differences | safety significance | recommended action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Program counter lifecycle | `ProgramCounterV1`, `GovernedLoopKernelV1` | `crates/ag-campaign/src/governed.rs::{ProgramCounterV1, GovernedLoopKernelV1}` | FAITHFUL_FINITE_REDUCTION | Preserves observation-required, proposal-recorded, standing-required, admissible-pending, authorization-consumed, dispatched, reconciliation-required, settled, halted, completed. | Collapses exact snapshots, digests, metadata, and durable journals. | Good for lifecycle conformance, not persistence integrity. | Keep as core finite quotient; document as occurrence lifecycle only. |
| Authority | One-use AG spend and non-serializable authority doctrine | `consume_authorization`, `AgAuthorizationSpendV1`, `Authority<F>` | FAITHFUL_FINITE_REDUCTION for occurrence spend; MISSING for family authority | Preserves one-use burn and proposal-is-not-authority. `CLAIM_AUTHORITY_RECORD` cannot create authority. | Does not model sealed `Authority<F>`, replay against book heads, principal chains, domains, epochs, or signatures. | Safe inside symbolic model; unsafe as runtime boundary if event encoder can forge admitted events. | Treat `CONSUME_AUTH_CURRENT` as a post-validation boundary event. Add family-authority semantics only in a separate model. |
| Capability/scope | Proposal subject/scope, standing scope, effect family crossing | `ExactWorkProposalV1::{subject, scope}`, `resolve_standing`, `ag-kernel::family` | QUESTIONABLE | Distinguishes `WORK_A` vs `WORK_B` as finite action kinds. | No separate authority scope/capability set; no out-of-scope standing refusal except generic inadmissibility; no standing/custody/obligation/capacity crossing. | A real scope mismatch can be safety-equivalent refusal but operationally different. A forged `RECORD_ADMISSIBLE_CURRENT` would bypass scope in this abstraction. | Add explicit finite scope/capability states before claiming shared authority semantics. |
| Qualification | Observation currentness plus admissibility decision | `ObservationStatusV1`, `AdmissionDispositionV1`, `resolve_admissibility` | INTENTIONAL_SIMPLIFICATION | Models current, stale, contradictory, inadmissible, and retry precondition basis. | Omits superseded/absent distinction, C1 repair citations, altered finding sets, exact policy basis, and resolver evidence. | Preserves no-action-on-bad-evidence for represented cases; loses diagnostic specificity and some replay constraints. | Good enough for lifecycle tests; add finite evidence-record IDs if used as conformance for admission boundaries. |
| Observation/evidence adequacy | Fresh observation resolver and normalized preconditions | `resolve_observation`, `ObservationResolutionV1` | FAITHFUL_FINITE_REDUCTION | Retry cannot proceed if normalized preconditions change; stale/contradictory observations refuse. | Proposal event itself implies fresh observation; claims about evidence are not separately represented. | Boundary assumption is load-bearing. A runtime must reject unsupported evidence before emitting current-observation events. | Add explicit `CLAIM_EVIDENCE` vs `ADMIT_EVIDENCE` if this becomes a shared IR. |
| Lease/expiry | `fresh_until_unix_ms`, `expires_at_unix_ms`, consequence-time rechecks | `resolve_observation`, `resolve_standing`, `consume_authorization` | INTENTIONAL_SIMPLIFICATION | Tiny `standing_lease` structurally expires before burn. | Time is a small countdown; observation freshness and standing expiry are not separate clocks. | Captures expiry-before-spend but not real clock/basis replay behavior. | Preserve as finite test knob; do not map one-to-one to AG-ng wall-clock semantics. |
| Budget/action ceiling | `LoopBudgetV1` retry/probe/escalation | `LoopBudgetV1`, `record_proposal`, `note_probe`, `escalate` | FAITHFUL_FINITE_REDUCTION | Preserves retry/probe/escalation ceilings and nonnegative counters. | Does not implement a general action budget; AG-ng occurrence law does not have one beyond one-use burn and retry/probe/escalation budgets. | Good for AG-ng loop budgets; do not call it a generic action ceiling. | Rename docs toward loop budgets when integrating. |
| Refusal | `KernelErrorV1`, `RefusalCodeV1`, store refusal records | `KernelErrorV1`, `RefusalOutcomeV1`, `CampaignStoreV1::record_refusal` | INTENTIONAL_SIMPLIFICATION | Every invalid request emits typed refusal and leaves state unchanged. | Python reasons are closer to kernel errors than durable `RefusalCodeV1`; evidence digests and store records are omitted. | Safety mostly preserved; operator/audit UX can diverge because reason precedence and evidence identity differ. | Build a refusal mapping table before cross-system use. |
| Action admission | Positive admission then one-use burn then issuance | `record_admissible`, `consume_authorization`, `AgIssuanceV1` | FAITHFUL_FINITE_REDUCTION | `admitted_action` appears only on `CONSUME_AUTH_CURRENT` from live admissible-pending state. | Issuance body/digest/mandate/spend are not represented. | Strong for language-of-actions checks; not enough for authentic issuance. | Use conformance corpus for transition shape; use AG-ng for real issuance. |
| Ambiguous outcome | Indeterminate Docket outcome and reconciliation | `require_reconciliation`, `recover_dispatched`, `IndeterminateOutcomeV1` | FAITHFUL_FINITE_REDUCTION | Distinguishes known settlement from reconciliation-required and blocks continuation while unresolved. | Omits exact indeterminate evidence and recovery-generated reconciliation record. | Preserves the no-blind-retry doctrine. | Keep. Add evidence IDs only if testing Docket correspondence. |
| Settlement | Known Docket settlement or reconciled settlement | `record_settlement`, `record_reconciled_settlement`, `DocketSettlementV1` | INTENTIONAL_SIMPLIFICATION | Success/failure outcomes move to settled-observation-required and allow continuation. | No issuance/attempt/executor-marker binding, no receipt digest. | Lifecycle is preserved; wrong-settlement refusal is not representable. | Add invalid settlement/custody events before runtime conformance claims. |
| Retry semantics | `ProposalClassV1::Retry`, `PriorOccurrenceBasisV1` | `record_proposal`, `open_continuation` | FAITHFUL_FINITE_REDUCTION | Retry must match prior proposal and normalized preconditions; retry budget increments. | Occurrence UUID distinctness is collapsed into `has_prior`; occurrence-reused is mostly unrepresented. | Good for abstract retry discipline; not for identity replay attacks. | Add occurrence identity if future shared corpus tests restart/replay. |
| Custody | Docket custody acceptance and attempt identity | `accept_docket_custody`, `validate_custody`, `DocketCustodyV1` | MISSING except lifecycle gate | `ACCEPT_DOCKET_CUSTODY` only legal after burn. | Wrong schema, substituted standing, wrong attempt, and executor-marker substitution are not represented. | Real fidelity gap for production. Current model assumes event encoder validates custody first. | Add negative custody events or exclude custody authenticity from kernel claims. |
| Authentication/provenance | Signed local RPC, principal chains, exact request evidence | `crates/ag-app/src/rpc_auth.rs`, `docs/signed-local-rpc.md`, `docs/architecture.md` | MISSING | None beyond finite event labels. | No signatures, nonce replay, principal/audience/epoch/domain. | Major runtime boundary; not needed for finite lifecycle but needed for constellation authority. | Keep outside this model or create a separate authentication/custody semantics. |
| Receipts/state digests | JCS canonical documents and state digest chain | `state_digest`, `CampaignTransitionKindV1`, `CampaignStoreV1` | MISSING | Conformance corpus has a semantic digest, not AG-ng receipts. | No receipt bytes, state digest predecessor chain, store atomicity, rollback handling. | Not runtime-ready; cannot audit actual committed evidence. | Use corpus as tests, not receipts. |
| Residual obligations | `ResidualSetV1`, human residual disposition | `ResidualObligationV1`, `ExactResidualDischargeV1`, `apply_human_disposition` | MISSING | Only unresolved attempt halt flag is modeled. | Open residuals blocking completion and exact residual discharge are omitted. | Real fidelity defect if residuals are in scope; changes admissibility of termination. | Exclude residual doctrine from v0; add if shared infrastructure needs halt/human paths. |
| BreakGlass | Deliberately absent runtime path | `docs/escalation-disposition.md`, `docs/formal-calculus-crosswalk.md` | EXACT absence | Kernel does not include break-glass. | None for current AG-ng runtime. | Correct omission; adding it would invent doctrine. | Keep absent until AG-ng has reviewed runtime semantics. |

## Semantic Flattening Findings

| suspicious pair | AG-ng distinguishes? | current kernel distinguishes? | trace / scenario | can collapse change admissibility? | verdict |
| --- | --- | --- | --- | --- | --- |
| proposal claim vs admitted evidence | Yes: proposal, observation resolution, standing, and admission are separate records. | Partly: proposal and admission are separate states, but a `RECORD_ADMISSIBLE_CURRENT` event is trusted. | `PROPOSE_INITIAL_A_P0 -> REQUIRE_STANDING -> RECORD_ADMISSIBLE_CURRENT -> CONSUME_AUTH_CURRENT`. | Yes if an untrusted source can emit `RECORD_ADMISSIBLE_CURRENT`; no within the symbolic boundary assumption. | Boundary-assumption risk. |
| signed/authenticated request vs plain event | Yes: signed local RPC and principal chain evidence are separate from authority. | No. | Any event sequence can be supplied directly to `transition`. | Yes for runtime use; no for finite transition testing. | Missing runtime doctrine. |
| authority absent vs out-of-scope authority | Yes: standing resolution binds exact subject/scope; family crossings keep books separate. | Weakly: absent standing is distinct, but out-of-scope is generic inadmissible/non-current. | `... REQUIRE_STANDING -> RECORD_ADMISSIBLE_INADMISSIBLE`. | Action remains refused, but reason/provenance changes. | Operational flattening; add scope state before shared runtime use. |
| expired vs revoked vs superseded standing | AG-ng maps all to `StandingNotCurrent` after absent is separated. | Yes, same collapsed refusal output. | `RECORD_ADMISSIBLE_REVOKED_STANDING` and `RECORD_ADMISSIBLE_EXPIRED_STANDING`. | No for action admission; reason is already collapsed in AG-ng durable code. | Acceptable flattening. |
| stale vs absent vs superseded observation | AG-ng maps stale/superseded/absent to observation-not-current; contradictory is distinct. | Partly: stale and contradictory only. | `PROPOSE_STALE_OBSERVATION` refuses; absent/superseded are not separate. | No for admission, possible for operator diagnosis. | Intentional simplification. |
| failure vs ambiguous outcome | Yes. | Yes. | `RECORD_SETTLEMENT_FAILURE` reaches settled; `REQUIRE_RECONCILIATION` blocks continuation. | No. | Preserved. |
| executor success vs Docket settlement | Yes: executor output has no AG transition authority. | Yes by omission: only settlement events affect state. | Direct executor-output event does not exist. | No within model; event vocabulary prevents projection. | Preserved. |
| wrong custody vs valid custody | Yes: `validate_custody` checks issuance, spend, standing substitution, canonical attempt, marker substitution. | No. | In AG-ng, wrong custody after burn refuses; in kernel, `ACCEPT_DOCKET_CUSTODY` after burn always advances. | Yes if event means raw custody, no if event means already-validated custody. | Real fidelity gap for custody authenticity. |
| wrong settlement evidence vs valid settlement | Yes: `validate_settlement` checks schema, issuance, attempt, marker. | No. | In AG-ng, wrong settlement after dispatch refuses; in kernel, success/failure settlement always advances. | Yes under raw settlement input. | Real fidelity gap for settlement evidence. |
| residuals open vs no residuals | Yes: residuals block completion/human termination until exact discharge. | No, except unresolved attempt flag. | Halt with residuals then human terminate. | Yes. | Out-of-scope but significant. |
| refusal vs no-op | Yes. | Yes. | `NOOP` emits `NO_OUTPUT`; invalid operations emit refusal outputs. | No. | Preserved. |
| settled failure vs unresolved ambiguity | Yes. | Yes. | Settlement failure permits continuation; reconciliation-required refuses continuation. | No. | Preserved. |

## Refusal Precedence

AG-ng refusal precedence is mostly procedural rather than a separately declared lattice:

1. Wrong program counter returns `IllegalTransition` before boundary-specific checks.
2. Exact schema/binding/currentness checks run inside the relevant resolver/validator.
3. `resolve_admissibility` checks observation, normalized preconditions, standing, C1 repair basis, and only then the admission decision.
4. Retry proposal recording checks class/binding, occurrence reuse, retry budget, proposal equality, and precondition equality in code order.
5. Durable `RefusalCodeV1` is coarser than `KernelErrorV1`; evidence identities may carry the detail.

The schmittformer kernel follows the same broad operation-first shape, but it sometimes chooses a refusal reason by event label rather than deriving it from simultaneous failing evidence. Multi-gate examples such as expired standing plus inadmissible work are not real simultaneous facts in the finite model; the event says which boundary result was observed. Therefore refusal reason selection is a v0 design choice where the model omits evidence fields.

Safety-equivalent refusal is preserved more strongly than exact operational reason selection. Before using this as shared constellation infrastructure, build an explicit refusal-precedence and error-code mapping against AG-ng's `KernelErrorV1`, `RefusalCodeV1`, and store refusal records.

## Differential Test Decision

I did not add `tests/test_governance_source_correspondence.py`. AG-ng does expose executable Rust transition functions, but constructing the same scenarios requires exact digests, resolvers, custody records, settlements, and occurrence IDs. A thin Python test that merely restates expectations would not be an honest differential harness.

Instead, I ran AG-ng's own governed-loop test target:

```text
cargo test -p ag-campaign --test governed_loop
12 passed
```

The source-derived correspondence is documented here. A future honest differential test should either call a small Rust conformance adapter or consume generated AG-ng-side vectors.
