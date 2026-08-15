# Governance Source Extraction

This checkpoint uses AG-ng as the current source of governance doctrine. The older `agent_gov`/classic material was deliberately not used as the authority for the selected kernel.

## Source Repositories

| Repository | Path | Revision | Status | Use |
| --- | --- | --- | --- | --- |
| AG-ng | `/home/jbeck/ag_ng` | `aab771b636d0e7f09b5e281fa2104d94dde7a595` | clean | Runtime/source doctrine for this extraction |
| schmittformer | `/home/jbeck/git/trans` | base `9710f2b4e92fc97624e9829a7e3388c06bad7a5a` (`softmax-bounded-margin-success`) plus uncommitted governance checkpoint files | dirty/uncommitted | Experimental semantic model and tests |

AG-ng does not contain Lean files in this checkout. It contains a formal-calculus crosswalk pinned to public Lean revision `ff491b808ebeab2a132d9ade46d234cf85dcfbe9`, release 14.0.0. That crosswalk is treated as specification evidence and an obligation map, not as runtime authority and not as a proof that this Python model conforms to Lean or to Rust.

## Source Files Inspected

- `/home/jbeck/ag_ng/README.md`
- `/home/jbeck/ag_ng/docs/architecture.md`
- `/home/jbeck/ag_ng/docs/preparation-ratification-kernel.md`
- `/home/jbeck/ag_ng/docs/docket-issuance.md`
- `/home/jbeck/ag_ng/docs/managed-pointer-activation-readiness.md`
- `/home/jbeck/ag_ng/docs/formal-calculus-crosswalk.md`
- `/home/jbeck/ag_ng/docs/source-baseline.md`
- `/home/jbeck/ag_ng/docs/escalation-disposition.md`
- `/home/jbeck/ag_ng/docs/campaign-orchestration-office.md`
- `/home/jbeck/ag_ng/crates/ag-campaign/src/governed.rs`
- `/home/jbeck/ag_ng/crates/ag-campaign/tests/governed_loop.rs`
- `/home/jbeck/ag_ng/crates/ag-kernel/src/authority.rs`
- `/home/jbeck/ag_ng/crates/ag-kernel/src/family.rs`
- `/home/jbeck/ag_ng/crates/ag-store/src/campaign.rs`
- `/home/jbeck/ag_ng/crates/ag-app/src/governed_loop.rs`

## Selected Kernel

The extracted schmittformer kernel is an implementation-neutral finite abstraction of the AG-ng governed-loop law for exact-work occurrences. It is not transformer-specific.

The finite state is:

```text
pc
proposal
preconditions
has_prior
prior_proposal
prior_preconditions
retries_used
probes_used
escalations_used
standing_lease
settlement_outcome
halted_unresolved_attempt
```

The event alphabet has 39 events, covering proposal classes, standing/admission outcomes, lease ticks, one-use authorization burn, Docket custody, settlement/reconciliation, continuation, probe, halt/escalate, human return/termination, completion, no-op, malformed input, and ignored authority-claim records.

The public API is:

```python
transition(state, event) -> TransitionResult
```

where `TransitionResult` contains `next_state`, `output`, `refusal_reason`, and `admitted_action`.

## Concepts

### Exact-work occurrence and program counter

Source repository: `/home/jbeck/ag_ng` at `aab771b636d0e7f09b5e281fa2104d94dde7a595`.

Source files: `crates/ag-campaign/src/governed.rs`, `crates/ag-store/src/campaign.rs`, `crates/ag-app/src/governed_loop.rs`, `README.md`.

Existing semantics: AG-ng has a pure canonical governed-loop law around one exact-work occurrence. The production path records a proposal, requires standing, records admission, consumes one authorization, accepts Docket custody, records settlement or reconciliation, opens continuations, halts, escalates, or completes. The Rust kernel is explicitly pure: external observation, standing, execution, scheduling, and human I/O enter only through boundary records.

Formal/source property: AG-ng tests exercise normal one-shot closure, terminal states, executor-output non-authority, restart/recovery, residual handling, and human-disposition boundaries. The formal crosswalk maps some refusal/family/adapter ideas to the pinned Lean calculus, but this occurrence FSM itself is a Rust/runtime law, not a Lean theorem in this checkout.

Simplification in schmittformer: Exact digests, UUIDs, schemas, signatures, principals, residual sets, and store-event chains are collapsed into small enums and counters. The program-counter lifecycle is preserved.

Deliberately omitted: JCS digest binding, occurrence UUID allocation, persistent SQLite transition chain, residual identities, exact request bytes, concurrency, crash recovery, and external resolver implementations.

### Proposal is not authority

Source files: `README.md`, `docs/architecture.md`, `docs/preparation-ratification-kernel.md`, `crates/ag-campaign/src/governed.rs`, `crates/ag-kernel/src/authority.rs`.

Existing semantics: Workers propose exact work; they do not authorize it. Serializable receipts, envelopes, standings, proposals, and historical records do not deserialize into live authority. `Authority<F>` in `ag-kernel` has no public constructor and is not a serializable cloneable value.

Formal/source property: The formal crosswalk states that Lean theorem evidence and adapter evidence do not confer runtime authority. AG-ng docs repeatedly separate evidence/testimony from authority.

Simplification in schmittformer: `CLAIM_AUTHORITY_RECORD` is a first-class event but always emits `CLAIM_IGNORED` and leaves state unchanged. It cannot create admission, standing, budget, custody, or an action.

Deliberately omitted: Non-serializable Rust types, replay-based authority reconstruction, principal chains, domain/epoch checks, and cryptographic validation.

### Observation adequacy and qualification evidence

Source files: `crates/ag-campaign/src/governed.rs`, `docs/preparation-ratification-kernel.md`, `docs/architecture.md`, `docs/formal-calculus-crosswalk.md`.

Existing semantics: Proposal recording depends on a fresh observation resolver. Observation status can be current, stale, superseded, contradictory, or absent. Admission must use a fresh observation/currentness witness and a normalized relevant-precondition basis. A stale or contradictory observation is a refusal source, not a permission.

Formal/source property: The crosswalk separates formal evidence, adapter evidence, runtime evidence, and authority. It also distinguishes semantic refusal from operational indeterminacy.

Simplification in schmittformer: Precondition basis is `P0` or `P1`. Proposal events encode current observations, while explicit stale/contradictory proposal events refuse. Retry checks require the same normalized precondition basis.

Deliberately omitted: Actual observation resolver calls, subject/scope digests, observation-currentness witnesses, stale-vs-superseded distinction, controlling rejected-review citations, and adequacy theorem details.

### Standing, scope, lease, and expiry

Source files: `crates/ag-campaign/src/governed.rs`, `docs/architecture.md`, `docs/preparation-ratification-kernel.md`, `docs/campaign-orchestration-office.md`.

Existing semantics: Standing is resolved against an exact observation/proposal/subject/scope basis. Standing may be current, absent, revoked, superseded, or expired. Current standing is required before positive admission and is rechecked before authorization spend. Serializable standing evidence is not authority.

Formal/source property: AG-ng tests recheck exact binding before spend and refuse absent/non-current standing. Docs emphasize that historical standing cannot be presented as current standing.

Simplification in schmittformer: Scope is represented by the proposal kind `WORK_A` or `WORK_B`. Live standing is represented by a tiny countdown `standing_lease in {0,1,2}` attached only to `ADMISSIBLE_PENDING_AUTHORIZATION`. `TICK` structurally expires it.

Deliberately omitted: Separate subject/scope digest fields, mandate references, revoked-vs-expired output distinction, resolver clocks beyond the lease counter, principal scope hierarchies, and multi-capability sets.

### Admission and one-use authorization burn

Source files: `crates/ag-campaign/src/governed.rs`, `crates/ag-app/src/governed_loop.rs`, `docs/docket-issuance.md`, `docs/architecture.md`.

Existing semantics: Admission records a positive AG decision but still contains no spent authorization. `consume_authorization` re-resolves premises and creates a durable one-use spend and deterministic issuance. Burn-before-effect is central.

Formal/source property: AG-ng tests cover currentness and binding rechecks before spend and one-shot closure. Docket issuance docs state that refusal emits no issuance and burns nothing.

Simplification in schmittformer: `RECORD_ADMISSIBLE_CURRENT` enters `ADMISSIBLE_PENDING_AUTHORIZATION`; `CONSUME_AUTH_CURRENT` emits `AUTHORIZATION_CONSUMED_A` or `AUTHORIZATION_CONSUMED_B`, sets `admitted_action`, and moves to `AUTHORIZATION_CONSUMED`. Any second burn attempt refuses.

Deliberately omitted: Exact `AgAuthorizationSpendV1`, `AgIssuanceV1`, authorization/spend/issuance digests, durable store transaction, and re-running real resolvers at spend time.

### Docket custody and execution boundary

Source files: `README.md`, `crates/ag-campaign/src/governed.rs`, `crates/ag-app/src/governed_loop.rs`, `crates/ag-store/src/campaign.rs`, `docs/docket-issuance.md`.

Existing semantics: Docket owns execution custody and attempt identity. AG issues only after burn; Docket custody is accepted only after `AuthorizationConsumed`. Executor output is not projected back into a campaign transition.

Formal/source property: AG-ng tests assert that executor output has no transition projection and that custody/settlement evidence must match the exact attempt.

Simplification in schmittformer: `ACCEPT_DOCKET_CUSTODY` is a single event valid only from `AUTHORIZATION_CONSUMED`, moving to `DISPATCHED`.

Deliberately omitted: Docket issuance envelopes, execution-standing references, executor markers, attempt digests, custody port behavior, and external execution mechanics.

### Settlement, reconciliation, and no blind retry

Source files: `crates/ag-campaign/src/governed.rs`, `crates/ag-app/src/governed_loop.rs`, `docs/architecture.md`, `docs/managed-pointer-activation-readiness.md`, `docs/docket-issuance.md`.

Existing semantics: Known success/failure settlement after dispatch moves to an observation-required settlement boundary. An indeterminate outcome moves to `ReconciliationRequired`. Ambiguous outcomes are not success, not failure, and not permission to retry. Continuation is legal only after exact settlement or reconciled settlement.

Formal/source property: AG-ng tests cover unknown outcome requiring reconciliation/halt and never repeat. The architecture docs call automatic retry after an ambiguous boundary forbidden.

Simplification in schmittformer: `REQUIRE_RECONCILIATION` enters `RECONCILIATION_REQUIRED`; `RECORD_RECONCILED_SUCCESS` and `RECORD_RECONCILED_FAILURE` settle. `OPEN_CONTINUATION` from unresolved reconciliation refuses.

Deliberately omitted: Distinguishing exact prestate/poststate/foreign reconciliation classes, receipt identities, target readback, operational pending status, and Docket polling details.

### Retry, probe, escalation, and budgets

Source files: `crates/ag-campaign/src/governed.rs`, `crates/ag-campaign/tests/governed_loop.rs`, `crates/ag-app/src/governed_loop.rs`, `docs/escalation-disposition.md`.

Existing semantics: Loop budgets bound retries, read-only probes, and escalations. Retry requires a distinct occurrence, the same prior proposal, and unchanged normalized preconditions. Successors cannot reuse the prior proposal. Budget exhaustion halts or refuses depending on boundary. AG-ng docs warn that some surfaces have no true higher-authority escalation speech act; campaign escalation is a bounded halt/control path, not a magic authority upgrade.

Formal/source property: AG-ng tests cover retry preconditions, occurrence identity independence, successor reuse refusal, probe budget, and unresolved-attempt blocking.

Simplification in schmittformer: Retry limit is 2, probe limit is 1, escalation limit is 1. Occurrence identity is collapsed to `has_prior` plus prior proposal/preconditions; distinctness is represented by transition class rather than UUIDs.

Deliberately omitted: UUID occurrence allocation, human escalation authority, residual-disposition details, and the difference between AG-ng campaign escalation and the absent generalized escalation doctrine.

### Typed refusal

Source files: `crates/ag-campaign/src/governed.rs`, `crates/ag-store/src/campaign.rs`, `crates/ag-app/src/governed_loop.rs`, `docs/formal-calculus-crosswalk.md`.

Existing semantics: Refusals are typed and attributable; they are not a catch-all for operational indeterminacy. The store records refusal separately from state transition. Invalid requests should produce explicit refusal without manufacturing authority, evidence, budget, or action.

Formal/source property: The crosswalk highlights exact claim-indexed refusal packets and the distinction between semantic refusal and operational indeterminacy.

Simplification in schmittformer: Refusal reasons are finite enums, and every refusal leaves the governance state unchanged. The output vocabulary includes explicit refusal codes.

Deliberately omitted: Refusal digest domains, signed refusal packets, partial native-family refusal retention, and store-level refusal records.

### Halt, human disposition, and completion

Source files: `crates/ag-campaign/src/governed.rs`, `crates/ag-campaign/tests/governed_loop.rs`, `crates/ag-app/src/governed_loop.rs`.

Existing semantics: Halt is effect-free. Human return/termination is blocked if an unresolved attempt remains. Completion is terminal and only legal from an authority-empty observation boundary or safe human termination.

Formal/source property: AG-ng tests cover human dispositions, unresolved-attempt blocking, halted effect-free behavior, and completed terminal behavior.

Simplification in schmittformer: `HALT`, `ESCALATE`, `HUMAN_RETURN`, `HUMAN_TERMINATE`, and `COMPLETE` cover the finite lifecycle. Halt from reconciliation records `halted_unresolved_attempt=1` and blocks human return/termination.

Deliberately omitted: Human principal verification, decision nonce reuse checks, residual discharge authority, exact human disposition records, and store journals.

### State abstraction and successor validation

Source files: `crates/ag-campaign/src/governed.rs`, `crates/ag-store/src/campaign.rs`, `crates/ag-campaign/tests/governed_loop.rs`.

Existing semantics: The store validates exact successor states through the pure kernel. Durable transition kinds form an authoritative chain. Histories matter only through the committed current state and exact retained basis needed for future legal transitions.

Formal/source property: AG-ng tests cover restart mapping, retry/successor validation, and terminal/halted behavior. This is runtime evidence, not a formal bisimulation proof.

Simplification in schmittformer: Complete logical state is a hashable frozen dataclass. Reachable-state enumeration computes the quotient of histories by complete state, and the reference tests check that equivalent histories have identical future outputs for sampled suffixes.

Deliberately omitted: Store digest chains, replay/recovery of exact external evidence, power-loss atomicity, and concurrent single-winner properties.

## Excluded From This First Kernel

- BreakGlass. AG-ng documents it as deliberately absent from runtime despite formal discussion. Adding it here would invent an exceptional authority mechanism rather than extract one.
- Full residual-obligation accounting. The finite model keeps halt/human unresolved-attempt discipline but not exact residual IDs or discharge sets.
- Full family-book crossing from `ag-kernel::family`. It is relevant future doctrine, but the selected kernel tracks one governed-loop occurrence rather than all standing/custody/obligation/capacity books.
- Provider, worker, effectd, Git, filesystem, systemd, and managed-pointer mechanics. They are execution/adapter surfaces, not this first semantic core.
- Durable storage, cryptographic signatures, JCS canonicalization, and crash recovery. These are production obligations for AG-ng, but they are outside this finite reference semantics pass.

## Semantic Core Status

The extracted model is intentionally closer to a reference monitor than a transformer program. That is the point of this checkpoint: shared semantics first, backend choice second.
