# Talkative Infrastructure

A specification for an agent-mediated coordination layer over existing
infrastructure. Written to be implemented from, and to be reviewed and
revised by engineers. Not a pitch.

---

## Problem

A non-technical user — a kid building a website, a business analyst
shipping a regulatory report, a product manager launching a feature —
has a project that requires coordinated changes across many
infrastructure components. Each component is owned by a different
team, has its own change process, and describes state in its own
vocabulary.

Today, the user pays the coordination cost. They open twelve tickets,
re-explain themselves twelve times, translate between teams, and hold
the cross-component picture in their head. The work is mostly
coordination, not engineering. The record of what was agreed lives in
Slack and in the requester's head.

This system moves that coordination cost into the substrate.

## The pattern: HRM for agents

The analogy is CRM, inverted.

A CRM exists so that any salesperson who picks up the phone already
knows what the previous five said to the customer. The customer never
re-explains. Continuity is the product; the data model is incidental.
In CRM, **one company faces many customers** and needs continuity
*toward* the customer.

We need the inverse. **One human faces many agents** and needs
continuity *from* the agents. The human is the constant; the agents
rotate. We call this Human Relationship Management — HRM.

The operational consequence: when a user approaches any agent, that
agent must be able to rehydrate the full relevant context from two
keys — the user's identity and the project they are working on — and
greet the user already oriented. The user says:

> "Hi, I'm XYZ. I've been working on project 123."

The agent looks up `(user=XYZ, project=123)` and has the full thread
of every conversation across every agent involved in that project,
plus the user's prior history. It does not ask the user to
re-explain.

## Architectural primitives

There are three primary objects and two protocols. Everything else is
downstream.

### Objects

**Human.** A stable identity for the requester. Carries history of
every interaction across every agent. Equivalent to a CRM Account.

**Project.** The unit of cross-component work. The equivalent of a
CRM Opportunity. Has a state machine, an owner, a current stage, a
log of every agent that has touched it, and the verbatim conversation
record at each touch.

**Agent.** A scoped, accountable interface to one infrastructure
component. Has a defined scope (what it owns), a defined authority
(what it can decide alone, what it must escalate), and a defined tool
surface (what it can do). It is not a chatbot — it can run queries,
apply config, roll back deployments, file tickets against itself.

The schema's primary record is the **interaction**: a tuple of
`(human, project, agent, timestamp, content, action_taken)`. Every
other view — "what has user XYZ done", "what is the state of project
123 across all agents", "what has the schema agent done this week" —
is a query over this table.

### Protocols

**Handoff.** When a request crosses an agent's scope boundary, the
agent produces a structured summary — who the requester is, what was
asked, what was agreed so far, what the blocker is, what the next
agent is being asked to do — and routes the user to the next agent.
The summary is filed against the project. The receiving agent reads
the summary plus the full project history before responding. The
chain of handoffs for any project is reconstructable.

**Coordination.** When a request requires simultaneous changes across
multiple components, the affected agents hold a structured exchange:

1. Originating agent states the request.
2. Each affected agent states its position, constraints, objections.
3. A facilitator agent compiles positions into a proposal.
4. Agents accept or dissent; dissents are recorded with reasons.
5. On consensus, the proposal becomes a plan decomposed into
   per-component tickets, each linked to the project.
6. On no consensus, the full exchange is escalated to a named human
   owner.

This is a defined protocol with a structured output (plan or
escalation), not an open conversation.

## Memory substrate

The pattern collapses if the `(human, project) → context` lookup is
slow or lossy. Memory is therefore load-bearing, not a side concern.

Requirements:

- **Verbatim.** No summarisation, no lossy compression of the actual
  conversation. An agent reading three months later sees what was
  said, not a paraphrase. Summaries exist as derived artefacts
  alongside the verbatim record, never replacing it.
- **Partitioned.** By human, by project, by agent, by component.
  Queries can be scoped to any combination.
- **Queryable two ways.** Semantic search (vector) for "what was
  discussed about X" and structured query for "all interactions on
  project 123 in the last 7 days."
- **Write-ahead-logged.** Writes that fail at the backend are
  persisted locally and replayed. No interaction is ever silently
  lost. This is non-negotiable — a memory system that loses writes
  destroys the trust that makes the pattern work.
- **Concurrent.** Multiple agent processes write at the same time
  without coordination overhead.
- **Fast.** Sub-200ms semantic search at the scale of hundreds of
  thousands of interactions. The user is on the other end of a chat;
  every additional second of agent latency is a re-explanation
  averted or paid for.

The current implementation is SurrealDB-backed and meets the
concurrency, WAL, and latency requirements at current scale. Scaling
beyond a single node and beyond a million-interaction estate is open
work.

## Project state model

A project is not a free-form thread. It has explicit state that any
agent can read and advance.

Minimum fields:

- `id`, `human_owner`, `created_at`
- `stage` — an enum agreed across agents (e.g. `scoping`,
  `in_coordination`, `executing`, `blocked`, `done`,
  `escalated`)
- `current_holder` — which agent is presently expected to act
- `next_action_owed_by` — a deadline; if exceeded, the project
  surfaces in an escalation queue
- `linked_tickets` — outputs of the coordination protocol; each
  per-component ticket back-links to the project
- `interactions` — the verbatim log

The CRM analogy holds: stage transitions are the equivalent of pipeline
stages, `current_holder` is "next action owed by," and the linked
tickets are the line items on the deal.

## Audit

Every agent action — query, config change, handoff, coordination
exchange, ticket filed, deployment — emits a structured record linking
the action to the originating project and to the interaction that
authorised it.

The audit layer's job is to answer, in one query: *what was changed
in service of project 123, by which agent, authorised by which
interaction, executed against which system?*

This subsumes compliance evidence, incident reconstruction,
new-engineer onboarding, and the "what did we agree two months ago"
question that currently requires Slack archaeology.

Current state: per-agent action logging is consistent. Linking from
those logs to commits and operational changes (e.g. via deployment
metadata, git trailers, change-management IDs) is per-component and
not yet unified. Unifying it is a near-term task and is straightforward
once the project ID is threaded through every action.

## Component layout

Five runtime pieces:

1. **Component agents** — one process per infrastructure component.
   Built on the OpenClaw runtime. Each agent owns its scope, exposes
   tools via MCP, and reads/writes the shared memory.
2. **Shared memory** — the `(human, project, agent, interaction)`
   store described above. Currently mempalace, refactored from its
   original form.
3. **Handoff service** — produces, files, and delivers structured
   handoff summaries between agents.
4. **Coordination service** — runs the multi-agent protocol, including
   the facilitator role, and emits plans or escalations.
5. **Audit service** — collects structured records from all of the
   above and exposes the project-trace query.

The user-facing surface is a chat client (currently `pinchchat`) that
routes the user to whichever agent is the current holder of their
project, and shows the project's state alongside the conversation.

## What this is not

- **Not a chatbot wrapper over Jira.** Tickets are an output of the
  coordination protocol, not the substrate. The substrate is the
  interaction record.
- **Not a replacement for component owners.** The human owners of
  each component remain. The agents are interfaces over the
  components, accountable to the same owners.
- **Not summarisation-first.** Summaries are derived. The verbatim
  record is the source of truth.
- **Not a single super-agent.** No agent claims authority outside its
  scope. Coordination across scopes happens through the defined
  protocol, not through a master agent that knows everything.

## Open questions for review

These are the parts I am least sure about and most want pushed on:

1. **Project identity.** A project is the obvious unit, but real work
   has sub-projects, dependencies, and projects that fork or merge.
   What is the minimum graph structure we need on top of a flat
   project ID, and can we defer it?
2. **Agent authority boundaries.** "What can the agent decide alone"
   is the load-bearing definition. We need a way to express it that is
   readable by humans, enforceable at runtime, and revisable without a
   redeploy.
3. **The facilitator role.** A facilitator agent compiling positions
   from peer agents is the most speculative piece of the coordination
   protocol. Does it need to be an agent, or is it a deterministic
   function over the agents' stated positions?
4. **Memory scaling.** Sub-200ms at hundreds of thousands of
   interactions on one node. The shape of the system at ten million
   interactions across a federated estate is not yet designed.
5. **Identity across organisations.** A project may span agents owned
   by different organisations (e.g. a school and an external service).
   Federation of identity and memory is open.
6. **Escalation back to humans.** When the coordination protocol
   fails, the named human owner gets the full exchange. The interface
   for that — what they see, what they're being asked to decide — is
   not specified.

## Implementation status

- Component agents: running on a small estate (Raspberry Pi cluster,
  edge server, cloud) on the OpenClaw runtime.
- Shared memory: SurrealDB-backed, concurrent, WAL-backed, sub-200ms
  semantic search at current scale.
- Handoff protocol: working at single-handoff. Multi-step handoffs
  with persistent context: working.
- Coordination protocol: defined; not yet in production.
- Audit layer: per-agent logging consistent; cross-system linking
  partial.

The remaining work is integration and scale rather than new
architecture.
