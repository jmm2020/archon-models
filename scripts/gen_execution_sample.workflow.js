export const meta = {
  name: 'gen-archon-execution-sample',
  description: 'Distill a small grounded sample of Archon in-node EXECUTION training examples (Claude teacher) from the real .archon/commands',
  phases: [{ title: 'Generate', detail: 'one agent per (role, scenario) spec; reads the real command, produces the gold artifact' }],
}

// ── Fixed synthetic scenarios (so gold artifacts are grounded + consistent) ──
const REPO = `REPO CONTEXT (synthetic, but treat as real):
- "ledgerlite" — a small SaaS for freelance invoicing.
- Stack: TypeScript, React 18 + Vite frontend, Node/Express API, Postgres via Prisma.
- Layout: apps/web (React), apps/api (Express), packages/shared (types/utils).
- Conventions (CLAUDE.md): functions <40 lines; all API handlers wrapped in asyncHandler; errors via AppError(status, code, msg); tests colocated *.test.ts with vitest; no console.log in committed code.`

const SCENARIO_FEATURE = `${REPO}

FEATURE REQUEST ($ARGUMENTS): "Add CSV export to the Reports page so a user can download their filtered invoice list as a .csv."`

const SCENARIO_BUG = `${REPO}

GITHUB ISSUE #412 ($fetch-issue.output):
title: "Login redirect loop after SSO sign-in"
body: "After signing in via Google SSO, the app bounces between /login and /dashboard ~5 times then lands on /login with no error. Started after we shipped the new auth guard. Only happens for users whose session cookie is set but whose /api/me call returns 401 on the first request. Manual email/password login is fine."
labels: ["bug", "auth"]
state: open`

const SCENARIO_PR_DIFF = `${REPO}

PR #240 DIFF UNDER REVIEW (apps/api/src/middleware/rateLimit.ts, new file):
\`\`\`ts
import { Request, Response, NextFunction } from 'express'
const hits: Record<string, number[]> = {}
export function rateLimit(maxPerMin = 60) {
  return (req: Request, res: Response, next: NextFunction) => {
    const key = req.ip
    const now = Date.now()
    hits[key] = (hits[key] || []).filter(t => now - t < 60000)
    hits[key].push(now)
    if (hits[key].length > maxPerMin) {
      res.status(429).send('Too many requests')
      return
    }
    next()
  }
}
\`\`\`
(Applied globally in app.ts via app.use(rateLimit()).)`

const SCEN = {
  feature: SCENARIO_FEATURE,
  bug: SCENARIO_BUG,
  pr: SCENARIO_PR_DIFF,
  none: REPO,
}

const CMD_DIR = '/media/jmm2020/AIDrive1/UCIS-v1/.archon/commands'

// ── Spec list: execution-dominant. (bucket, role, command file, scenario, ask) ──
const SPECS = [
  // routing (thin slice)
  { bucket: 'routing', role: 'route-fix', cmd: null, scen: 'none',
    ask: 'Produce a ROUTING example: a realistic natural user request to fix a specific bug, mapped to {"workflow":"archon-fix-github-issue","confidence":"high","escalate":false}. System prompt = the Archon workflow router role. user = the natural request. assistant = ONLY the JSON.' },
  { bucket: 'routing', role: 'route-boundary', cmd: null, scen: 'none',
    ask: 'Produce a hard ROUTING boundary example: user explicitly wants the FULL comprehensive multi-agent fix+review pipeline (not a quick fix), mapped to {"workflow":"archon-issue-review-full","confidence":"high","escalate":false}. assistant = ONLY the JSON.' },
  // escalation reflex
  { bucket: 'escalation', role: 'escalate-design', cmd: null, scen: 'none',
    ask: 'Produce an ESCALATION example: user asks an open-ended hard architecture/design question that no Archon workflow fits and that should go to a frontier model. assistant = {"workflow":null,"confidence":"low","escalate":true}. The system prompt should make clear the local agent escalates genuinely hard planning/design.' },
  // structured-output / classify
  { bucket: 'structured-output', role: 'classify-issue', cmd: 'archon-fix-github-issue', scen: 'bug',
    ask: 'Produce a STRUCTURED-OUTPUT example for the "classify" node of archon-fix-github-issue: given the issue, emit ONLY the JSON classification the node requires (issue_type bug|feature|enhancement|question, plus a short reason). Read the workflow file at /media/jmm2020/AIDrive1/UCIS-v1/.archon/workflows/archon-fix-github-issue.yaml to match the classify nodes exact output_format.' },
  // execution — the bulk
  { bucket: 'execution', role: 'create-plan', cmd: 'archon-create-plan', scen: 'feature',
    ask: 'Execute the create-plan command on the feature scenario. assistant = the gold plan.md content (concise but real: goal, files to touch, ordered tasks, test plan, risks). PLAN ONLY, no code.' },
  { bucket: 'execution', role: 'investigate-issue', cmd: 'archon-investigate-issue', scen: 'bug',
    ask: 'Execute the investigate-issue command on the bug scenario. assistant = the gold investigation.md: root-cause hypothesis (the SSO/auth-guard race), evidence, the exact files/lines likely involved, and the minimal fix direction.' },
  { bucket: 'execution', role: 'fix-issue', cmd: 'archon-fix-issue', scen: 'bug',
    ask: 'Execute the fix-issue command after investigation. assistant = the gold fix: a focused code change (diff or clearly-scoped edits) plus a one-paragraph rationale, honoring the repo conventions.' },
  { bucket: 'execution', role: 'code-review', cmd: 'archon-code-review-agent', scen: 'pr',
    ask: 'Execute the code-review-agent on the PR diff. assistant = the gold code-review-findings.md. It MUST catch the real bugs in the diff (in-memory map never evicts keys -> unbounded memory leak; req.ip can be undefined behind a proxy; global mutable state not testable; off-by-one on >maxPerMin). Findings with severity + fix options.' },
  { bucket: 'execution', role: 'error-handling-review', cmd: 'archon-error-handling-agent', scen: 'pr',
    ask: 'Execute the error-handling-agent on the PR diff. assistant = gold error-handling findings: not wrapped in asyncHandler, uses res.send not AppError, no handling for missing ip, silent on Date.now drift. Match repo conventions.' },
  { bucket: 'execution', role: 'test-coverage-review', cmd: 'archon-test-coverage-agent', scen: 'pr',
    ask: 'Execute the test-coverage-agent on the PR diff. assistant = gold test-coverage findings: zero tests for rateLimit; list the specific vitest cases that should exist (under limit, over limit, window reset, per-ip isolation, missing-ip).' },
  { bucket: 'execution', role: 'synthesize-review', cmd: 'archon-synthesize-review', scen: 'pr',
    ask: 'Execute synthesize-review merging the three review artifacts above into one decision matrix. assistant = gold synthesis: deduped prioritized findings + an overall verdict (request-changes) + ordered fix list.' },
  { bucket: 'execution', role: 'self-fix-all', cmd: 'archon-self-fix-all', scen: 'pr',
    ask: 'Execute self-fix-all: implement fixes for the synthesized findings (not just report). assistant = the gold corrected rateLimit.ts (bounded store with eviction, proxy-safe key, injectable clock, asyncHandler/AppError, >= fix) + the new vitest test file.' },
  { bucket: 'execution', role: 'simplify-changes', cmd: 'archon-simplify-changes', scen: 'pr',
    ask: 'Execute simplify-changes on the corrected code. assistant = gold simplification notes + any concrete reduction (scope-limited to the changed file), or an honest "already minimal" with reasoning.' },
  { bucket: 'execution', role: 'validate', cmd: 'archon-validate', scen: 'pr',
    ask: 'Execute the validate command. assistant = the gold validation.md report: the exact commands run (pnpm -C apps/api typecheck/lint/test), realistic pass/fail summary, and a clear PASS/FAIL gate verdict.' },
  { bucket: 'execution', role: 'implement-tasks', cmd: 'archon-implement-tasks', scen: 'feature',
    ask: 'Execute implement-tasks for the FIRST task of the CSV-export plan. assistant = the gold implementation of that one task (the export util + wiring), with the colocated vitest test, honoring conventions.' },
]

const SYS_HINT = `You are generating ONE supervised fine-tuning example that teaches a LOCAL model (Gemma4-Archon) to BE the AI agent inside an Archon workflow node. Archon is a YAML DAG workflow engine for AI coding agents; each prompt/command node runs an agent that must produce a specific artifact. You are the TEACHER (a frontier model) producing the gold target.`

const SCHEMA = {
  type: 'object',
  properties: {
    system: { type: 'string', description: 'The Archon in-node agent role/system prompt for this example' },
    user: { type: 'string', description: 'The resolved node input the agent sees: the command mission + the concrete scenario context (resolved $ARGUMENTS / upstream outputs)' },
    assistant: { type: 'string', description: 'The GOLD artifact a strong agent would produce for this node — complete, correct, in the exact format the command/node demands' },
    role: { type: 'string' },
    bucket: { type: 'string' },
  },
  required: ['system', 'user', 'assistant', 'role', 'bucket'],
}

phase('Generate')
const results = await parallel(SPECS.map((s, i) => () =>
  agent(
    `${SYS_HINT}

${s.cmd ? `Read the REAL Archon command template first: ${CMD_DIR}/${s.cmd}.md — your gold "assistant" output must satisfy that command's actual mission and output format.` : 'This is a routing/escalation example — no command file; follow the routing-agent contract described below.'}

SCENARIO:
${SCEN[s.scen]}

TASK: ${s.ask}

Rules:
- The "user" field must read like a real resolved Archon node input (mission text + concrete context), NOT a meta-description.
- The "assistant" field is the GOLD completion. Make it genuinely correct and useful, faithful to the repo conventions. Keep it focused: complete but not padded (target 150-600 words for execution artifacts; exact JSON only for routing/classify).
- Do NOT invent Archon features. Stay faithful to the real command you read.
- Return ONLY the structured object (system, user, assistant, role="${s.role}", bucket="${s.bucket}").`,
    { label: `${s.bucket}:${s.role}`, phase: 'Generate', schema: SCHEMA }
  ).then(r => r ? ({
    conversations: [
      { role: 'system', content: r.system },
      { role: 'user', content: r.user },
      { role: 'assistant', content: r.assistant },
    ],
    meta: { bucket: r.bucket, role: r.role, scenario: s.scen, source: 'distilled', teacher: 'claude' },
  }) : null)
))

const records = results.filter(Boolean)
log(`generated ${records.length}/${SPECS.length} records`)
return { count: records.length, records }
