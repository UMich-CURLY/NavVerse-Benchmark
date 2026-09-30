import { bridgeAuthorized, nowIso, relayEnv, unauthorized } from "@/lib/relay";

type AgentRegistration = {
  id?: unknown;
  label?: unknown;
  description?: unknown;
  capabilities?: unknown;
};

export async function POST(request: Request) {
  if (!bridgeAuthorized(request)) return unauthorized();
  const payload = (await request.json()) as { agents?: AgentRegistration[] };
  const agents = Array.isArray(payload.agents) ? payload.agents.slice(0, 32) : [];
  if (!agents.length) return Response.json({ error: "agents are required" }, { status: 400 });
  const now = nowIso();
  const statements = [];
  for (const agent of agents) {
    const id = String(agent.id ?? "").trim();
    const label = String(agent.label ?? "").trim().slice(0, 100);
    if (!/^[a-z0-9][a-z0-9._-]{0,63}$/.test(id) || !label) {
      return Response.json({ error: "invalid agent registration" }, { status: 400 });
    }
    statements.push(relayEnv().DB.prepare(
      `INSERT INTO agents (id, label, description, capabilities_json, updated_at)
       VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         label = excluded.label,
         description = excluded.description,
         capabilities_json = excluded.capabilities_json,
         updated_at = excluded.updated_at`,
    ).bind(
      id,
      label,
      String(agent.description ?? "").slice(0, 240),
      JSON.stringify(agent.capabilities ?? {}),
      now,
    ));
  }
  await relayEnv().DB.batch(statements);
  return Response.json({ ok: true, registered: statements.length, updated_at: now });
}
