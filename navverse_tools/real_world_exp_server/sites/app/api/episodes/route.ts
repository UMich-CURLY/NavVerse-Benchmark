import { episodeJson, nowIso, parseJson, relayEnv, safeEpisodeId } from "@/lib/relay";

export async function GET() {
  const rows = await relayEnv().DB.prepare(
    `SELECT id, instruction, agent_id, agent_label, status, started_at, updated_at, termination_json,
            (SELECT COUNT(*) FROM jobs WHERE episode_id = episodes.id AND kind = 'step' AND status = 'complete') AS num_steps
       FROM episodes ORDER BY updated_at DESC LIMIT 50`,
  ).all<Record<string, unknown>>();
  return Response.json({
    episodes: rows.results.map((row) => ({
      episode_id: row.id,
      instruction: row.instruction,
      agent: { id: row.agent_id, label: row.agent_label },
      status: row.status,
      started_at: row.started_at,
      updated_at: row.updated_at,
      num_steps: row.num_steps,
      termination: row.termination_json ? JSON.parse(String(row.termination_json)) : null,
    })),
  });
}

export async function POST(request: Request) {
  const payload = (await request.json()) as Record<string, unknown>;
  const instruction = String(payload.instruction ?? "").trim();
  if (!instruction) return Response.json({ error: "instruction is required" }, { status: 400 });
  const agentId = String(payload.agent_id ?? "").trim();
  if (!/^[a-z0-9][a-z0-9._-]{0,63}$/.test(agentId)) {
    return Response.json({ error: "select a valid local agent" }, { status: 400 });
  }
  const onlineAfter = new Date(Date.now() - 60_000).toISOString();
  const agent = await relayEnv().DB.prepare(
    "SELECT id, label, capabilities_json FROM agents WHERE id = ? AND updated_at >= ?",
  )
    .bind(agentId, onlineAfter)
    .first<{ id: string; label: string; capabilities_json: string }>();
  if (!agent) return Response.json({ error: "selected local agent is offline" }, { status: 409 });
  const capabilities = parseJson<Record<string, unknown>>(agent.capabilities_json, {});
  if (capabilities.selectable === false) {
    return Response.json(
      { error: String(capabilities.detail ?? "selected local agent cannot use this observation") },
      { status: 409 },
    );
  }
  const active = await relayEnv().DB.prepare("SELECT id FROM episodes WHERE status = 'active' LIMIT 1").first();
  if (active) return Response.json({ error: `episode '${active.id}' is still active` }, { status: 409 });
  const episodeId = safeEpisodeId(payload.episode_id);
  const hfov = Number(payload.hfov ?? 79);
  if (!Number.isFinite(hfov) || hfov < 1 || hfov > 179) {
    return Response.json({ error: "hfov must be between 1 and 179 degrees" }, { status: 400 });
  }
  const now = nowIso();
  const warmupRequestId = `warmup-${crypto.randomUUID()}`;
  try {
    await relayEnv().DB.batch([
      relayEnv().DB.prepare(
        `INSERT INTO episodes
           (id, instruction, agent_id, agent_label, status, hfov, client_metadata_json, started_at, updated_at)
         VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?)`,
      ).bind(
          episodeId,
          instruction,
          agent.id,
          agent.label,
          hfov,
          JSON.stringify(payload.client_metadata ?? {}),
          now,
          now,
        ),
      relayEnv().DB.prepare(
        `INSERT INTO jobs
           (request_id, episode_id, kind, status, created_at, updated_at)
         VALUES (?, ?, 'warmup', 'pending', ?, ?)`,
      ).bind(warmupRequestId, episodeId, now, now),
    ]);
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : "episode creation failed" }, { status: 409 });
  }
  return Response.json({ episode: await episodeJson(episodeId) }, { status: 201 });
}
