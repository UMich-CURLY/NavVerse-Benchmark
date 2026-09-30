import { bridgeAuthorized, nowIso, parseJson, relayEnv, unauthorized } from "@/lib/relay";

export async function GET(request: Request) {
  if (!bridgeAuthorized(request)) return unauthorized();
  const agentIds = new URL(request.url).searchParams
    .getAll("agent_id")
    .filter((agentId) => /^[a-z0-9][a-z0-9._-]{0,63}$/.test(agentId))
    .slice(0, 32);
  if (!agentIds.length) return Response.json({ error: "agent_id is required" }, { status: 400 });
  const now = nowIso();
  const leaseUntil = new Date(Date.now() + 20 * 60_000).toISOString();
  const placeholders = agentIds.map(() => "?").join(", ");
  const row = await relayEnv().DB.prepare(
    `SELECT jobs.id, jobs.request_id, jobs.episode_id, jobs.kind, jobs.frame_key,
            jobs.captured_at, jobs.sensor_json, jobs.result_json,
            episodes.instruction, episodes.agent_id, episodes.agent_label,
            episodes.hfov, episodes.client_metadata_json
       FROM jobs JOIN episodes ON episodes.id = jobs.episode_id
      WHERE episodes.agent_id IN (${placeholders})
        AND (jobs.status = 'pending'
         OR (jobs.status = 'processing' AND jobs.lease_until < ?))
      ORDER BY jobs.id LIMIT 1`,
  )
    .bind(...agentIds, now)
    .first<Record<string, unknown>>();
  if (!row) return new Response(null, { status: 204 });
  const claimed = await relayEnv().DB.prepare(
    `UPDATE jobs SET status = 'processing', lease_until = ?, updated_at = ?
      WHERE id = ? AND (status = 'pending' OR (status = 'processing' AND lease_until < ?))`,
  )
    .bind(leaseUntil, now, row.id, now)
    .run();
  if (!claimed.meta.changes) return new Response(null, { status: 204 });
  return Response.json({
    job: {
      id: row.id,
      request_id: row.request_id,
      episode_id: row.episode_id,
      kind: row.kind,
      captured_at: row.captured_at,
      sensor: parseJson(String(row.sensor_json ?? "{}"), {}),
      termination: parseJson(String(row.result_json ?? ""), null),
      instruction: row.instruction,
      agent_id: row.agent_id,
      agent_label: row.agent_label,
      hfov: row.hfov,
      client_metadata: parseJson(String(row.client_metadata_json ?? "{}"), {}),
    },
  });
}
