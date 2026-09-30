import { episodeJson, nowIso, relayEnv } from "@/lib/relay";

const REASONS = new Set(["success", "collision", "stuck", "timeout", "agent_stop", "user_abort"]);

export async function POST(request: Request, context: { params: Promise<{ episodeId: string }> }) {
  const { episodeId } = await context.params;
  const episode = await episodeJson(episodeId);
  if (!episode) return Response.json({ error: "episode does not exist" }, { status: 404 });
  if (episode.status === "finished") return Response.json({ episode });
  const payload = (await request.json()) as Record<string, unknown>;
  const reason = String(payload.reason ?? "");
  if (!REASONS.has(reason)) return Response.json({ error: "unsupported termination reason" }, { status: 400 });
  const now = nowIso();
  const termination = {
    reason,
    success: reason === "success",
    notes: String(payload.notes ?? "").slice(0, 1000),
    client_time: payload.client_time ?? null,
    server_time: now,
    sensor: payload.sensor ?? {},
  };
  const finishRequestId = `finish-${crypto.randomUUID()}`;
  await relayEnv().DB.batch([
    relayEnv().DB.prepare(
      "UPDATE episodes SET status = 'finished', updated_at = ?, termination_json = ? WHERE id = ?",
    ).bind(now, JSON.stringify(termination), episodeId),
    relayEnv().DB.prepare(
      `INSERT INTO jobs
         (request_id, episode_id, kind, status, result_json, created_at, updated_at)
       VALUES (?, ?, 'finish', 'pending', ?, ?, ?)`,
    ).bind(finishRequestId, episodeId, JSON.stringify(termination), now, now),
  ]);
  return Response.json({ episode: await episodeJson(episodeId) });
}
