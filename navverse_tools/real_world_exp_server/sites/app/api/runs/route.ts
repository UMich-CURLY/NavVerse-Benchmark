import { parseJson, relayEnv, runVideoKey } from "@/lib/relay";

export async function GET() {
  const onlineAfter = new Date(Date.now() - 20_000).toISOString();
  const sync = await relayEnv().DB.prepare(
    `SELECT MAX(synced_at) AS indexed_at FROM (
       SELECT indexed_at AS synced_at FROM local_runs
       UNION ALL
       SELECT updated_at AS synced_at FROM agents
     )`,
  ).first<{ indexed_at: string | null }>();
  const rows = await relayEnv().DB.prepare(
    `SELECT episode_id, instruction, agent_id, agent_label, started_at, updated_at,
            termination_json, num_steps, video_available, video_size_bytes
       FROM local_runs
      WHERE status = 'finished'
      ORDER BY updated_at DESC
      LIMIT 50`,
  ).all<Record<string, unknown>>();
  const runs = await Promise.all(
    rows.results.map(async (row) => {
      const episodeId = String(row.episode_id);
      const video = await relayEnv().FRAMES.head(runVideoKey(episodeId));
      return {
        episode_id: episodeId,
        instruction: row.instruction,
        agent: { id: row.agent_id, label: row.agent_label },
        started_at: row.started_at,
        finished_at: row.updated_at,
        num_steps: Number(row.num_steps ?? 0),
        termination: parseJson(String(row.termination_json ?? ""), null),
        video: {
          available: Number(row.video_available) === 1,
          cached: Boolean(video),
          url: `/api/runs/${encodeURIComponent(episodeId)}/video`,
          size_bytes: Number(row.video_size_bytes ?? video?.size ?? 0) || null,
        },
      };
    }),
  );
  return Response.json(
    {
      source: "local_bridge",
      bridge_online: Boolean(sync?.indexed_at && sync.indexed_at >= onlineAfter),
      last_synced_at: sync?.indexed_at ?? null,
      runs,
    },
    { headers: { "Cache-Control": "no-store, max-age=0" } },
  );
}
