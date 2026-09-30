import { relayEnv, runVideoKey } from "@/lib/relay";

export async function GET(
  _request: Request,
  context: { params: Promise<{ episodeId: string }> },
) {
  const { episodeId } = await context.params;
  const cached = await relayEnv().FRAMES.head(runVideoKey(episodeId));
  if (cached) {
    return Response.json(
      {
        status: "ready",
        url: `/api/runs/${encodeURIComponent(episodeId)}/video`,
        size_bytes: cached.size,
      },
      { headers: { "Cache-Control": "no-store, max-age=0" } },
    );
  }
  const row = await relayEnv().DB.prepare(
    "SELECT status, error, updated_at FROM video_requests WHERE episode_id = ?",
  )
    .bind(episodeId)
    .first<Record<string, unknown>>();
  return Response.json(
    row
      ? { status: row.status, error: row.error, updated_at: row.updated_at }
      : { status: "not_requested" },
    { headers: { "Cache-Control": "no-store, max-age=0" } },
  );
}
