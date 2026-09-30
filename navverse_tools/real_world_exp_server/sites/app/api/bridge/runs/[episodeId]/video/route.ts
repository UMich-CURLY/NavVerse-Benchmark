import { bridgeAuthorized, relayEnv, runVideoKey, unauthorized } from "@/lib/relay";

const MAX_VIDEO_BYTES = 256 * 1024 * 1024;

export async function PUT(
  request: Request,
  context: { params: Promise<{ episodeId: string }> },
) {
  if (!bridgeAuthorized(request)) return unauthorized();
  const { episodeId } = await context.params;
  const contentLength = Number(request.headers.get("Content-Length") ?? 0);
  if (contentLength > MAX_VIDEO_BYTES) {
    return Response.json({ error: "video exceeds 256 MB" }, { status: 413 });
  }
  const episode = await relayEnv().DB.prepare(
    "SELECT episode_id FROM local_runs WHERE episode_id = ? AND status = 'finished' AND video_available = 1",
  )
    .bind(episodeId)
    .first();
  if (!episode) return Response.json({ error: "finished episode does not exist" }, { status: 404 });

  const bytes = await request.arrayBuffer();
  if (!bytes.byteLength || bytes.byteLength > MAX_VIDEO_BYTES) {
    return Response.json({ error: "video size is invalid" }, { status: 400 });
  }
  await relayEnv().FRAMES.put(runVideoKey(episodeId), bytes, {
    httpMetadata: { contentType: "video/mp4", cacheControl: "public, max-age=3600" },
    customMetadata: { episode_id: episodeId },
  });
  await relayEnv().DB.prepare(
    `UPDATE video_requests
        SET status = 'complete', error = NULL, lease_until = NULL, updated_at = ?
      WHERE episode_id = ?`,
  )
    .bind(new Date().toISOString(), episodeId)
    .run();
  return Response.json({ ok: true, size_bytes: bytes.byteLength });
}
