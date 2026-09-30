import { attachmentKey, bridgeAuthorized, relayEnv, unauthorized } from "@/lib/relay";

export async function GET(
  request: Request,
  context: { params: Promise<{ jobId: string; assetName: string }> },
) {
  if (!bridgeAuthorized(request)) return unauthorized();
  const { jobId, assetName } = await context.params;
  if (assetName !== "depth" && assetName !== "confidence") {
    return Response.json({ error: "unsupported observation asset" }, { status: 404 });
  }
  const row = await relayEnv().DB.prepare("SELECT frame_key FROM jobs WHERE id = ? AND kind = 'step'")
    .bind(jobId)
    .first<{ frame_key: string }>();
  if (!row?.frame_key) return Response.json({ error: "frame does not exist" }, { status: 404 });
  const object = await relayEnv().FRAMES.get(attachmentKey(row.frame_key, assetName));
  if (!object) return Response.json({ error: "observation asset is unavailable" }, { status: 404 });
  return new Response(object.body, { headers: { "Content-Type": "image/png", "Cache-Control": "no-store" } });
}
