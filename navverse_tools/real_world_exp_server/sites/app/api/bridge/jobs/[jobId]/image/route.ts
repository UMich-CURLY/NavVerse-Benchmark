import { bridgeAuthorized, relayEnv, unauthorized } from "@/lib/relay";

export async function GET(request: Request, context: { params: Promise<{ jobId: string }> }) {
  if (!bridgeAuthorized(request)) return unauthorized();
  const { jobId } = await context.params;
  const row = await relayEnv().DB.prepare("SELECT frame_key FROM jobs WHERE id = ? AND kind = 'step'")
    .bind(jobId)
    .first<{ frame_key: string }>();
  if (!row?.frame_key) return Response.json({ error: "frame does not exist" }, { status: 404 });
  const object = await relayEnv().FRAMES.get(row.frame_key);
  if (!object) return Response.json({ error: "frame is unavailable" }, { status: 404 });
  return new Response(object.body, { headers: { "Content-Type": "image/jpeg", "Cache-Control": "no-store" } });
}
