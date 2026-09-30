import { episodeJson } from "@/lib/relay";

export async function GET(_request: Request, context: { params: Promise<{ episodeId: string }> }) {
  const { episodeId } = await context.params;
  const episode = await episodeJson(episodeId);
  if (!episode) return Response.json({ error: "episode does not exist" }, { status: 404 });
  return Response.json({ episode });
}
