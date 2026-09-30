export async function GET() {
  return Response.json({ ok: true, auth_required: false, secure_context: true, relay: true });
}
