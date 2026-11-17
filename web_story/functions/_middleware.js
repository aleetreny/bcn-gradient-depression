// HTTP Basic Auth gate for every request. Any username; the password is the
// SITE_PASSWORD secret. Fails closed: if the secret is missing, nothing is served.
const realm = 'BCN web story';

function deny(status, msg) {
  return new Response(msg, {
    status,
    headers: {
      'WWW-Authenticate': `Basic realm="${realm}", charset="UTF-8"`,
      'Cache-Control': 'no-store',
      'Content-Type': 'text/plain; charset=utf-8',
    },
  });
}

function safeEqual(a, b) {
  const ea = new TextEncoder().encode(a);
  const eb = new TextEncoder().encode(b);
  let diff = ea.length ^ eb.length;
  for (let i = 0; i < Math.max(ea.length, eb.length); i++) diff |= (ea[i] || 0) ^ (eb[i] || 0);
  return diff === 0;
}

export async function onRequest({ request, env, next }) {
  const expected = env.SITE_PASSWORD;
  if (!expected) return new Response('Site locked: password not configured.', { status: 503, headers: { 'Cache-Control': 'no-store' } });

  const header = request.headers.get('Authorization') || '';
  if (!header.startsWith('Basic ')) return deny(401, 'Password required.');
  let decoded = '';
  try {
    decoded = new TextDecoder().decode(Uint8Array.from(atob(header.slice(6)), (c) => c.charCodeAt(0)));
  } catch {
    return deny(401, 'Password required.');
  }
  const password = decoded.slice(decoded.indexOf(':') + 1);
  if (!safeEqual(password, expected)) return deny(401, 'Wrong password.');

  const res = await next();
  const out = new Response(res.body, res);
  out.headers.set('Cache-Control', 'private, max-age=3600');
  return out;
}
