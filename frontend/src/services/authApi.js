const API_BASE = '/api/v1';

async function parseResponse(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail || data.message || `HTTP ${res.status}`;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return data;
}

export async function postImage(blob, model, personId) {
  const form = new FormData();
  form.append('face', blob, 'face.jpg');
  if (personId) form.append('person_id', personId);
  const url = `${API_BASE}/auth/verify-image?model=${encodeURIComponent(model)}`;
  const res = await fetch(url, { method: 'POST', body: form });
  return parseResponse(res);
}

export async function postFrames(blobs, model, personId) {
  const form = new FormData();
  blobs.forEach((b, i) => form.append('frames', b, `frame_${i}.jpg`));
  if (personId) form.append('person_id', personId);
  const url = `${API_BASE}/auth/verify-realtime?model=${encodeURIComponent(model)}`;
  const res = await fetch(url, { method: 'POST', body: form });
  return parseResponse(res);
}

export async function enrollIdentity(blob, personId) {
  const form = new FormData();
  form.append('person_id', personId);
  form.append('face', blob, 'face.jpg');
  const res = await fetch(`${API_BASE}/identity/enroll`, { method: 'POST', body: form });
  return parseResponse(res);
}
