// Define la ruta base de todos los endpoints de la API backend.
const API_BASE = '/api/v1';


// Procesa la respuesta HTTP recibida desde el backend.
async function parseResponse(res) {
  // Intenta convertir la respuesta a JSON; si falla, usa un objeto vacío.
  const data = await res.json().catch(() => ({}));

  // Verifica si la respuesta HTTP indica error.
  if (!res.ok) {
    // Obtiene el detalle del error enviado por el backend o construye uno con el código HTTP.
    const detail = data.detail || data.message || `HTTP ${res.status}`;

    // Lanza un error legible para mostrarlo en la interfaz.
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }

  // Devuelve la respuesta JSON cuando la solicitud fue exitosa.
  return data;
}


// Envía una imagen facial al backend para verificación de liveness/autenticación.
export async function postImage(blob, model, personId) {
  // Crea un formulario multipart para enviar la imagen al backend.
  const form = new FormData();

  // Agrega el crop facial con el nombre de campo esperado por FastAPI.
  form.append('face', blob, 'face.jpg');

  // Agrega el person_id solo cuando se requiere verificación de identidad.
  if (personId) form.append('person_id', personId);

  // Construye la URL del endpoint de verificación por imagen con el modelo seleccionado.
  const url = `${API_BASE}/auth/verify-image?model=${encodeURIComponent(model)}`;

  // Envía la solicitud POST al backend con el formulario.
  const res = await fetch(url, { method: 'POST', body: form });

  // Procesa y retorna la respuesta del backend.
  return parseResponse(res);
}


// Envía varios frames faciales al backend para verificación temporal.
export async function postFrames(blobs, model, personId) {
  // Crea un formulario multipart para enviar la secuencia de frames.
  const form = new FormData();

  // Agrega cada frame al formulario con el campo esperado por FastAPI.
  blobs.forEach((b, i) => form.append('frames', b, `frame_${i}.jpg`));

  // Agrega el person_id solo cuando se requiere verificación de identidad.
  if (personId) form.append('person_id', personId);

  // Construye la URL del endpoint de verificación por video con el modelo seleccionado.
  const url = `${API_BASE}/auth/verify-realtime?model=${encodeURIComponent(model)}`;

  // Envía la solicitud POST al backend con todos los frames.
  const res = await fetch(url, { method: 'POST', body: form });

  // Procesa y retorna la respuesta del backend.
  return parseResponse(res);
}


// Envía una imagen facial al backend para registrar una identidad.
export async function enrollIdentity(blob, personId) {
  // Crea un formulario multipart para enviar datos de enrolamiento.
  const form = new FormData();

  // Agrega el identificador de persona requerido por el backend.
  form.append('person_id', personId);

  // Agrega el crop facial que será usado para generar el embedding.
  form.append('face', blob, 'face.jpg');

  // Envía la solicitud POST al endpoint de enrolamiento de identidad.
  const res = await fetch(`${API_BASE}/identity/enroll`, { method: 'POST', body: form });

  // Procesa y retorna la respuesta del backend.
  return parseResponse(res);
}