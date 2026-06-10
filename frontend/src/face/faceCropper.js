export async function startCamera(video) {
  if (video.srcObject) {
    await video.play();
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720 },
    audio: false
  });

  video.srcObject = stream;
  await video.play();
}

export function stopCamera(video) {
  const stream = video.srcObject;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
  video.pause();
  video.srcObject = null;
}

function estimateCenteredFaceBox(video) {
  // Fallback robusto: crop cuadrado centrado. Puede reemplazarse por MediaPipe JS.
  const w = video.videoWidth;
  const h = video.videoHeight;
  const side = Math.floor(Math.min(w, h) * 0.62);
  return {
    x: Math.floor((w - side) / 2),
    y: Math.floor((h - side) / 2),
    w: side,
    h: side
  };
}

export async function captureFaceCropBlob(video, canvas) {
  if (!video.srcObject || !video.videoWidth) {
    throw new Error('La cámara no está lista. Primero inicia la cámara.');
  }

  const box = estimateCenteredFaceBox(video);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, box.x, box.y, box.w, box.h, 0, 0, canvas.width, canvas.height);

  return await new Promise((resolve, reject) => {
    canvas.toBlob(blob => {
      if (!blob) reject(new Error('No se pudo generar el crop facial.'));
      else resolve(blob);
    }, 'image/jpeg', 0.82);
  });
}

export async function captureFrameBatch(video, canvas, n = 16, delayMs = 120) {
  const frames = [];
  for (let i = 0; i < n; i++) {
    frames.push(await captureFaceCropBlob(video, canvas));
    await new Promise(r => setTimeout(r, delayMs));
  }
  return frames;
}
