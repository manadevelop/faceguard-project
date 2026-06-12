import {
  FaceDetector,
  FilesetResolver
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15';

const OUTPUT_SIZE = 224;

const MEDIAPIPE_WASM_PATH =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm';

const FACE_DETECTOR_MODEL_PATH =
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite';

let faceDetector = null;
let detectorLoadingPromise = null;

/**
 * Inicializa MediaPipe Face Detector una sola vez.
 * Se usa runningMode VIDEO porque la entrada viene desde webcam.
 */
async function getMediaPipeFaceDetector() {
  if (faceDetector) {
    return faceDetector;
  }

  if (!detectorLoadingPromise) {
    detectorLoadingPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_PATH);

      const detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: FACE_DETECTOR_MODEL_PATH,
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        minDetectionConfidence: 0.55
      });

      faceDetector = detector;
      return detector;
    })();
  }

  return detectorLoadingPromise;
}

export async function startCamera(video) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: 'user'
    },
    audio: false
  });

  video.srcObject = stream;
  await video.play();

  // Carga anticipada del detector para que el primer crop no demore.
  try {
    await getMediaPipeFaceDetector();
  } catch (error) {
    console.warn('No se pudo inicializar MediaPipe Face Detector:', error);
  }
}

export function stopCamera(video) {
  const stream = video.srcObject;

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  video.pause();
  video.srcObject = null;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

function clampCropBox(box, frameWidth, frameHeight) {
  let { x, y, size } = box;

  if (size > frameWidth) {
    size = frameWidth;
  }

  if (size > frameHeight) {
    size = frameHeight;
  }

  x = clamp(x, 0, frameWidth - size);
  y = clamp(y, 0, frameHeight - size);

  return {
    x: Math.round(x),
    y: Math.round(y),
    w: Math.round(size),
    h: Math.round(size)
  };
}

/**
 * Selecciona el rostro más grande detectado.
 * Esto evita problemas si aparece más de una persona en cámara.
 */
async function detectLargestFace(video) {
  const detector = await getMediaPipeFaceDetector();
  const timestampMs = performance.now();

  const result = detector.detectForVideo(video, timestampMs);

  if (!result || !result.detections || result.detections.length === 0) {
    return null;
  }

  const largestDetection = result.detections
    .filter((detection) => detection.boundingBox)
    .sort((a, b) => {
      const boxA = a.boundingBox;
      const boxB = b.boundingBox;
      return (boxB.width * boxB.height) - (boxA.width * boxA.height);
    })[0];

  if (!largestDetection || !largestDetection.boundingBox) {
    return null;
  }

  const box = largestDetection.boundingBox;

  return {
    x: box.originX,
    y: box.originY,
    w: box.width,
    h: box.height
  };
}

/**
 * Calcula un crop tipo documento/retrato:
 * - rostro completo centrado;
 * - parte superior de la cabeza visible;
 * - hombros visibles;
 * - parte del torso visible;
 * - salida cuadrada para normalizar a 224x224.
 *
 * Objetivo visual aproximado:
 * - cabeza/rostro: zona superior predominante;
 * - hombros/torso: tercio inferior.
 */
function estimatePortraitCropFromFace(faceBox, video) {
  const frameWidth = video.videoWidth;
  const frameHeight = video.videoHeight;

  const faceCenterX = faceBox.x + faceBox.w / 2;

  /*
   * Objetivo final:
   * - rostro completo dentro del 224x224;
   * - cabeza/cabello completo visible;
   * - mentón visible;
   * - rostro grande;
   * - mínimo fondo posible.
   */

  const cropSizeFromWidth = faceBox.w * 1.62;
  const cropSizeFromHeight = faceBox.h * 1.72;

  const cropSize = Math.max(cropSizeFromWidth, cropSizeFromHeight);

  const cropX = faceCenterX - cropSize / 2;

  /*
   * Se sube el crop para recuperar la parte superior de la cabeza.
   * 0.36 funciona mejor cuando MediaPipe detecta solo rostro y no cabello completo.
   */
  const cropY = faceBox.y - cropSize * 0.36;

  return clampCropBox(
    {
      x: cropX,
      y: cropY,
      size: cropSize
    },
    frameWidth,
    frameHeight
  );
}

/**
 * Fallback solo si MediaPipe no detecta rostro.
 * No es lo ideal, pero evita romper la aplicación.
 */
function estimateFallbackPortraitCrop(video) {
  const frameWidth = video.videoWidth;
  const frameHeight = video.videoHeight;

  const cropSize = Math.floor(Math.min(frameWidth, frameHeight) * 0.88);
  const cropX = (frameWidth - cropSize) / 2;
  const cropY = (frameHeight - cropSize) * 0.34;

  return clampCropBox(
    {
      x: cropX,
      y: cropY,
      size: cropSize
    },
    frameWidth,
    frameHeight
  );
}

function drawCrop(video, canvas, box) {
  const ctx = canvas.getContext('2d');

  canvas.width = OUTPUT_SIZE;
  canvas.height = OUTPUT_SIZE;

  ctx.clearRect(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);

  ctx.drawImage(
    video,
    box.x,
    box.y,
    box.w,
    box.h,
    0,
    0,
    OUTPUT_SIZE,
    OUTPUT_SIZE
  );
}

export async function captureFaceCropBlob(video, canvas) {
  if (!video.srcObject || !video.videoWidth || !video.videoHeight) {
    throw new Error('La cámara no está lista. Primero inicia la cámara.');
  }

  let cropBox = null;

  try {
    const detectedFace = await detectLargestFace(video);

    if (detectedFace) {
      cropBox = estimatePortraitCropFromFace(detectedFace, video);
    }
  } catch (error) {
    console.warn('Error detectando rostro con MediaPipe:', error);
  }

  if (!cropBox) {
    cropBox = estimateFallbackPortraitCrop(video);
  }

  drawCrop(video, canvas, cropBox);

  return await new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error('No se pudo generar el crop facial.'));
          return;
        }

        resolve(blob);
      },
      'image/jpeg',
      0.92
    );
  });
}

export async function captureFrameBatch(video, canvas, n = 16, delayMs = 120) {
  const frames = [];

  for (let i = 0; i < n; i++) {
    frames.push(await captureFaceCropBlob(video, canvas));
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }

  return frames;
}