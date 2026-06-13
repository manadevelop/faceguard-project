import {
  FaceDetector,
  FilesetResolver
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15';


// Tamaño final del crop facial enviado al backend.
const OUTPUT_SIZE = 224;

// Ruta del paquete WASM necesario para ejecutar MediaPipe en el navegador.
const MEDIAPIPE_WASM_PATH =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm';

// Ruta del modelo BlazeFace usado por MediaPipe para detectar rostros.
const FACE_DETECTOR_MODEL_PATH =
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite';

// Instancia global del detector facial para evitar recrearlo en cada captura.
let faceDetector = null;

// Promesa global para controlar la carga única del detector.
let detectorLoadingPromise = null;

/**
 * Inicializa MediaPipe Face Detector una sola vez.
 * Se usa runningMode VIDEO porque la entrada viene desde webcam.
 */
async function getMediaPipeFaceDetector() {
  // Reutiliza el detector si ya fue inicializado.
  if (faceDetector) {
    return faceDetector;
  }

  // Crea una única promesa de carga si todavía no existe.
  if (!detectorLoadingPromise) {
    detectorLoadingPromise = (async () => {
      // Carga los recursos WASM requeridos por MediaPipe.
      const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_PATH);

      // Crea el detector facial con modelo BlazeFace y ejecución en modo video.
      const detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: FACE_DETECTOR_MODEL_PATH,
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        minDetectionConfidence: 0.55
      });

      // Guarda el detector para reutilizarlo en próximas capturas.
      faceDetector = detector;

      // Retorna la instancia creada del detector facial.
      return detector;
    })();
  }

  // Retorna la promesa de carga del detector.
  return detectorLoadingPromise;
}

// Inicia la cámara del usuario y prepara el video para captura.
export async function startCamera(video) {
  // Solicita acceso a la webcam con resolución ideal HD y cámara frontal.
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: 'user'
    },
    audio: false
  });

  // Asigna el stream de la cámara al elemento video.
  video.srcObject = stream;

  // Reproduce el video para iniciar la vista en vivo.
  await video.play();

  // Carga anticipada del detector para que el primer crop no demore.
  try {
    await getMediaPipeFaceDetector();
  } catch (error) {
    console.warn('No se pudo inicializar MediaPipe Face Detector:', error);
  }
}

// Detiene la cámara y libera los recursos del stream.
export function stopCamera(video) {
  // Obtiene el stream actualmente asociado al video.
  const stream = video.srcObject;

  // Detiene todas las pistas activas de la cámara.
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  // Pausa el video mostrado en pantalla.
  video.pause();

  // Limpia la referencia al stream de la cámara.
  video.srcObject = null;
}

// Limita un valor numérico dentro de un rango mínimo y máximo.
function clamp(value, min, max) {
  // Retorna el valor ajustado para no salir de los límites permitidos.
  return Math.max(min, Math.min(value, max));
}

// Ajusta el cuadro de crop para que siempre permanezca dentro del frame.
function clampCropBox(box, frameWidth, frameHeight) {
  // Extrae las coordenadas y tamaño del cuadro de crop.
  let { x, y, size } = box;

  // Limita el tamaño del crop al ancho máximo del frame.
  if (size > frameWidth) {
    size = frameWidth;
  }

  // Limita el tamaño del crop al alto máximo del frame.
  if (size > frameHeight) {
    size = frameHeight;
  }

  // Ajusta la coordenada X para mantener el crop dentro del frame.
  x = clamp(x, 0, frameWidth - size);

  // Ajusta la coordenada Y para mantener el crop dentro del frame.
  y = clamp(y, 0, frameHeight - size);

  // Retorna el cuadro final con valores enteros.
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
  // Obtiene la instancia inicializada del detector facial.
  const detector = await getMediaPipeFaceDetector();

  // Genera una marca de tiempo requerida por detectForVideo.
  const timestampMs = performance.now();

  // Ejecuta la detección facial sobre el frame actual del video.
  const result = detector.detectForVideo(video, timestampMs);

  // Retorna null si no se detectó ningún rostro.
  if (!result || !result.detections || result.detections.length === 0) {
    return null;
  }

  // Ordena las detecciones por área y selecciona el rostro más grande.
  const largestDetection = result.detections
    .filter((detection) => detection.boundingBox)
    .sort((a, b) => {
      const boxA = a.boundingBox;
      const boxB = b.boundingBox;
      return (boxB.width * boxB.height) - (boxA.width * boxA.height);
    })[0];

  // Retorna null si la detección seleccionada no tiene bounding box válido.
  if (!largestDetection || !largestDetection.boundingBox) {
    return null;
  }

  // Extrae el bounding box del rostro seleccionado.
  const box = largestDetection.boundingBox;

  // Retorna las coordenadas y dimensiones del rostro detectado.
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
  // Obtiene el ancho real del frame de video.
  const frameWidth = video.videoWidth;

  // Obtiene el alto real del frame de video.
  const frameHeight = video.videoHeight;

  // Calcula el centro horizontal del rostro detectado.
  const faceCenterX = faceBox.x + faceBox.w / 2;

  /*
   * Objetivo final:
   * - rostro completo dentro del 224x224;
   * - cabeza/cabello completo visible;
   * - mentón visible;
   * - rostro grande;
   * - mínimo fondo posible.
   */

  // Calcula un tamaño de crop proporcional al ancho del rostro.
  const cropSizeFromWidth = faceBox.w * 1.62;

  // Calcula un tamaño de crop proporcional al alto del rostro.
  const cropSizeFromHeight = faceBox.h * 1.72;

  // Usa el mayor tamaño para asegurar que el rostro entre completo.
  const cropSize = Math.max(cropSizeFromWidth, cropSizeFromHeight);

  // Centra horizontalmente el crop respecto al rostro.
  const cropX = faceCenterX - cropSize / 2;

  /*
   * Se sube el crop para recuperar la parte superior de la cabeza.
   * 0.36 funciona mejor cuando MediaPipe detecta solo rostro y no cabello completo.
   */
  const cropY = faceBox.y - cropSize * 0.36;

  // Ajusta el crop para que no salga de los límites del frame.
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
  // Obtiene el ancho real del frame de video.
  const frameWidth = video.videoWidth;

  // Obtiene el alto real del frame de video.
  const frameHeight = video.videoHeight;

  // Calcula un crop centrado usando la menor dimensión del frame.
  const cropSize = Math.floor(Math.min(frameWidth, frameHeight) * 0.88);

  // Centra horizontalmente el crop en el frame.
  const cropX = (frameWidth - cropSize) / 2;

  // Ubica verticalmente el crop ligeramente hacia la zona superior.
  const cropY = (frameHeight - cropSize) * 0.34;

  // Ajusta el crop para que permanezca dentro del frame.
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

// Dibuja el crop facial en el canvas con tamaño final 224x224.
function drawCrop(video, canvas, box) {
  // Obtiene el contexto 2D del canvas.
  const ctx = canvas.getContext('2d');

  // Define el ancho final del canvas.
  canvas.width = OUTPUT_SIZE;

  // Define el alto final del canvas.
  canvas.height = OUTPUT_SIZE;

  // Limpia cualquier contenido previo del canvas.
  ctx.clearRect(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);

  // Copia la región facial del video y la escala al tamaño de salida.
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

// Captura un crop facial desde el video y lo devuelve como Blob JPEG.
export async function captureFaceCropBlob(video, canvas) {
  // Valida que la cámara esté activa y que el video tenga dimensiones válidas.
  if (!video.srcObject || !video.videoWidth || !video.videoHeight) {
    throw new Error('La cámara no está lista. Primero inicia la cámara.');
  }

  // Inicializa el cuadro de crop como nulo hasta detectar rostro o usar fallback.
  let cropBox = null;

  // Intenta detectar el rostro más grande con MediaPipe.
  try {
    const detectedFace = await detectLargestFace(video);

    // Calcula el crop tipo retrato si se detectó un rostro.
    if (detectedFace) {
      cropBox = estimatePortraitCropFromFace(detectedFace, video);
    }

  // Continúa con fallback si MediaPipe falla durante la detección.
  } catch (error) {
    console.warn('Error detectando rostro con MediaPipe:', error);
  }

  // Usa crop centrado de respaldo si no se obtuvo detección facial.
  if (!cropBox) {
    cropBox = estimateFallbackPortraitCrop(video);
  }

  // Dibuja el crop calculado en el canvas.
  drawCrop(video, canvas, cropBox);

  // Convierte el canvas a Blob JPEG para enviarlo al backend.
  return await new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        // Rechaza la promesa si el navegador no pudo generar el Blob.
        if (!blob) {
          reject(new Error('No se pudo generar el crop facial.'));
          return;
        }

        // Resuelve la promesa con el Blob generado.
        resolve(blob);
      },
      'image/jpeg',
      0.92
    );
  });
}

// Captura una secuencia de crops faciales para verificación por video.
export async function captureFrameBatch(video, canvas, n = 16, delayMs = 120) {
  // Lista donde se almacenan los frames capturados.
  const frames = [];

  // Captura n frames con una pausa entre cada captura.
  for (let i = 0; i < n; i++) {
    frames.push(await captureFaceCropBlob(video, canvas));
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }

  // Retorna la colección de frames lista para enviarse al backend.
  return frames;
}