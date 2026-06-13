import {
  startCamera,
  stopCamera,
  captureFaceCropBlob,
  captureFrameBatch
} from './face/faceCropper.js';
import { postImage, postFrames, enrollIdentity } from './services/authApi.js';


// Referencia al indicador visual del estado global de la cámara.
const cameraStatusChip = document.getElementById('cameraStatusChip');

// Referencia al menú lateral de navegación.
const sidebar = document.getElementById('sidebar');

// Referencia al fondo oscuro usado cuando se abre el menú en móvil.
const drawerScrim = document.getElementById('drawerScrim');

// Referencia al botón que abre el menú lateral en vista móvil.
const openDrawerBtn = document.getElementById('openDrawerBtn');

// Lista de vistas principales disponibles en la interfaz.
const views = [...document.querySelectorAll('.view')];

// Lista de botones de navegación del menú lateral.
const navButtons = [...document.querySelectorAll('.nav-item')];

// Contexto de elementos HTML usados por el flujo de verificación de vida.
const liveness = {
  view: document.getElementById('livenessView'),
  card: document.getElementById('livenessCaptureCard'),
  title: document.getElementById('livenessCaptureTitle'),
  subtitle: document.getElementById('livenessCaptureSubtitle'),
  idleBox: document.getElementById('livenessIdleBox'),
  video: document.getElementById('livenessVideo'),
  canvas: document.getElementById('livenessCropCanvas'),
  result: document.getElementById('livenessResult'),
  model: document.getElementById('livenessModelSelect'),
  startBtn: document.getElementById('livenessStartBtn'),
  stopBtn: document.getElementById('livenessStopBtn'),
  actionGroup: document.getElementById('livenessVerifyActions'),
  imageBtn: document.getElementById('verifyImageBtn'),
  videoBtn: document.getElementById('verifyVideoBtn'),
  type: 'liveness'
};

// Contexto de elementos HTML usados por el flujo de enrolamiento facial.
const enrollment = {
  view: document.getElementById('enrollmentView'),
  card: document.getElementById('enrollCaptureCard'),
  title: document.getElementById('enrollCaptureTitle'),
  subtitle: document.getElementById('enrollCaptureSubtitle'),
  idleBox: document.getElementById('enrollIdleBox'),
  video: document.getElementById('enrollVideo'),
  canvas: document.getElementById('enrollCropCanvas'),
  result: document.getElementById('enrollResult'),
  model: document.getElementById('enrollModelSelect'),
  personId: document.getElementById('enrollPersonId'),
  startBtn: document.getElementById('enrollStartBtn'),
  stopBtn: document.getElementById('enrollStopBtn'),
  actionGroup: document.getElementById('enrollActionGroup'),
  registerBtn: document.getElementById('registerIdentityBtn'),
  type: 'enrollment'
};

// Muestra en pantalla un texto simple o un objeto JSON formateado.
function setResult(target, value) {
  // Convierte objetos a JSON legible y deja los textos tal como vienen.
  target.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

// Activa o desactiva el estado de carga de un botón.
function setLoading(button, isLoading, loadingText = 'Procesando...') {
  // Evita errores si el botón no existe.
  if (!button) return;

  // Cambia el texto y bloquea el botón mientras se procesa una acción.
  if (isLoading) {
    button.dataset.originalText = button.textContent;
    button.textContent = loadingText;
    button.disabled = true;

  // Restaura el texto original y vuelve a habilitar el botón.
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
    button.disabled = false;
  }
}

// Actualiza el indicador global del estado de la cámara.
function updateGlobalCameraChip(isActive) {
  // Muestra si la cámara está activa o detenida.
  cameraStatusChip.textContent = isActive ? 'Cámara activa' : 'Cámara detenida';

  // Aplica el estilo visual de cámara activa.
  cameraStatusChip.classList.toggle('status-active', isActive);

  // Aplica el estilo visual de cámara detenida.
  cameraStatusChip.classList.toggle('status-muted', !isActive);
}

// Cambia el estado visual de un flujo entre idle, camera y captured.
function setFlowState(ctx, state) {
  // Guarda el estado actual en el atributo data-state de la tarjeta.
  ctx.card.dataset.state = state;

  // Elimina estados visuales anteriores de la vista.
  ctx.view?.classList.remove('flow-state-idle', 'flow-state-camera', 'flow-state-captured');

  // Agrega el estado visual actual a la vista.
  ctx.view?.classList.add(`flow-state-${state}`);

  // Indica si el flujo está en estado inicial.
  const isIdle = state === 'idle';

  // Indica si el flujo está mostrando cámara en vivo.
  const isCamera = state === 'camera';

  // Indica si el flujo ya capturó un crop facial.
  const isCaptured = state === 'captured';

  // Muestra el video solo cuando la cámara está activa.
  ctx.video.classList.toggle('hidden', !isCamera);

  // Muestra el canvas solo cuando ya existe una captura.
  ctx.canvas.classList.toggle('hidden', !isCaptured);

  // Muestra el panel inicial solo cuando el flujo está inactivo.
  ctx.idleBox.classList.toggle('hidden', !isIdle);

  // Oculta el botón de iniciar mientras la cámara está activa.
  ctx.startBtn.classList.toggle('hidden', isCamera);

  // Muestra el botón de detener solo mientras la cámara está activa.
  ctx.stopBtn.classList.toggle('hidden', !isCamera);

  // Muestra las acciones solo cuando existe cámara activa.
  ctx.actionGroup.classList.toggle('hidden', !isCamera);

  // Deshabilita iniciar cámara mientras ya está encendida.
  ctx.startBtn.disabled = isCamera;

  // Deshabilita detener cámara cuando no está encendida.
  ctx.stopBtn.disabled = !isCamera;

  // Actualiza textos cuando ya se capturó el crop facial.
  if (isCaptured) {
    ctx.title.textContent = 'Crop facial enviado';
    ctx.subtitle.textContent = 'Rostro normalizado a 224x224.';

  // Actualiza textos cuando la cámara está lista o en estado inicial.
  } else {
    ctx.title.textContent = 'Cámara en vivo';
    ctx.subtitle.textContent = 'Vista completa de la webcam. No se envía completa al backend.';
  }

  // Sincroniza el chip global con el estado de la cámara.
  updateGlobalCameraChip(isCamera);
}

// Cierra el menú lateral en dispositivos móviles.
function closeMobileDrawer() {
  // Quita la clase que mantiene abierto el menú lateral.
  sidebar.classList.remove('drawer-open');

  // Oculta el fondo oscuro del menú móvil.
  drawerScrim.classList.remove('visible');

  // Marca el scrim como oculto para accesibilidad.
  drawerScrim.setAttribute('aria-hidden', 'true');
}

// Abre el menú lateral en dispositivos móviles.
function openMobileDrawer() {
  // Agrega la clase que muestra el menú lateral.
  sidebar.classList.add('drawer-open');

  // Muestra el fondo oscuro del menú móvil.
  drawerScrim.classList.add('visible');

  // Marca el scrim como visible para accesibilidad.
  drawerScrim.setAttribute('aria-hidden', 'false');
}

// Inicia la cámara de forma segura y actualiza la interfaz.
async function safeStartCamera(ctx) {
  // Intenta iniciar la webcam asociada al flujo recibido.
  try {
    await startCamera(ctx.video);
    setFlowState(ctx, 'camera');
    setResult(ctx.result, 'Cámara iniciada. Ubica el rostro dentro de la vista y ejecuta la acción.');

  // Muestra un error controlado si la cámara no puede iniciarse.
  } catch (error) {
    setFlowState(ctx, 'idle');
    setResult(ctx.result, `No se pudo iniciar la cámara: ${error.message}`);
  }
}

// Detiene la cámara de forma segura y opcionalmente reinicia el mensaje.
function safeStopCamera(ctx, options = { resetResult: true }) {
  // Intenta detener la cámara aunque luego se actualice el estado visual.
  try {
    stopCamera(ctx.video);

  // Garantiza que la interfaz vuelva al estado inicial.
  } finally {
    setFlowState(ctx, 'idle');

    // Muestra mensaje de cámara detenida si corresponde.
    if (options.resetResult) {
      setResult(ctx.result, 'Cámara detenida.');
    }
  }
}

// Detiene la cámara después de capturar y deja visible el crop generado.
function stopAfterCapture(ctx) {
  // Detiene el stream de video activo.
  try {
    stopCamera(ctx.video);

  // Cambia el flujo al estado capturado.
  } finally {
    setFlowState(ctx, 'captured');
  }
}

// Cambia la vista activa entre liveness y enrolamiento.
function setActiveView(targetId) {
  // Detiene la cámara de liveness si está activa antes de cambiar de vista.
  if (liveness.video.srcObject) safeStopCamera(liveness, { resetResult: false });

  // Detiene la cámara de enrolamiento si está activa antes de cambiar de vista.
  if (enrollment.video.srcObject) safeStopCamera(enrollment, { resetResult: false });

  // Activa visualmente solo la vista seleccionada.
  views.forEach((view) => {
    view.classList.toggle('active-view', view.id === targetId);
  });

  // Marca como activo el botón de navegación correspondiente.
  navButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.target === targetId);
  });

  // Cierra el drawer móvil después de seleccionar una vista.
  closeMobileDrawer();
}

// Asocia cada botón de navegación con el cambio de vista correspondiente.
navButtons.forEach((button) => {
  // Cambia a la vista indicada en el atributo data-target del botón.
  button.addEventListener('click', () => setActiveView(button.dataset.target));
});

// Abre el menú lateral móvil al presionar el botón correspondiente.
openDrawerBtn?.addEventListener('click', openMobileDrawer);

// Cierra el menú lateral móvil al presionar el fondo oscuro.
drawerScrim?.addEventListener('click', closeMobileDrawer);

// Inicia la cámara para el flujo de liveness.
liveness.startBtn.addEventListener('click', () => safeStartCamera(liveness));

// Detiene la cámara para el flujo de liveness.
liveness.stopBtn.addEventListener('click', () => safeStopCamera(liveness));

// Inicia la cámara para el flujo de enrolamiento.
enrollment.startBtn.addEventListener('click', () => safeStartCamera(enrollment));

// Detiene la cámara para el flujo de enrolamiento.
enrollment.stopBtn.addEventListener('click', () => safeStopCamera(enrollment));

// Ejecuta verificación de vida usando una sola imagen capturada.
liveness.imageBtn.addEventListener('click', async () => {
  // Bloquea el botón de imagen mientras se verifica.
  setLoading(liveness.imageBtn, true, 'Verificando...');

  // Bloquea el botón de video para evitar acciones simultáneas.
  setLoading(liveness.videoBtn, true, 'Espere...');

  // Captura, envía y muestra el resultado del backend.
  try {
    // Captura el crop facial normalizado como Blob JPEG.
    const blob = await captureFaceCropBlob(liveness.video, liveness.canvas);

    // Detiene la cámara y muestra el crop capturado.
    stopAfterCapture(liveness);

    // Informa al usuario que el crop será enviado al backend.
    setResult(liveness.result, 'Enviando crop facial al backend...');

    // Envía la imagen al endpoint de autenticación/liveness.
    const data = await postImage(blob, liveness.model.value, null);

    // Muestra la respuesta recibida desde el backend.
    setResult(liveness.result, data);

  // Muestra el error si falla la verificación por imagen.
  } catch (error) {
    setResult(liveness.result, `Error al verificar liveness en imagen: ${error.message}`);

  // Restaura botones y oculta acciones después del proceso.
  } finally {
    setLoading(liveness.imageBtn, false);
    setLoading(liveness.videoBtn, false);
    liveness.actionGroup.classList.add('hidden');
  }
});

// Ejecuta verificación de vida usando una secuencia de frames.
liveness.videoBtn.addEventListener('click', async () => {
  // Bloquea el botón de video mientras captura frames.
  setLoading(liveness.videoBtn, true, 'Capturando frames...');

  // Bloquea el botón de imagen para evitar acciones simultáneas.
  setLoading(liveness.imageBtn, true, 'Espere...');

  // Captura frames, los envía y muestra el resultado del backend.
  try {
    // Informa que se iniciará la captura temporal.
    setResult(liveness.result, 'Capturando 16 frames para verificación temporal...');

    // Captura una secuencia de 16 crops faciales con intervalo de 120 ms.
    const blobs = await captureFrameBatch(liveness.video, liveness.canvas, 16, 120);

    // Detiene la cámara y deja visible el último crop capturado.
    stopAfterCapture(liveness);

    // Informa que los frames serán enviados al backend.
    setResult(liveness.result, 'Enviando frames al backend...');

    // Envía los frames al endpoint de verificación temporal.
    const data = await postFrames(blobs, liveness.model.value, null);

    // Muestra la respuesta recibida desde el backend.
    setResult(liveness.result, data);

  // Muestra el error si falla la verificación por video.
  } catch (error) {
    setResult(liveness.result, `Error al verificar liveness en video: ${error.message}`);

  // Restaura botones y oculta acciones después del proceso.
  } finally {
    setLoading(liveness.videoBtn, false);
    setLoading(liveness.imageBtn, false);
    liveness.actionGroup.classList.add('hidden');
  }
});

// Registra una identidad facial después de validar liveness.
enrollment.registerBtn.addEventListener('click', async () => {
  // Obtiene el identificador de persona escrito por el usuario.
  const personId = enrollment.personId.value.trim();

  // Valida que el usuario haya ingresado un Person ID.
  if (!personId) {
    setResult(enrollment.result, 'Ingresa un Person ID antes de registrar identidad.');
    enrollment.personId.focus();
    return;
  }

  // Bloquea el botón mientras se valida la prueba de vida.
  setLoading(enrollment.registerBtn, true, 'Validando vida...');

  // Captura el rostro, valida liveness y luego registra la identidad.
  try {
    // Captura el crop facial normalizado como Blob JPEG.
    const blob = await captureFaceCropBlob(enrollment.video, enrollment.canvas);

    // Detiene la cámara y muestra el crop capturado.
    stopAfterCapture(enrollment);

    // Informa que primero se validará liveness.
    setResult(enrollment.result, 'Validando liveness antes del enrolamiento...');

    // Envía el crop al backend para validar que sea una muestra viva.
    const livenessData = await postImage(blob, enrollment.model.value, null);

    // Rechaza el enrolamiento si liveness no fue aprobado.
    if (!livenessData.access_granted || livenessData.decision !== 'ACCESS_GRANTED') {
      setResult(enrollment.result, {
        enrolled: false,
        message: 'Enrolamiento rechazado. La validación de vida no fue aprobada.',
        liveness_validation: livenessData
      });
      return;
    }

    // Cambia el estado del botón para indicar registro de identidad.
    setLoading(enrollment.registerBtn, true, 'Registrando identidad...');

    // Informa que la validación de vida fue aprobada.
    setResult(enrollment.result, 'Liveness aprobado. Registrando identidad...');

    // Envía el crop facial al endpoint de enrolamiento con el Person ID.
    const enrollData = await enrollIdentity(blob, personId);

    // Muestra el resultado final del enrolamiento.
    setResult(enrollment.result, {
      enrolled: true,
      message: 'Identidad registrada correctamente después de validar liveness.',
      liveness_validation: livenessData,
      enrollment: enrollData
    });

  // Muestra error si falla el registro de identidad.
  } catch (error) {
    setResult(enrollment.result, `Error al registrar identidad: ${error.message}`);

  // Restaura el botón y oculta acciones después del proceso.
  } finally {
    setLoading(enrollment.registerBtn, false);
    enrollment.actionGroup.classList.add('hidden');
  }
});

// Inicializa el flujo de liveness en estado inactivo.
setFlowState(liveness, 'idle');

// Inicializa el flujo de enrolamiento en estado inactivo.
setFlowState(enrollment, 'idle');