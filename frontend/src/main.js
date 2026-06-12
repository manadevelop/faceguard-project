import {
  startCamera,
  stopCamera,
  captureFaceCropBlob,
  captureFrameBatch
} from './face/faceCropper.js';
import { postImage, postFrames, enrollIdentity } from './services/authApi.js';

const cameraStatusChip = document.getElementById('cameraStatusChip');
const sidebar = document.getElementById('sidebar');
const drawerScrim = document.getElementById('drawerScrim');
const openDrawerBtn = document.getElementById('openDrawerBtn');

const views = [...document.querySelectorAll('.view')];
const navButtons = [...document.querySelectorAll('.nav-item')];

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

function setResult(target, value) {
  target.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

function setLoading(button, isLoading, loadingText = 'Procesando...') {
  if (!button) return;

  if (isLoading) {
    button.dataset.originalText = button.textContent;
    button.textContent = loadingText;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
    button.disabled = false;
  }
}

function updateGlobalCameraChip(isActive) {
  cameraStatusChip.textContent = isActive ? 'Cámara activa' : 'Cámara detenida';
  cameraStatusChip.classList.toggle('status-active', isActive);
  cameraStatusChip.classList.toggle('status-muted', !isActive);
}

function setFlowState(ctx, state) {
  ctx.card.dataset.state = state;
  ctx.view?.classList.remove('flow-state-idle', 'flow-state-camera', 'flow-state-captured');
  ctx.view?.classList.add(`flow-state-${state}`);

  const isIdle = state === 'idle';
  const isCamera = state === 'camera';
  const isCaptured = state === 'captured';

  ctx.video.classList.toggle('hidden', !isCamera);
  ctx.canvas.classList.toggle('hidden', !isCaptured);
  ctx.idleBox.classList.toggle('hidden', !isIdle);

  ctx.startBtn.classList.toggle('hidden', isCamera);
  ctx.stopBtn.classList.toggle('hidden', !isCamera);
  ctx.actionGroup.classList.toggle('hidden', !isCamera);

  ctx.startBtn.disabled = isCamera;
  ctx.stopBtn.disabled = !isCamera;

  if (isCaptured) {
    ctx.title.textContent = 'Crop facial enviado';
    ctx.subtitle.textContent = 'Rostro normalizado a 224x224.';
  } else {
    ctx.title.textContent = 'Cámara en vivo';
    ctx.subtitle.textContent = 'Vista completa de la webcam. No se envía completa al backend.';
  }

  updateGlobalCameraChip(isCamera);
}

function closeMobileDrawer() {
  sidebar.classList.remove('drawer-open');
  drawerScrim.classList.remove('visible');
  drawerScrim.setAttribute('aria-hidden', 'true');
}

function openMobileDrawer() {
  sidebar.classList.add('drawer-open');
  drawerScrim.classList.add('visible');
  drawerScrim.setAttribute('aria-hidden', 'false');
}

async function safeStartCamera(ctx) {
  try {
    await startCamera(ctx.video);
    setFlowState(ctx, 'camera');
    setResult(ctx.result, 'Cámara iniciada. Ubica el rostro dentro de la vista y ejecuta la acción.');
  } catch (error) {
    setFlowState(ctx, 'idle');
    setResult(ctx.result, `No se pudo iniciar la cámara: ${error.message}`);
  }
}

function safeStopCamera(ctx, options = { resetResult: true }) {
  try {
    stopCamera(ctx.video);
  } finally {
    setFlowState(ctx, 'idle');
    if (options.resetResult) {
      setResult(ctx.result, 'Cámara detenida.');
    }
  }
}

function stopAfterCapture(ctx) {
  try {
    stopCamera(ctx.video);
  } finally {
    setFlowState(ctx, 'captured');
  }
}

function setActiveView(targetId) {
  if (liveness.video.srcObject) safeStopCamera(liveness, { resetResult: false });
  if (enrollment.video.srcObject) safeStopCamera(enrollment, { resetResult: false });

  views.forEach((view) => {
    view.classList.toggle('active-view', view.id === targetId);
  });

  navButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.target === targetId);
  });

  closeMobileDrawer();
}

navButtons.forEach((button) => {
  button.addEventListener('click', () => setActiveView(button.dataset.target));
});

openDrawerBtn?.addEventListener('click', openMobileDrawer);
drawerScrim?.addEventListener('click', closeMobileDrawer);

liveness.startBtn.addEventListener('click', () => safeStartCamera(liveness));
liveness.stopBtn.addEventListener('click', () => safeStopCamera(liveness));

enrollment.startBtn.addEventListener('click', () => safeStartCamera(enrollment));
enrollment.stopBtn.addEventListener('click', () => safeStopCamera(enrollment));

liveness.imageBtn.addEventListener('click', async () => {
  setLoading(liveness.imageBtn, true, 'Verificando...');
  setLoading(liveness.videoBtn, true, 'Espere...');

  try {
    const blob = await captureFaceCropBlob(liveness.video, liveness.canvas);
    stopAfterCapture(liveness);
    setResult(liveness.result, 'Enviando crop facial al backend...');
    const data = await postImage(blob, liveness.model.value, null);
    setResult(liveness.result, data);
  } catch (error) {
    setResult(liveness.result, `Error al verificar liveness en imagen: ${error.message}`);
  } finally {
    setLoading(liveness.imageBtn, false);
    setLoading(liveness.videoBtn, false);
    liveness.actionGroup.classList.add('hidden');
  }
});

liveness.videoBtn.addEventListener('click', async () => {
  setLoading(liveness.videoBtn, true, 'Capturando frames...');
  setLoading(liveness.imageBtn, true, 'Espere...');

  try {
    setResult(liveness.result, 'Capturando 16 frames para verificación temporal...');
    const blobs = await captureFrameBatch(liveness.video, liveness.canvas, 16, 120);
    stopAfterCapture(liveness);
    setResult(liveness.result, 'Enviando frames al backend...');
    const data = await postFrames(blobs, liveness.model.value, null);
    setResult(liveness.result, data);
  } catch (error) {
    setResult(liveness.result, `Error al verificar liveness en video: ${error.message}`);
  } finally {
    setLoading(liveness.videoBtn, false);
    setLoading(liveness.imageBtn, false);
    liveness.actionGroup.classList.add('hidden');
  }
});

enrollment.registerBtn.addEventListener('click', async () => {
  const personId = enrollment.personId.value.trim();

  if (!personId) {
    setResult(enrollment.result, 'Ingresa un Person ID antes de registrar identidad.');
    enrollment.personId.focus();
    return;
  }

  setLoading(enrollment.registerBtn, true, 'Registrando...');

  try {
    const blob = await captureFaceCropBlob(enrollment.video, enrollment.canvas);
    stopAfterCapture(enrollment);
    setResult(enrollment.result, 'Enviando identidad al backend...');
    const data = await enrollIdentity(blob, personId);
    setResult(enrollment.result, data);
  } catch (error) {
    setResult(enrollment.result, `Error al registrar identidad: ${error.message}`);
  } finally {
    setLoading(enrollment.registerBtn, false);
    enrollment.actionGroup.classList.add('hidden');
  }
});

setFlowState(liveness, 'idle');
setFlowState(enrollment, 'idle');
