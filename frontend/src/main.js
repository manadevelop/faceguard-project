import {
  startCamera,
  stopCamera,
  captureFaceCropBlob,
  captureFrameBatch
} from './face/faceCropper.js';
import { postImage, postFrames, enrollIdentity } from './services/authApi.js';

const video = document.getElementById('video');
const cropCanvas = document.getElementById('cropCanvas');
const result = document.getElementById('result');
const modelSelect = document.getElementById('modelSelect');
const personId = document.getElementById('personId');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const captureBtn = document.getElementById('captureBtn');
const realtimeBtn = document.getElementById('realtimeBtn');
const enrollBtn = document.getElementById('enrollBtn');
const cameraStatus = document.getElementById('cameraStatus');
const navButtons = [...document.querySelectorAll('.drawer-link, .rail-item')];

function setResult(message) {
  result.textContent = typeof message === 'string' ? message : JSON.stringify(message, null, 2);
}

function setCameraState(isRunning) {
  startBtn.disabled = isRunning;
  stopBtn.disabled = !isRunning;
  if (cameraStatus) {
    cameraStatus.textContent = isRunning ? 'Cámara activa' : 'Cámara detenida';
    cameraStatus.classList.toggle('running', isRunning);
  }
}

function setActiveSection(targetId) {
  document.querySelectorAll('.surface-section').forEach(section => {
    section.classList.toggle('active-section', section.id === targetId);
  });
  navButtons.forEach(button => {
    button.classList.toggle('active', button.dataset.target === targetId);
  });
}

navButtons.forEach(button => {
  button.addEventListener('click', () => setActiveSection(button.dataset.target));
});

startBtn.onclick = async () => {
  try {
    await startCamera(video);
    setCameraState(true);
    setResult('Cámara iniciada. Ubica el rostro y selecciona una acción.');
  } catch (error) {
    setResult(`No se pudo iniciar la cámara: ${error.message}`);
  }
};

stopBtn.onclick = () => {
  stopCamera(video);
  setCameraState(false);
  setResult('Cámara detenida.');
};

captureBtn.onclick = async () => {
  try {
    const blob = await captureFaceCropBlob(video, cropCanvas);
    const data = await postImage(blob, modelSelect.value, personId.value.trim() || null);
    setResult(data);
  } catch (error) {
    setResult(`Error al verificar imagen: ${error.message}`);
  }
};

realtimeBtn.onclick = async () => {
  try {
    setResult('Capturando frames...');
    const blobs = await captureFrameBatch(video, cropCanvas, 16, 120);
    const data = await postFrames(blobs, modelSelect.value, personId.value.trim() || null);
    setResult(data);
  } catch (error) {
    setResult(`Error al verificar video/frames: ${error.message}`);
  }
};

enrollBtn.onclick = async () => {
  try {
    const pid = personId.value.trim();
    if (!pid) {
      alert('Ingresa un person_id para registrar.');
      return;
    }
    const blob = await captureFaceCropBlob(video, cropCanvas);
    const data = await enrollIdentity(blob, pid);
    setResult(data);
  } catch (error) {
    setResult(`Error al registrar identidad: ${error.message}`);
  }
};

window.addEventListener('beforeunload', () => stopCamera(video));
setCameraState(false);
