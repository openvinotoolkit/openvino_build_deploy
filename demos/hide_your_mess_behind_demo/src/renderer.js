document.addEventListener('DOMContentLoaded', () => {

  const webcamSelect = document.getElementById('webcamSelect');

  updateDeviceSelect();
  updateWebcamSelect();

  webcamSelect.addEventListener('change', () => {
    if (webcamStream) {
      stopWebcam(true);
      startWebcam(webcamSelect.value);
    }
  });

  // VARIABLES:
  // UI elements:
  const videoElement = document.createElement('video');
  const canvasElement = document.createElement('canvas');
  const ctx = canvasElement.getContext('2d');
  const imgElement = document.getElementById('webcam');
  const deviceSelect = document.getElementById("deviceSelect")
  const toggleWebcamButton = document.getElementById('toggleWebcamButton');
  const toggleSwitch = document.getElementById('toggleSwitch');
  const toggleValue = document.getElementById('toggleValue');
  // streaming:
  let webcamStream = null;
  let captureInterval = null;
  // collecting inference results:
  let resultMask = null;
  let inferenceTime = null;
  let tempImg = null;
  // semaphores:
  let processingMask = false;
  let processingActive = false;
  let processingOn = true;


  // START/STOP BUTTON
  toggleWebcamButton.addEventListener('click', () => {
    if (webcamStream) {
      stopWebcam(false);
    } else {
      startWebcam(webcamSelect.value);
    }
  });

  // ON/OF INFERENCE BUTTON
  toggleSwitch.addEventListener('change', () => {
    toggleValue.textContent = toggleSwitch.checked ? 'on' : 'off';
  });


  // START STREAMING
  async function startWebcam(deviceId) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: deviceId,
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: false
      });
      
      webcamStream = stream;
      videoElement.srcObject = stream;

      videoElement.addEventListener('loadedmetadata', () => {
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
      });

      await videoElement.play();
      processingActive = true;

      toggleWebcamButton.textContent = 'Stop';

      await captureFrame();
    } catch (error) {
      console.error('Error accessing webcam:', error);
    }
  }


  // INFERENCE MANAGING
  async function processMask(imageData, canvasElement, ovDevice){
    processingMask = true;
    resultMask = await window.electronAPI.runModel(imageData, canvasElement.width, canvasElement.height, ovDevice);
    processingMask = false;
  }


  // CAPTURING FRAMES
  async function captureFrame() {
    if (!processingActive) return;
    let ovDevice= deviceSelect.value;
    try {
      ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
      const imageData = ctx.getImageData(0, 0, canvasElement.width, canvasElement.height);
      
      if (!processingMask && processingOn){
        processMask(imageData, canvasElement, ovDevice);
      }

      if (toggleSwitch.checked){
        processingOn = true;

        const result = await window.electronAPI.blurImage(imageData, canvasElement.width, canvasElement.height);
        tempImg = new ImageData(result.img, result.width, result.height);
        ctx.putImageData(tempImg, 0, 0);
        inferenceTime = resultMask.inferenceTime;
        document.getElementById('processingTime').innerText = `Inference time: ${inferenceTime} ms (${(1000 / inferenceTime).toFixed(1)} FPS)`;
      } else {
        document.getElementById('processingTime').innerText = `Inference OFF`;
        processingOn = false;
      }
      imgElement.src = canvasElement.toDataURL('image/jpeg');
      if (processingActive) {
        setTimeout(captureFrame, 0);
      }
    } catch (error) {
      console.error('Error during capture:', error);
    }
  }


  // STOP STREAMING
  function stopWebcam(keepActive) {
    processingActive = false; 
    clearInterval(captureInterval);
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
    }

    videoElement.srcObject = null;
    imgElement.src = '../assets/webcam_placeholder.png';
    document.getElementById('processingTime').innerText = `Streaming stopped`;
    webcamStream = null;

    if (!keepActive) {
      toggleWebcamButton.textContent = 'Start';
    }
  }
});


// GETTING THE LIST OF AVAILABLE DEVICES
function updateDeviceSelect() {
  const deviceSelect = document.getElementById('deviceSelect');

  window.electronAPI.detectDevices().then(devices =>
    devices.forEach(device => {
      const option = document.createElement('option');
      option.value = device;
      option.text = device;
      deviceSelect.appendChild(option);
    })
  );
}


// GETTING THE LIST OF AVAILABLE WEBCAMS
function updateWebcamSelect() {
  const webcamSelect = document.getElementById('webcamSelect');

  navigator.mediaDevices.enumerateDevices().then(devices =>
    devices.forEach(device => {
      if (device.kind === "videoinput") {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.text = device.label || `Camera ${device.deviceId}`;
        webcamSelect.appendChild(option);
      }
    })
  );
}
