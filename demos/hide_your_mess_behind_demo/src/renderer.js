document.addEventListener('DOMContentLoaded', () => {
  // VARIABLES:
  // UI elements:
  const videoElement = document.createElement('video');
  const canvasElement = document.createElement('canvas');
  const ctx = canvasElement.getContext('2d');
  const imgElement = document.getElementById('webcam');
  const deviceSelect = document.getElementById("deviceSelect")
  const webcamSelect = document.getElementById('webcamSelect');
  const toggleWebcamButton = document.getElementById('toggleWebcamButton');
  const toggleInferenceSwitch = document.getElementById('toggleSwitch');
  const toggleValue = document.getElementById('toggleValue');
  const processingTimeElement = document.getElementById('processingTime');
  // streaming:
  let webcamStream = null;

  let streamingActive = false;
  let inferenceActive = false;

  updateDeviceSelect();
  updateWebcamSelect();

  webcamSelect.addEventListener('change', () => {
    if (webcamStream) {
      stopWebcam(true);
      startWebcam(webcamSelect.value);
    }
  });


  // START/STOP BUTTON
  toggleWebcamButton.addEventListener('click', () => {
    if (webcamStream) {
      stopWebcam(false);
    } else {
      startWebcam(webcamSelect.value);
    }
  });


  // ON/OF INFERENCE BUTTON
  toggleInferenceSwitch.addEventListener('change', () => {
    toggleValue.textContent = toggleInferenceSwitch.checked ? 'on' : 'off';
    inferenceActive = toggleInferenceSwitch.checked;
  });


  // CAPTURING FRAMES
const processingTimes = [];

async function processFrame() {
  if (!streamingActive) return;

  let device = deviceSelect.value;
  try {
    ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    const imageData = ctx.getImageData(0, 0, canvasElement.width, canvasElement.height);

    if (inferenceActive) {
      let resultMask = await window.electronAPI.runModel(imageData, canvasElement.width, canvasElement.height, device);            
      let result = await window.electronAPI.blurImage(imageData, canvasElement.width, canvasElement.height);
      let blurredImage = new ImageData(result.img, result.width, result.height);      
      blurredImage = new ImageData(result.img, result.width, result.height);
      
      // Add the inference time to the processing times array
      processingTimes.push(resultMask.inferenceTime / 1000); // Convert ms to seconds to match Python
      
      // Keep only last 200 frames like Python code
      if (processingTimes.length > 200) {
        processingTimes.shift();
      }
      
      // Calculate mean processing time in milliseconds
      const processingTime = (processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length) * 1000;
      const fps = 1000 / processingTime;

      ctx.putImageData(blurredImage, 0, 0);

      if (!streamingActive) return;
      processingTimeElement.innerText = `Inference time: ${processingTime.toFixed(1)}ms (${fps.toFixed(1)} FPS)`;
    } else {
      processingTimeElement.innerText = `Inference OFF`;
    }

    if (!streamingActive) return;
    imgElement.src = canvasElement.toDataURL('image/jpeg');

  } catch (error) {
    console.error('Error during capture:', error);
  }

  requestAnimationFrame(processFrame);
}

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
      streamingActive = true;

      toggleWebcamButton.textContent = 'Stop';

      requestAnimationFrame(processFrame);
    } catch (error) {
      console.error('Error accessing webcam:', error);
    }
  }


  // STOP STREAMING
  function stopWebcam(keepActive) {
    streamingActive = false;

    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      webcamStream = null;
    }

    videoElement.srcObject = null;
    imgElement.src = '../assets/webcam_placeholder.png';
    processingTimeElement.innerText = `Click START to run the demo`;

    if (!keepActive) {
      toggleWebcamButton.textContent = 'Start';
    }
  }


// GETTING THE LIST OF AVAILABLE DEVICES
  function updateDeviceSelect() {
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

});