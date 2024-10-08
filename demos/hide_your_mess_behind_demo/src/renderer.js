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
  const toggleSwitch = document.getElementById('toggleSwitch');
  const toggleValue = document.getElementById('toggleValue');
  const processingTimeElement = document.getElementById('processingTime');
  // streaming:
  let webcamStream = null;
  let captureInterval = null;
  // collecting inference results:
  let resultMask = null;
  let inferenceTime = 0;
  let tempImg = null;

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
  toggleSwitch.addEventListener('change', () => {
    toggleValue.textContent = toggleSwitch.checked ? 'on' : 'off';
    inferenceActive = toggleSwitch.checked;
  });


  // INFERENCE MANAGING
  async function getMask(imageData, canvasElement, ovDevice){
    resultMask = await window.electronAPI.runModel(imageData, canvasElement.width, canvasElement.height, ovDevice);
    inferenceTime = resultMask.inferenceTime;
  }


  // CAPTURING FRAMES
  async function processFrame() {
    if (!streamingActive) return;

    let ovDevice = deviceSelect.value;
    try {
      ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
      const imageData = ctx.getImageData(0, 0, canvasElement.width, canvasElement.height);

      if (inferenceActive) {
        await getMask(imageData, canvasElement, ovDevice);
        const result = await window.electronAPI.blurImage(imageData, canvasElement.width, canvasElement.height);
        tempImg = new ImageData(result.img, result.width, result.height);
        ctx.putImageData(tempImg, 0, 0);
        processingTimeElement.innerText = `Inference time: ${inferenceTime} ms (${(1000 / inferenceTime).toFixed(1)} FPS)`;
      } else {
        processingTimeElement.innerText = `Inference OFF`;
      }

      imgElement.src = canvasElement.toDataURL('image/jpeg');

      requestAnimationFrame(processFrame);
    } catch (error) {
      console.error('Error during capture:', error);
    }
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

    clearInterval(captureInterval);

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