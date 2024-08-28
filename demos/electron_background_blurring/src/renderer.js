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

  const videoElement = document.createElement('video');
  const canvasElement = document.createElement('canvas');
  const ctx = canvasElement.getContext('2d');
  const imgElement = document.getElementById('webcam');
  const deviceSelect = document.getElementById("deviceSelect")
  const toggleWebcamButton = document.getElementById('toggleWebcamButton');
  let webcamStream = null;
  let captureInterval = null;
  let inferenceTime = null;
  let begin = null;
  let endTime = null;
  let processingActive = false;

  toggleWebcamButton.addEventListener('click', () => {
    if (webcamStream) {
      stopWebcam(false);
      webcamStream = null;
    } else {
      startWebcam(webcamSelect.value);
    }
  });

  let tempImg = null;

  async function startWebcam(deviceId) {
    try {
      let ovDevice = null;
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

      videoElement.play();
      processingActive = true; 

      async function captureFrame() {
        if (!processingActive) return; 
        try {
          begin = await window.electronAPI.takeTime();

          ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
          const imageData = ctx.getImageData(0, 0, canvasElement.width, canvasElement.height);
          ovDevice = deviceSelect.value;

          const result = await window.electronAPI.runModel(imageData, canvasElement.width, canvasElement.height, ovDevice);
          inferenceTime = result.inferenceTime;
          tempImg = result.img;

          ctx.putImageData(tempImg,0,0);

          imgElement.src = canvasElement.toDataURL('image/png');
          document.getElementById('processingTime').innerText = `Inference time: ${inferenceTime} ms (${(1000 / inferenceTime).toFixed(1)} FPS)`;

          endTime = await window.electronAPI.takeTime();

          const delay = Math.max(0, 100 - (endTime - begin));
          if (processingActive) {
            setTimeout(captureFrame, delay);
          }
        } catch (error) {
          console.error('Error during capture:', error);
        }
      }

      captureFrame();
      toggleWebcamButton.textContent = 'Stop';
    } catch (error) {
      console.error('Error accessing webcam:', error);
    }
  }

  function stopWebcam(keepActive) {
    processingActive = false; 
    clearInterval(captureInterval);
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
    }
    videoElement.srcObject = null;
    imgElement.src = '../assets/webcam_placeholder.png';
    document.getElementById('processingTime').innerText = `Inference time: 0 ms (0 FPS)`;
    if (!keepActive) {
      toggleWebcamButton.textContent = 'Start';
    }
  }
});

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
