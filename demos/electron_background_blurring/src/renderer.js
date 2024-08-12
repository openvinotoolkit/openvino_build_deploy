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

  toggleWebcamButton.addEventListener('click', () => {
    if (webcamStream) {
      stopWebcam(false);
      webcamStream = null;
    } else {
      startWebcam(webcamSelect.value);
    }
  });

  function startWebcam(deviceId) {
    let tempImg, ovDevice = null;
    navigator.mediaDevices.getUserMedia({ video: {
        deviceId : deviceId,
        width : {ideal: 1920},
        height : {ideal: 1080}
    } })
      .then(stream => {
        webcamStream = stream;
        videoElement.srcObject = stream;

        videoElement.addEventListener('loadedmetadata', () => {
          canvasElement.width = videoElement.videoWidth;
          canvasElement.height = videoElement.videoHeight;
        });

        videoElement.play().then();

        captureInterval = setInterval(() => {
          ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
          tempImg = canvasElement.toDataURL('image/jpeg');
          ovDevice = deviceSelect.value

          var resultOfInference = window.electronAPI.runModel(tempImg, ovDevice);

          tempImg = resultOfInference.img;
          document.getElementById('processingTime').innerText = `Processing time: ${resultOfInference.inferenceTime} ms`;

          imgElement.src = tempImg
        }, 25); // number here means delay in ms

        toggleWebcamButton.textContent = 'Stop';
      })
      .catch(error => {
        console.error('Error accessing webcam:', error);
      });
  }

  function stopWebcam(keepActive) {
    clearInterval(captureInterval);
    webcamStream.getTracks().forEach(track => track.stop());
    videoElement.srcObject = null;
    imgElement.src = '../assets/webcam_placeholder.png';
    if (!keepActive){
      toggleWebcamButton.textContent = 'Start';
    }
  }
});

function updateDeviceSelect() {
  const deviceSelect = document.getElementById('deviceSelect');

  window.electronAPI.detectDevices().then( devices =>
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

  navigator.mediaDevices.enumerateDevices().then( devices =>
    devices.forEach(device => {
      if (device.kind === "videoinput") {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.text =  device.label || `Camera ${device.deviceId}`;
        webcamSelect.appendChild(option);
        }
      })
  );
}