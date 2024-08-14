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

  toggleWebcamButton.addEventListener('click', () => {
    if (webcamStream) {
      stopWebcam(false);
      webcamStream = null;
    } else {
      startWebcam(webcamSelect.value);
    }
  });


  function startWebcam(deviceId) {

    let ovDevice = null;
    navigator.mediaDevices.getUserMedia({ video: {
        deviceId : deviceId,
        width : {ideal: 1920},
        height : {ideal: 1080}
    },
    audio : false })
      .then(stream => {
        webcamStream = stream;
        videoElement.srcObject = stream;

        videoElement.addEventListener('loadedmetadata', () => {
          canvasElement.width = videoElement.videoWidth;
          canvasElement.height = videoElement.videoHeight;
        });

        videoElement.play().then();

        captureInterval = setInterval(() => {

          begin = window.electronAPI.takeTime();

          ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
          const imageData = ctx.getImageData(0,0,canvasElement.width, canvasElement.height);
          ovDevice = deviceSelect.value;

        // var tempImg = wCap.read();

          window.electronAPI.runModel(imageData, ovDevice).then(result => {
            inferenceTime = result.inferenceTime;
            tempImg = result.img;
          });
          tempImg = canvasElement.toDataURL('image/jpeg');
          
          imgElement.src = tempImg;
          document.getElementById('processingTime').innerText = `Processing time: ${inferenceTime} ms`;

        }, 25-(window.electronAPI.takeTime()-begin)); // number here means delay in ms

        toggleWebcamButton.textContent = 'Stop';
      }
    )
      .catch(error => {
        console.error('Error accessing webcam:', error);
      });
  }

  function stopWebcam(keepActive) {
    clearInterval(captureInterval);
    webcamStream.getTracks().forEach(track => track.stop());
    videoElement.srcObject = null;
    imgElement.src = '../assets/webcam_placeholder.png';
    document.getElementById('processingTime').innerText = `Processing time: 0 ms`; 
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