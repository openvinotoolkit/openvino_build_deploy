document.addEventListener('DOMContentLoaded', () => {

  updateDeviceSelect();

  const webcamOpts = {
    width: 1280,
    height: 720,
    quality: 100,
    output: "jpeg",
    device: false, 
    callbackReturn: "buffer",
    verbose: false
  };

  const webcamElement = document.getElementById('webcam');
  const toggleWebcamButton = document.getElementById('toggleWebcamButton');
  let webcamStream = null;
  let captureInterval = null;

  toggleWebcamButton.addEventListener('click', () => {
    if (webcamStream) {
      stopWebcam(webcamElement, toggleWebcamButton);
      webcamStream = null;
    } else {
      startWebcam(webcamElement, webcamOpts, stream => {
        webcamStream = stream;
        toggleWebcamButton.textContent = 'Stop';
      });
    }
  });

  async function startWebcam(videoElement, webcamOpts, onStreamReady) {
    var Webcam = await window.electronAPI.NodeWebcam.create(webcamOpts);

    function captureFrame() {
      videoElement.src = window.electronAPI.runWebcam(Webcam, inferenceOn);
    }

    captureInterval = setInterval(captureFrame, 1000 / 30);

    if (typeof onStreamReady === 'function') {
      onStreamReady(Webcam);
    }
  }

  function stopWebcam(videoElement, buttonElement) {
    if (captureInterval) {
      clearInterval(captureInterval);
      captureInterval = null;
    }

    videoElement.src = '';
    buttonElement.textContent = 'Start';
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

