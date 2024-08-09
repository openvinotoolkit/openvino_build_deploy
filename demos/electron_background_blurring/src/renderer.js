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
      stopWebcam(webcamElement, webcamStream, toggleWebcamButton);
      webcamStream = null;
    } else {
      startWebcam(webcamElement, stream => {
        webcamStream = stream;
        toggleWebcamButton.textContent = 'Stop';
      });
    }
  });

  function startWebcam(videoElement, onStreamReady) {
    navigator.mediaDevices.getUserMedia({ video: true , audio : false })
      .then(stream => {
        videoElement.srcObject = stream;
        onStreamReady(stream);
      })
      .catch(error => {
        console.error('Error accessing webcam:', error);
      });
  }

  function stopWebcam(videoElement, stream, buttonElement) {
    stream.getTracks().forEach(track => track.stop());
    videoElement.srcObject = null;
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

