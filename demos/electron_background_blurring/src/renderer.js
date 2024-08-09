document.addEventListener('DOMContentLoaded', () => {

  updateDeviceSelect();

  const videoElement = document.createElement('video');
  const canvasElement = document.createElement('canvas');
  const ctx = canvasElement.getContext('2d');
  const imgElement = document.getElementById('webcam');
  const toggleWebcamButton = document.getElementById('toggleWebcamButton');
  let webcamStream = null;
  let captureInterval = null;

  toggleWebcamButton.addEventListener('click', () => {
    if (webcamStream) {
      stopWebcam();
      webcamStream = null;
    } else {
      startWebcam();
    }
  });

  function startWebcam() {
    let tempImg = null;
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        webcamStream = stream;
        videoElement.srcObject = stream;

        videoElement.addEventListener('loadedmetadata', () => {
          // Ustawiamy wymiary canvas na podstawie strumienia wideo
          canvasElement.width = videoElement.videoWidth;
          canvasElement.height = videoElement.videoHeight;
        });

        videoElement.play();

        captureInterval = setInterval(() => {
          ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
          tempImg = canvasElement.toDataURL('image/jpeg');

          imgElement.src = tempImg;
        }, 30); // number here means delay (33FPS)

        toggleWebcamButton.textContent = 'Stop';
      })
      .catch(error => {
        console.error('Error accessing webcam:', error);
      });
  }

  function stopWebcam() {
    clearInterval(captureInterval);
    webcamStream.getTracks().forEach(track => track.stop());
    videoElement.srcObject = null;
    imgElement.src = '';
    toggleWebcamButton.textContent = 'Start';
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

