document.addEventListener('DOMContentLoaded', () => {
  updateDeviceSelect();
  const toggleWebcamButton = document.getElementById('toggleWebcamButton');
  const webcamElement = document.getElementById('webcam');
  let webcamStream = null;

  const takePhotoButton = document.getElementById('takePhotoButton');
  const takePhotoButton2 = document.getElementById('takePhotoButton2');

  const photoElement1 = document.getElementById('photo1');

  const savePhotoButton1 = document.getElementById('savePhotoButton1');

  let capturedPhotoBlob1 = null;
  let capturedPhotoBlob2 = null;



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

  takePhotoButton.addEventListener('click', () => {
    takePhoto(webcamElement, photoElement1, 1);
  });

  savePhotoButton1.addEventListener('click', () => {
    if (capturedPhotoBlob1) {
      savePhoto(capturedPhotoBlob1);
    } else {
      alert('No photo captured yet!');
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

  function takePhoto(videoElement, photoElement, webcamNumber) {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
      const url = URL.createObjectURL(blob);
      photoElement.src = url;
      if (webcamNumber === 1) {
        capturedPhotoBlob1 = blob;
      } else if (webcamNumber === 2) {
        capturedPhotoBlob2 = blob;
      }
    }, 'image/png');
  }

  function savePhoto(blob) {
    const reader = new FileReader();
    reader.onloadend = () => {
      const buffer = new window.electron.Buffer(reader.result);
      window.electron.ipcRenderer.send('save-photo', buffer);
    };
    reader.readAsArrayBuffer(blob);
  }
});

function updateDeviceSelect() {
  const deviceSelect = document.getElementById('deviceSelect');

  window.electron.detectDevices().then( devices =>
      devices.forEach(device => {
        const option = document.createElement('option');
        option.value = device;
        option.text = device;
        deviceSelect.appendChild(option);
      })
  );
}

