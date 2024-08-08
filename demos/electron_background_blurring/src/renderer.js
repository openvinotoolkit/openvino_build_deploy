document.addEventListener('DOMContentLoaded', () => {
  updateDeviceSelect();
  const toggleWebcamButton = document.getElementById('toggleWebcamButton');
  const webcamElement = document.getElementById('webcam');
  let webcamStream = null;

  const toggleWebcamButton2 = document.getElementById('toggleWebcamButton2');
  const webcamElement2 = document.getElementById('webcam2');
  let webcamStream2 = null;

  const takePhotoButton = document.getElementById('takePhotoButton');
  const takePhotoButton2 = document.getElementById('takePhotoButton2');

  const photoElement1 = document.getElementById('photo1');
  const photoElement2 = document.getElementById('photo2');

  const savePhotoButton1 = document.getElementById('savePhotoButton1');
  const savePhotoButton2 = document.getElementById('savePhotoButton2');

  let capturedPhotoBlob1 = null;
  let capturedPhotoBlob2 = null;



  toggleWebcamButton.addEventListener('click', () => {
    if (webcamStream) {
      stopWebcam(webcamElement, webcamStream, toggleWebcamButton, "1");
      webcamStream = null;
    } else {
      startWebcam(webcamElement, stream => {
        webcamStream = stream;
        toggleWebcamButton.textContent = 'Stop Webcam 1';
      });
    }
  });

  toggleWebcamButton2.addEventListener('click', () => {
    if (webcamStream2) {
      stopWebcam(webcamElement2, webcamStream2, toggleWebcamButton2, "2");
      webcamStream2 = null;
    } else {
      startWebcam(webcamElement2, stream => {
        webcamStream2 = stream;
        toggleWebcamButton2.textContent = 'Stop Webcam 2';
      });
    }
  });

  takePhotoButton.addEventListener('click', () => {
    takePhoto(webcamElement, photoElement1, 1);
  });

  takePhotoButton2.addEventListener('click', () => {
    takePhoto(webcamElement2, photoElement2, 2);
  });

  savePhotoButton1.addEventListener('click', () => {
    if (capturedPhotoBlob1) {
      savePhoto(capturedPhotoBlob1);
    } else {
      alert('No photo captured yet!');
    }
  });

  savePhotoButton2.addEventListener('click', () => {
    if (capturedPhotoBlob2) {
      savePhoto(capturedPhotoBlob2);
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

  function stopWebcam(videoElement, stream, buttonElement, number) {
    stream.getTracks().forEach(track => track.stop());
    videoElement.srcObject = null;
    buttonElement.textContent = 'Start Webcam ' + number;
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

