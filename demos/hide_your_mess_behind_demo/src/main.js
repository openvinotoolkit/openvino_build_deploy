const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

const { detectDevices, runModel, blurImage, addWatermark } = require('./ov-jobs')

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    minWidth: 600,
    minHeight: 500,
    autoHideMenuBar: true,
    icon: path.join(__dirname, 'assets', 'icons', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      enableRemoteModule: false,
      nodeIntegration: false,
    }
  });

  mainWindow.loadFile('src/index.html');
}

app.on('ready', createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

ipcMain.handle('detect-devices', async () => {
  return detectDevices();
});

ipcMain.handle('run-model', async (event, img, width, height, device) => {
  return runModel(img, width, height, device);
})

ipcMain.handle('blur-image', async (event, image, width, height) => {
  return blurImage(image, width, height);
})

ipcMain.handle('add-watermark', async (event, image, width, height) => {
  return addWatermark(image, width, height);
})

ipcMain.handle('detect-webcam', async () => {
  return navigator.mediaDevices.enumerateDevices();
});

// ipcMain.handle('take-time', async () => {
//   return takeTime();
// });

