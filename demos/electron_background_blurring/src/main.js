const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');

const { detectDevices } = require('./ovJobs')

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
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

ipcMain.on('save-photo', (event, buffer) => {
  dialog.showSaveDialog({
    title: 'Save Photo',
    defaultPath: 'photo.png',
    filters: [
      { name: 'Images', extensions: ['png'] }
    ]
  }).then(result => {
    if (!result.canceled) {
      fs.writeFile(result.filePath, buffer, (err) => {
        if (err) {
          console.error('Error saving photo:', err);
        }
      });
    }
  }).catch(err => {
    console.error('Error during save dialog:', err);
  });
});

ipcMain.handle('detect-devices', async () => {
  const devices = detectDevices();
  console.log("Devices:", devices);
  return devices;
});

