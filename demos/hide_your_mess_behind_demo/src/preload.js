const { contextBridge, ipcRenderer } = require('electron');
const { Buffer } = require('buffer');

contextBridge.exposeInMainWorld('electronAPI', {
  ipcRenderer: {
    send: (channel, data) => ipcRenderer.send(channel, data),
    on: (channel, func) => ipcRenderer.on(channel, (event, ...args) => func(event, ...args))
  },
  Buffer: Buffer,
  detectDevices: () => ipcRenderer.invoke('detect-devices'),
  runModel: (img, width, height, device) => ipcRenderer.invoke('run-model', img, width, height, device),
  blurImage: (image, width, height) => ipcRenderer.invoke('blur-image', image, width, height),
  takeTime: () => ipcRenderer.invoke('take-time')
});