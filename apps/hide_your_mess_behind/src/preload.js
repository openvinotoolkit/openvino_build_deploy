const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  ipcRenderer: {
    send: (channel, data) => ipcRenderer.send(channel, data),
    on: (channel, func) => ipcRenderer.on(channel, (event, ...args) => func(event, ...args))
  },
  detectDevices: () => ipcRenderer.invoke('detect-devices'),
  runModel: (img, width, height, device) => ipcRenderer.invoke('run-model', img, width, height, device),
  blurImage: (image, width, height) => ipcRenderer.invoke('blur-image', image, width, height),
  removeBackground: (image, width, height) => ipcRenderer.invoke('remove-background', image, width, height),
  addWatermark: (image, width, height) => ipcRenderer.invoke('add-watermark', image, width, height),
  clearWatermarkCache: () => ipcRenderer.invoke('clear-watermark-cache')  
});