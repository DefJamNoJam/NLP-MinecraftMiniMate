const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Navigation
  navigateTo: (page) => ipcRenderer.invoke('navigate', page),
  
  // Authentication
  checkAuth: () => ipcRenderer.invoke('check-auth'),
  logout: () => ipcRenderer.send('logout'),
  
  // Settings
  setTheme: (theme) => ipcRenderer.send('set-theme', theme),
  setWindowStyle: (style) => ipcRenderer.send('set-window-style', style),
  
  // Store data in main process
  storeSession: (session) => ipcRenderer.send('store-session', session),
  
  // Window controls
  windowControl: (action) => ipcRenderer.send(`window-${action}`),
  
  // Back navigation
  goBack: (page) => ipcRenderer.send('go-back', page),
  
  // SLM Server controls
  slmStart: () => ipcRenderer.invoke('slm-start'),
  slmStop: () => ipcRenderer.invoke('slm-stop'),
  slmStatus: () => ipcRenderer.invoke('slm-status'),
  
  // SLM Server port
  getRagPort: () => ipcRenderer.invoke('get-rag-port')
});

contextBridge.exposeInMainWorld('storageAPI', {
  getItem: (key) => localStorage.getItem(key),
  setItem: (key, value) => localStorage.setItem(key, value),
  removeItem: (key) => localStorage.removeItem(key)
});

// RAG 서버 설정 노출
contextBridge.exposeInMainWorld('config', {
  // 기본 포트 8504, 동적으로 변경 가능
  getServerConfig: async () => {
    const port = await ipcRenderer.invoke('get-rag-port');
    return {
      ragPort: port || 8504,
      apiBaseUrl: `http://localhost:${port || 8504}`
    };
  }
});
