const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const Store = require('electron-store');
const slmServer = require('./src/slm_server_starter');

// Load environment variables
require('dotenv').config({ path: path.join(__dirname, '.env') });

// Log environment variables for debugging (remove in production)
console.log('Environment variables loaded');
if (process.env.VITE_SUPABASE_URL) {
  console.log('Supabase URL:', process.env.VITE_SUPABASE_URL);
} else {
  console.warn('Supabase URL not found in environment variables');
}

// Disable hardware acceleration to prevent GPU process errors
app.disableHardwareAcceleration();
console.log('Hardware acceleration disabled');

// Initialize the store for app settings
const store = new Store();

// Force session reset on app start
store.set('session', null);
console.log('Session reset on app start');

// Keep a global reference of the window object to prevent garbage collection
let mainWindow;

// Default window settings
const defaultSettings = {
  theme: 'dark',
  windowStyle: 'normal'
};

function createWindow() {
  // Get saved settings or use defaults
  const theme = store.get('theme', defaultSettings.theme);
  const windowStyle = store.get('windowStyle', defaultSettings.windowStyle);
  
  // Check for app icon in assets folder
  let iconPath = null;
  const iconExtensions = ['ico', 'png'];
  
  for (const ext of iconExtensions) {
    const possibleIcons = fs.readdirSync(path.join(__dirname, 'assets'))
      .filter(file => file.endsWith(`.${ext}`));
    
    if (possibleIcons.length > 0) {
      iconPath = path.join(__dirname, 'assets', possibleIcons[0]);
      console.log(`Using app icon: ${iconPath}`);
      break;
    }
  }

  // Ensure translucent mode is off by default on first launch
  if (store.get('windowStyle') === undefined) {
    store.set('windowStyle', 'normal');
  }

  // Create the main browser window with minimal configuration
  mainWindow = new BrowserWindow({
    width: 600,
    height: 400,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    frame: false,
    backgroundColor: '#1a1a1a',
    show: true  // Show immediately
  });

  // Set alwaysOnTop after window is created
  mainWindow.setAlwaysOnTop(true, 'screen-saver');
  
  // Disable resizing
  mainWindow.setResizable(false);
  
  // Set window icon if available
  if (fs.existsSync(iconPath)) {
    mainWindow.setIcon(iconPath);
  }

  // Load the splash screen by default
  mainWindow.loadFile('splash.html');
  console.log('Loaded splash.html as initial page');

  // Open DevTools in development mode
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }

  // Handle window closed event
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Create window when Electron is ready
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit when all windows are closed, except on macOS
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC handlers for navigation
ipcMain.handle('navigate', async (event, page) => {
  try {
    if (!mainWindow || mainWindow.isDestroyed()) {
      throw new Error('Main window is not available');
    }
    
    console.log(`Navigating to: ${page}`);
    
    // Load the requested page
    await mainWindow.loadFile(page);
    
    // Make sure window is visible
    if (!mainWindow.isVisible()) {
      mainWindow.show();
    }
    
    return { success: true, message: `Navigated to ${page}` };
    
  } catch (error) {
    console.error('Navigation error:', error);
    return { success: false, error: error.message };
  }
});

// IPC handler for opening login page
ipcMain.on('open-login', () => {
  if (mainWindow) {
    mainWindow.loadFile('login.html');
  }
});

// IPC handler for window transparency
ipcMain.on('set-window-style', (event, style) => {
  try {
    if (!mainWindow || mainWindow.isDestroyed()) {
      return;
    }
    
    // Save the style preference
    store.set('windowStyle', style);
    
    // Update the current window's properties
    if (style === 'translucent') {
      mainWindow.setBackgroundColor('#00000000');
      mainWindow.setOpacity(0.7);
      // On Windows, we need to set transparent to true
      if (process.platform === 'win32') {
        mainWindow.setBackgroundColor('transparent');
      }
    } else {
      mainWindow.setBackgroundColor('#1a1a1a');
      mainWindow.setOpacity(1.0);
    }
    
    // Force a repaint
    mainWindow.webContents.send('window-style-changed', style);
    
  } catch (error) {
    console.error('Error changing window style:', error);
  }
});

// IPC handler for theme changes
ipcMain.on('set-theme', (event, theme) => {
  store.set('theme', theme);
});

// Check if user is authenticated
ipcMain.handle('check-auth', async () => {
  const session = store.get('session');
  console.log('Checking auth, session:', session);
  return !!session && session !== null && session !== 'null' && session !== 'undefined';
});

// Handle logout
ipcMain.on('logout', () => {
  console.log('Logging out, clearing session');
  store.delete('session');
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.loadFile('login.html').catch(err => {
      console.error('Error loading login page after logout:', err);
    });
  }
});

// Store session data
ipcMain.on('store-session', (event, session) => {
  store.set('session', session);
  console.log('Session stored:', session);
});

// Window control handlers
ipcMain.on('window-minimize', () => {
  if (mainWindow) {
    mainWindow.minimize();
  }
});

ipcMain.on('window-back-to-launch', () => {
  if (mainWindow) {
    console.log('Going back to launch.html');
    mainWindow.loadFile('launch.html');
  }
});

ipcMain.on('window-back-to-index', () => {
  if (mainWindow) {
    console.log('Going back to index.html');
    mainWindow.loadFile('index.html');
  }
});

ipcMain.on('window-close', () => {
  if (mainWindow) {
    mainWindow.close();
  }
});

ipcMain.on('go-back', (event, targetPage) => {
  console.log(`Going back to ${targetPage}`);
  if (mainWindow) {
    mainWindow.loadFile(targetPage);
  }
});

ipcMain.on('window-maximize', () => {
  if (mainWindow) {
    if (mainWindow.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow.maximize();
    }
  }
});

ipcMain.on('window-style-change', (event, style) => {
  if (mainWindow) {
    if (style === 'translucent') {
      mainWindow.setBackgroundColor('rgba(0,0,0,0)');
      if (process.platform === 'win32') {
        mainWindow.setVibrancy('dark');
      }
    } else {
      mainWindow.setBackgroundColor('#1a1a1a');
      if (process.platform === 'win32') {
        mainWindow.setVibrancy(null);
      }
    }
    store.set('windowStyle', style);
  }
});

// SLM 서버 제어 핸들러
ipcMain.handle('slm-start', async () => {
  console.log('Starting SLM server...');
  const result = await slmServer.startServer();
  return result;
});

ipcMain.handle('slm-stop', () => {
  console.log('Stopping SLM server...');
  slmServer.stopServer();
  return true;
});

ipcMain.handle('slm-status', () => {
  const status = slmServer.isRunning();
  console.log(`SLM server status: ${status ? 'running' : 'stopped'}`);
  return status;
});

// RAG 서버 포트 정보 제공
ipcMain.handle('get-rag-port', () => {
  // 환경변수에서 포트 값을 가져오거나 기본값 사용
  return process.env.RAG_PORT || 8504;
});

// 앱 종료 시 SLM 서버 정리
app.on('will-quit', () => {
  console.log('App is quitting, cleaning up SLM server...');
  slmServer.stopServer();
});
