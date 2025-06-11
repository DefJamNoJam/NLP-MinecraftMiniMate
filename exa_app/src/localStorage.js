/**
 * localStorage.js
 * Utility functions for managing localStorage and app settings
 */

const storageKeys = {
  AUTH_TOKEN: 'supabase.auth.token',
  THEME: 'app.theme',
  WINDOW_STYLE: 'app.windowStyle'
};

// Default settings
const defaults = {
  theme: 'dark',
  windowStyle: 'normal'
};

/**
 * Get auth session from localStorage
 * @returns {Object|null} The session object or null
 */
function getAuthSession() {
  try {
    const sessionStr = localStorage.getItem(storageKeys.AUTH_TOKEN);
    return sessionStr ? JSON.parse(sessionStr) : null;
  } catch (error) {
    console.error('Error getting auth session:', error);
    return null;
  }
}

/**
 * Save auth session to localStorage
 * @param {Object} session - The session object to save
 */
function saveAuthSession(session) {
  try {
    localStorage.setItem(storageKeys.AUTH_TOKEN, JSON.stringify(session));
    // Also notify the main process
    if (window.electronAPI) {
      window.electronAPI.storeSession(session);
    }
  } catch (error) {
    console.error('Error saving auth session:', error);
  }
}

/**
 * Clear auth session from localStorage
 */
function clearAuthSession() {
  localStorage.removeItem(storageKeys.AUTH_TOKEN);
  // Also notify the main process
  if (window.electronAPI) {
    window.electronAPI.logout();
  }
}

/**
 * Get theme preference
 * @returns {string} The theme ('dark' or 'light')
 */
function getTheme() {
  return localStorage.getItem(storageKeys.THEME) || defaults.theme;
}

/**
 * Save theme preference
 * @param {string} theme - The theme to save ('dark' or 'light')
 */
function saveTheme(theme) {
  localStorage.setItem(storageKeys.THEME, theme);
  // Also notify the main process
  if (window.electronAPI) {
    window.electronAPI.setTheme(theme);
  }
  // Apply theme to document
  document.documentElement.setAttribute('data-theme', theme);
}

/**
 * Get window style preference
 * @returns {string} The window style ('normal' or 'translucent')
 */
function getWindowStyle() {
  return localStorage.getItem(storageKeys.WINDOW_STYLE) || defaults.windowStyle;
}

/**
 * Save window style preference
 * @param {string} style - The window style to save ('normal' or 'translucent')
 */
function saveWindowStyle(style) {
  localStorage.setItem(storageKeys.WINDOW_STYLE, style);
  // Also notify the main process
  if (window.electronAPI) {
    window.electronAPI.setWindowStyle(style);
  }
}

/**
 * Apply saved theme to document
 */
function applyTheme() {
  const theme = getTheme();
  document.documentElement.setAttribute('data-theme', theme);
}

// Export all functions
module.exports = {
  getAuthSession,
  saveAuthSession,
  clearAuthSession,
  getTheme,
  saveTheme,
  getWindowStyle,
  saveWindowStyle,
  applyTheme,
  storageKeys,
  defaults
};
