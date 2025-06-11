/**
 * errorHandler.js
 * Comprehensive error handling and logging system for the application
 */

// Error categories
const ErrorCategory = {
  AUTH: 'auth',
  API: 'api',
  APP: 'app',
  NETWORK: 'network',
  UNKNOWN: 'unknown'
};

// Log levels
const LogLevel = {
  INFO: 'info',
  WARNING: 'warning',
  ERROR: 'error',
  DEBUG: 'debug'
};

// Store logs in memory
const logs = [];
const MAX_LOGS = 1000;

/**
 * Log an event with specified level
 * @param {string} level - Log level (info, warning, error, debug)
 * @param {string} message - Log message
 * @param {Object} data - Additional data to log
 */
function log(level, message, data = {}) {
  const timestamp = new Date().toISOString();
  const logEntry = {
    timestamp,
    level,
    message,
    data
  };
  
  // Add to in-memory logs
  logs.unshift(logEntry);
  
  // Trim logs if they exceed max size
  if (logs.length > MAX_LOGS) {
    logs.length = MAX_LOGS;
  }
  
  // Log to console in development mode
  if (process.env.NODE_ENV !== 'production') {
    const consoleMethod = level === LogLevel.ERROR ? 'error' 
                        : level === LogLevel.WARNING ? 'warn'
                        : level === LogLevel.DEBUG ? 'debug'
                        : 'log';
    console[consoleMethod](`[${timestamp}] [${level.toUpperCase()}] ${message}`, data);
  }
}

/**
 * Log an info message
 * @param {string} message - Log message
 * @param {Object} data - Additional data to log
 */
function logInfo(message, data = {}) {
  log(LogLevel.INFO, message, data);
}

/**
 * Log a warning message
 * @param {string} message - Log message
 * @param {Object} data - Additional data to log
 */
function logWarning(message, data = {}) {
  log(LogLevel.WARNING, message, data);
}

/**
 * Log an error message
 * @param {string} message - Log message
 * @param {Object} data - Additional data to log
 */
function logError(message, data = {}) {
  log(LogLevel.ERROR, message, data);
}

/**
 * Log a debug message
 * @param {string} message - Log message
 * @param {Object} data - Additional data to log
 */
function logDebug(message, data = {}) {
  log(LogLevel.DEBUG, message, data);
}

/**
 * Get all logs
 * @param {string} level - Optional filter by log level
 * @param {number} limit - Maximum number of logs to return
 * @returns {Array} - Array of log entries
 */
function getLogs(level = null, limit = 100) {
  let filteredLogs = logs;
  
  if (level) {
    filteredLogs = logs.filter(log => log.level === level);
  }
  
  return filteredLogs.slice(0, limit);
}

/**
 * Categorize an error
 * @param {Error} error - Error object
 * @returns {string} - Error category
 */
function categorizeError(error) {
  if (!error) return ErrorCategory.UNKNOWN;
  
  const message = error.message || '';
  
  if (message.includes('authentication') || message.includes('auth') || 
      message.includes('login') || message.includes('permission')) {
    return ErrorCategory.AUTH;
  }
  
  if (message.includes('API') || message.includes('server') || 
      message.includes('response') || message.includes('endpoint')) {
    return ErrorCategory.API;
  }
  
  if (message.includes('network') || message.includes('connection') || 
      message.includes('offline') || message.includes('fetch')) {
    return ErrorCategory.NETWORK;
  }
  
  return ErrorCategory.APP;
}

/**
 * Handle an error and perform appropriate actions
 * @param {Error} error - Error object
 * @param {Object} options - Options for error handling
 * @returns {Object} - Error handling result
 */
function handleError(error, options = {}) {
  const { 
    showUser = true,
    logToSystem = true,
    category = null,
    context = {}
  } = options;
  
  // Determine error category
  const errorCategory = category || categorizeError(error);
  
  // Log error
  if (logToSystem) {
    logError(error.message || 'Unknown error', {
      category: errorCategory,
      stack: error.stack,
      context,
      originalError: error
    });
  }
  
  // Create user-friendly message
  let userMessage = 'An unexpected error occurred. Please try again.';
  
  switch (errorCategory) {
    case ErrorCategory.AUTH:
      userMessage = 'Authentication error. Please log in again.';
      break;
    case ErrorCategory.API:
      userMessage = 'Server error. Please try again later.';
      break;
    case ErrorCategory.NETWORK:
      userMessage = 'Network error. Please check your connection.';
      break;
  }
  
  // Perform category-specific recovery actions
  let recoveryAction = null;
  
  switch (errorCategory) {
    case ErrorCategory.AUTH:
      recoveryAction = () => {
        // Clear auth session and redirect to login
        const storage = require('./localStorage');
        storage.clearAuthSession();
        if (window.electronAPI) {
          window.electronAPI.navigateTo('login.html');
        }
      };
      break;
    case ErrorCategory.NETWORK:
      recoveryAction = () => {
        // Offer to retry the action
        return { canRetry: true };
      };
      break;
  }
  
  return {
    category: errorCategory,
    message: userMessage,
    showUser,
    recoveryAction,
    originalError: error
  };
}

/**
 * Display an error message to the user
 * @param {string} message - Error message to display
 * @param {string} elementId - ID of the element to display the error in
 * @param {number} timeout - Time in ms to display the error (0 for no timeout)
 */
function showErrorMessage(message, elementId = 'error-message', timeout = 5000) {
  const errorElement = document.getElementById(elementId);
  if (!errorElement) return;
  
  errorElement.textContent = message;
  errorElement.classList.add('error-message--visible');
  
  if (timeout > 0) {
    setTimeout(() => {
      errorElement.classList.remove('error-message--visible');
    }, timeout);
  }
}

/**
 * Global error handler for uncaught exceptions
 * @param {Error} error - Error object
 */
function globalErrorHandler(error) {
  const result = handleError(error, {
    showUser: true,
    logToSystem: true
  });
  
  logError('Uncaught exception', {
    message: error.message,
    stack: error.stack
  });
  
  // In development, log to console
  if (process.env.NODE_ENV !== 'production') {
    console.error('Uncaught exception:', error);
  }
  
  return result;
}

// Set up global error handling
if (typeof window !== 'undefined') {
  window.addEventListener('error', (event) => {
    globalErrorHandler(event.error);
  });
  
  window.addEventListener('unhandledrejection', (event) => {
    globalErrorHandler(event.reason);
  });
}

// Export all functions and constants
module.exports = {
  ErrorCategory,
  LogLevel,
  log,
  logInfo,
  logWarning,
  logError,
  logDebug,
  getLogs,
  categorizeError,
  handleError,
  showErrorMessage,
  globalErrorHandler
};
