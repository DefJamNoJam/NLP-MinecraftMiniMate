/**
 * apiClient.js
 * Handles communication with the local RAG API
 */

// API endpoint configuration
const API_CONFIG = {
  baseUrl: 'http://localhost:8504',
  endpoints: {
    query: '/query'
  },
  timeout: 30000, // 30 seconds timeout
  retryCount: 3,
  retryDelay: 1000 // 1 second delay between retries
};

/**
 * Send a query to the RAG API
 * @param {string} query - The user's question
 * @param {Object} options - Additional options
 * @returns {Promise<Object>} - The API response
 */
async function sendQuery(query, options = {}) {
  const { onStart, onProgress, onComplete, onError } = options;
  
  // Call onStart callback if provided
  if (onStart && typeof onStart === 'function') {
    onStart();
  }
  
  let retries = 0;
  let lastError = null;
  
  // Retry loop
  while (retries <= API_CONFIG.retryCount) {
    try {
      // Create AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.timeout);
      
      // Make API request
      const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.query}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query }),
        signal: controller.signal
      });
      
      // Clear timeout
      clearTimeout(timeoutId);
      
      // Handle non-OK responses
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error: ${response.status} - ${errorText}`);
      }
      
      // Parse response
      const data = await response.json();
      
      // Call onComplete callback if provided
      if (onComplete && typeof onComplete === 'function') {
        onComplete(data);
      }
      
      return data;
    } catch (error) {
      lastError = error;
      
      // Check if error is due to timeout
      if (error.name === 'AbortError') {
        lastError = new Error('Request timed out. The server took too long to respond.');
      }
      
      // Log error
      console.error(`API request failed (attempt ${retries + 1}/${API_CONFIG.retryCount + 1}):`, error);
      
      // Call onProgress callback if provided
      if (onProgress && typeof onProgress === 'function') {
        onProgress(retries, API_CONFIG.retryCount);
      }
      
      // If we've reached max retries, break out of loop
      if (retries >= API_CONFIG.retryCount) {
        break;
      }
      
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, API_CONFIG.retryDelay));
      retries++;
    }
  }
  
  // If we get here, all retries failed
  if (onError && typeof onError === 'function') {
    onError(lastError);
  }
  
  throw lastError;
}

/**
 * Check if the RAG API server is running
 * @returns {Promise<boolean>} - True if server is running
 */
async function checkServerStatus() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
    
    const response = await fetch(API_CONFIG.baseUrl, {
      method: 'GET',
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    console.error('Server status check failed:', error);
    return false;
  }
}

/**
 * Format error messages for user display
 * @param {Error} error - The error object
 * @returns {string} - User-friendly error message
 */
function formatErrorMessage(error) {
  if (!error) return 'An unknown error occurred';
  
  // Check for specific error types
  if (error.name === 'AbortError' || error.message.includes('timed out')) {
    return 'Request timed out. The server took too long to respond. Please try again.';
  }
  
  if (error.message.includes('Failed to fetch')) {
    return 'Could not connect to the RAG server. Please make sure it\'s running on port 8501.';
  }
  
  if (error.message.includes('API error')) {
    return `Server error: ${error.message}`;
  }
  
  return error.message || 'An unexpected error occurred. Please try again.';
}

// Export all functions
module.exports = {
  sendQuery,
  checkServerStatus,
  formatErrorMessage,
  API_CONFIG
};
