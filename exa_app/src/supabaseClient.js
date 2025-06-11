// Import the Supabase client from CDN
// Note: This file is loaded after the Supabase client from CDN

// Import environment variables
const { supabaseUrl, supabaseAnonKey } = require('./config');

// Create a single supabase client for interacting with your database
const supabase = supabase.createClient(supabaseUrl, supabaseAnonKey);

console.log('Supabase client initialized with URL:', supabaseUrl);

// Authentication functions
const auth = {
  /**
   * Sign up a new user
   * @param {string} email - User's email
   * @param {string} password - User's password
   * @returns {Promise} - Supabase auth response
   */
  signUp: async (email, password) => {
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
      });
      
      if (error) throw error;
      
      // Store session if successful
      if (data?.session) {
        localStorage.setItem('supabase.auth.token', JSON.stringify(data.session));
        window.electronAPI.storeSession(data.session);
      }
      
      return { data, error: null };
    } catch (error) {
      console.error('Error signing up:', error.message);
      return { data: null, error };
    }
  },
  
  /**
   * Sign in an existing user
   * @param {string} email - User's email
   * @param {string} password - User's password
   * @returns {Promise} - Supabase auth response
   */
  signIn: async (email, password) => {
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      
      if (error) throw error;
      
      // Store session if successful
      if (data?.session) {
        localStorage.setItem('supabase.auth.token', JSON.stringify(data.session));
        window.electronAPI.storeSession(data.session);
      }
      
      return { data, error: null };
    } catch (error) {
      console.error('Error signing in:', error.message);
      return { data: null, error };
    }
  },
  
  /**
   * Sign out the current user
   * @returns {Promise} - Supabase sign out response
   */
  signOut: async () => {
    try {
      const { error } = await supabase.auth.signOut();
      
      if (error) throw error;
      
      // Clear local storage
      localStorage.removeItem('supabase.auth.token');
      
      // Notify main process
      window.electronAPI.logout();
      
      return { error: null };
    } catch (error) {
      console.error('Error signing out:', error.message);
      return { error };
    }
  },
  
  /**
   * Get the current session
   * @returns {Object|null} - Current session or null
   */
  getSession: () => {
    try {
      const sessionStr = localStorage.getItem('supabase.auth.token');
      return sessionStr ? JSON.parse(sessionStr) : null;
    } catch (error) {
      console.error('Error getting session:', error.message);
      return null;
    }
  },
  
  /**
   * Check if user is authenticated
   * @returns {boolean} - True if authenticated
   */
  isAuthenticated: () => {
    const session = auth.getSession();
    return !!session && new Date(session.expires_at) > new Date();
  }
};

// Make supabase and auth available globally
window.supabase = supabase;
window.supabaseAuth = auth;
