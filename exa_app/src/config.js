// Load environment variables
require('dotenv').config({ path: require('path').resolve(__dirname, '../.env') });

// Export environment variables
module.exports = {
  supabaseUrl: process.env.VITE_SUPABASE_URL.replace(/'/g, ''), // Remove single quotes if present
  supabaseAnonKey: process.env.VITE_SUPABASE_ANON_KEY.replace(/'/g, '') // Remove single quotes if present
};
