import { defineConfig } from 'vite';

// Served from the custom domain https://azaz.zachzundel.com/ (site root).
export default defineConfig({
  base: '/',
  build: {
    target: 'es2020',
    chunkSizeWarningLimit: 2000,
  },
});
