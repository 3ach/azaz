import { defineConfig } from 'vite';

// Project is served from https://<user>.github.io/azaz/
export default defineConfig({
  base: '/azaz/',
  build: {
    target: 'es2020',
    chunkSizeWarningLimit: 2000,
  },
});
