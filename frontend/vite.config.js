import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),

  ],
  resolve: {
    alias: {
      '@pages': '/src/pages',
      '@services': '/src/services',
      '@components': '/src/components',
      '@assets': '/src/assets'
    }
  },
  server: {
    host: true,
    port: 3000
  }
})
