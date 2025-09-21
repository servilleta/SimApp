import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import svgr from 'vite-plugin-svgr';

export default defineConfig({
  plugins: [
    react(),
    svgr(),
    {
      name: 'history-fallback',
      configureServer(server) {
        server.middlewares.use('/api-test', (req, res, next) => {
          req.url = '/';
          next();
        });
        server.middlewares.use('/api-docs', (req, res, next) => {
          req.url = '/';
          next();
        });
        server.middlewares.use('/api-keys', (req, res, next) => {
          req.url = '/';
          next();
        });
        server.middlewares.use('/webhooks', (req, res, next) => {
          req.url = '/';
          next();
        });
        server.middlewares.use('/my-dashboard', (req, res, next) => {
          req.url = '/';
          next();
        });
        server.middlewares.use('/dashboard', (req, res, next) => {
          req.url = '/';
          next();
        });
        server.middlewares.use('/upload', (req, res, next) => {
          req.url = '/';
          next();
        });
        server.middlewares.use('/simulate', (req, res, next) => {
          req.url = '/';
          next();
        });
        server.middlewares.use('/results', (req, res, next) => {
          req.url = '/';
          next();
        });
        server.middlewares.use('/account', (req, res, next) => {
          req.url = '/';
          next();
        });
      }
    }
  ],
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: [
      'localhost',
      'frontend',
      'backend',
      'nginx',
      '127.0.0.1',
      '0.0.0.0'
    ],
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path,
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log('Proxy error:', err);
          });
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log('Proxying request to:', proxyReq.path);
          });
        }
      }
    },
    hmr: {
      port: 24678,
      host: '0.0.0.0'
    }
  },
  build: {
    // Security: Disable source maps in production
    sourcemap: process.env.NODE_ENV !== 'production',
    // Security: Enable minification in production
    minify: process.env.NODE_ENV === 'production' ? 'terser' : false,
    // Security: Terser options for code obfuscation
    terserOptions: process.env.NODE_ENV === 'production' ? {
      compress: {
        drop_console: true, // Remove console.log statements
        drop_debugger: true, // Remove debugger statements
        pure_funcs: ['console.log', 'console.info', 'console.debug'], // Remove specific console methods
      },
      mangle: {
        properties: {
          regex: /^_/, // Mangle properties starting with underscore
        },
      },
      format: {
        comments: false, // Remove comments
      },
    } : {},
    rollupOptions: {
      output: {
        // Security: Randomize chunk names to make analysis harder
        entryFileNames: process.env.NODE_ENV === 'production' ? 'assets/[name]-[hash].js' : '[name].js',
        chunkFileNames: process.env.NODE_ENV === 'production' ? 'assets/[name]-[hash].js' : '[name].js',
        assetFileNames: process.env.NODE_ENV === 'production' ? 'assets/[name]-[hash].[ext]' : '[name].[ext]',
        // Security: Split vendor chunks to make reverse engineering harder
        manualChunks: process.env.NODE_ENV === 'production' ? {
          vendor: ['react', 'react-dom'],
          router: ['react-router-dom'],
          ui: ['@emotion/react', '@emotion/styled'],
        } : undefined,
      },
    },
  },
  define: {
    __VUE_PROD_DEVTOOLS__: true,
  },
  logLevel: 'info',
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.js']
  }
}); 