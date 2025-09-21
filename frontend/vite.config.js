import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import svgr from 'vite-plugin-svgr';

export default defineConfig(({ command, mode }) => {
  const isProduction = mode === 'production';
  
  return {
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
      // Completely disable HMR in production
      hmr: isProduction ? false : {
        port: 24678,
        host: '0.0.0.0'
      }
    },
    build: {
      // Disable source maps in production
      sourcemap: false,
      // Enable minification in production
      minify: isProduction ? 'terser' : false,
      target: 'es2020',
      emptyOutDir: true,
      // Conservative terser options that preserve React
      terserOptions: isProduction ? {
        compress: {
          drop_console: false, // Keep console for debugging
          drop_debugger: true,
          // Don't remove function calls that might be needed
          pure_funcs: [],
        },
        mangle: false, // Disable mangling to preserve React internals
        format: {
          comments: false,
        },
      } : {},
      rollupOptions: {
        output: {
          entryFileNames: isProduction ? 'assets/[name]-[hash].js' : '[name].js',
          chunkFileNames: isProduction ? 'assets/[name]-[hash].js' : '[name].js',
          assetFileNames: isProduction ? 'assets/[name]-[hash].[ext]' : '[name].[ext]',
          // Safe manual chunks that preserve React
          manualChunks: isProduction ? {
            vendor: ['react', 'react-dom'],
            router: ['react-router-dom'],
          } : undefined,
        },
      },
    },
    define: {
      'process.env.NODE_ENV': JSON.stringify(mode),
      // Safely disable HMR without breaking React
      'import.meta.hot': isProduction ? false : 'import.meta.hot',
    },
    logLevel: 'info',
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: ['./src/test-setup.js']
    }
  };
});