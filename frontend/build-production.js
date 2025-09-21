#!/usr/bin/env node

// Custom production build script that eliminates ALL Vite dev server references
const { build } = require('vite');
const react = require('@vitejs/plugin-react');
const svgr = require('vite-plugin-svgr');

async function buildProduction() {
  console.log('🚀 Building production version with NO dev server dependencies...');
  
  try {
    await build({
      plugins: [
        react(),
        svgr.default(),
        {
          name: 'remove-vite-client',
          generateBundle(options, bundle) {
            // Remove any chunks that contain Vite client code
            Object.keys(bundle).forEach(fileName => {
              const chunk = bundle[fileName];
              if (chunk.type === 'chunk' && chunk.code) {
                // Remove any references to Vite dev server
                chunk.code = chunk.code
                  .replace(/ws:\/\/[^"']*24678[^"']*/g, '')
                  .replace(/http:\/\/[^"']*24678[^"']*/g, '')
                  .replace(/@vite\/client/g, '')
                  .replace(/import\.meta\.hot/g, 'undefined')
                  .replace(/if\s*\(\s*import\.meta\.hot\s*\)/g, 'if (false)')
                  .replace(/import\.meta\.hot\.[^;]*/g, '');
              }
            });
          }
        }
      ],
      build: {
        target: 'es2020',
        emptyOutDir: true,
        minify: 'terser',
        sourcemap: false,
        terserOptions: {
          compress: {
            drop_console: true,
            drop_debugger: true,
            pure_funcs: ['console.log', 'console.info', 'console.debug'],
          },
          mangle: {
            properties: {
              regex: /^_/,
            },
          },
          format: {
            comments: false,
          },
        },
        rollupOptions: {
          output: {
            entryFileNames: 'assets/[name]-[hash].js',
            chunkFileNames: 'assets/[name]-[hash].js',
            assetFileNames: 'assets/[name]-[hash].[ext]',
            manualChunks: {
              vendor: ['react', 'react-dom'],
              router: ['react-router-dom'],
              ui: ['@emotion/react', '@emotion/styled'],
            },
          },
        },
      },
      define: {
        'import.meta.hot': 'undefined',
        'process.env.NODE_ENV': '"production"',
      },
    });
    
    console.log('✅ Production build completed successfully!');
  } catch (error) {
    console.error('❌ Build failed:', error);
    process.exit(1);
  }
}

buildProduction();

