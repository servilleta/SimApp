/**
 * Webhook Testing Utilities
 * 
 * Provides utilities for testing and validating webhook endpoints
 */

// Test webhook URL validation
export const validateWebhookURL = (url) => {
  try {
    const urlObj = new URL(url);
    
    // Check for HTTPS (recommended but not required for localhost)
    if (urlObj.protocol !== 'https:' && !urlObj.hostname.includes('localhost') && urlObj.hostname !== '127.0.0.1') {
      return {
        valid: false,
        warning: 'HTTPS is strongly recommended for webhook URLs in production'
      };
    }
    
    // Check for localhost/development URLs
    if (urlObj.hostname.includes('localhost') || urlObj.hostname === '127.0.0.1') {
      return {
        valid: true,
        warning: 'This appears to be a development URL. Make sure your server is accessible.'
      };
    }
    
    return { valid: true };
  } catch (error) {
    return {
      valid: false,
      error: 'Invalid URL format'
    };
  }
};

// Generate sample webhook payload for testing
export const generateTestPayload = (eventType = 'simulation.completed') => {
  const basePayload = {
    event: eventType,
    timestamp: new Date().toISOString(),
    simulation_id: `b2b_sim_${Date.now().toString(36)}${Math.random().toString(36).substr(2, 5)}`,
  };

  switch (eventType) {
    case 'simulation.started':
      return {
        ...basePayload,
        data: {
          model_id: `mdl_${Date.now().toString(36)}${Math.random().toString(36).substr(2, 8)}`,
          iterations: 10000,
          variables_count: 2,
          output_cells_count: 1,
          estimated_completion: new Date(Date.now() + 300000).toISOString() // 5 minutes from now
        }
      };

    case 'simulation.progress':
      return {
        ...basePayload,
        data: {
          model_id: `mdl_${Date.now().toString(36)}${Math.random().toString(36).substr(2, 8)}`,
          progress: {
            percentage: 50,
            phase: 'processing',
            stage_description: 'Running Monte Carlo iterations'
          },
          iterations_completed: 5000,
          estimated_completion: new Date(Date.now() + 150000).toISOString() // 2.5 minutes from now
        }
      };

    case 'simulation.completed':
      return {
        ...basePayload,
        data: {
          status: 'completed',
          execution_time: '45.2s',
          iterations_completed: 10000,
          results: {
            'J25': {
              mean: 1250.67,
              std: 234.89,
              min: 445.12,
              max: 2890.23,
              percentiles: {
                5: 678.45,
                25: 980.12,
                50: 1250.67,
                75: 1520.89,
                95: 1890.23
              },
              var_95: 890.23,
              var_99: 645.12
            }
          },
          download_links: {
            pdf: `/simapp-api/simulations/${basePayload.simulation_id}/download/pdf`,
            xlsx: `/simapp-api/simulations/${basePayload.simulation_id}/download/xlsx`,
            json: `/simapp-api/simulations/${basePayload.simulation_id}/download/json`
          }
        }
      };

    case 'simulation.failed':
      return {
        ...basePayload,
        data: {
          status: 'failed',
          error: 'Invalid cell reference in formula',
          execution_time: '12.3s',
          failed_at: new Date().toISOString()
        }
      };

    case 'simulation.cancelled':
      return {
        ...basePayload,
        data: {
          status: 'cancelled',
          cancelled_by: 'user',
          execution_time: '28.7s',
          cancelled_at: new Date().toISOString()
        }
      };

    default:
      return {
        ...basePayload,
        data: {
          message: 'Test webhook payload',
          test: true
        }
      };
  }
};

// Webhook security utilities
export const generateHMACSignature = async (payload, secret) => {
  const encoder = new TextEncoder();
  const keyData = encoder.encode(secret);
  const messageData = encoder.encode(JSON.stringify(payload));
  
  const cryptoKey = await window.crypto.subtle.importKey(
    'raw',
    keyData,
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );
  
  const signature = await window.crypto.subtle.sign('HMAC', cryptoKey, messageData);
  const hashArray = Array.from(new Uint8Array(signature));
  const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  
  return `sha256=${hashHex}`;
};

// Verify webhook signature (for demonstration)
export const verifyWebhookSignature = async (payload, signature, secret) => {
  const expectedSignature = await generateHMACSignature(payload, secret);
  return signature === expectedSignature;
};

// Event type descriptions
export const eventDescriptions = {
  'simulation.started': 'Triggered when a simulation begins processing',
  'simulation.progress': 'Triggered for simulation progress updates (every 25%)',
  'simulation.completed': 'Triggered when a simulation finishes successfully',
  'simulation.failed': 'Triggered when a simulation encounters an error',
  'simulation.cancelled': 'Triggered when a simulation is cancelled by user'
};

// Example webhook endpoint code snippets
export const getWebhookExamples = (language = 'javascript') => {
  const examples = {
    javascript: `// Express.js webhook endpoint example
const express = require('express');
const crypto = require('crypto');
const app = express();

// Middleware to capture raw body for signature verification
app.use('/webhooks', express.raw({ type: 'application/json' }));

app.post('/webhooks/simulations', (req, res) => {
  try {
    const signature = req.headers['x-simapp-signature'];
    const payload = req.body;
    const secret = process.env.WEBHOOK_SECRET || 'your-secret';
    
    // Verify signature
    const expectedSignature = 'sha256=' + crypto
      .createHmac('sha256', secret)
      .update(payload)
      .digest('hex');
    
    if (signature !== expectedSignature) {
      return res.status(401).json({ error: 'Invalid signature' });
    }
    
    const event = JSON.parse(payload);
    console.log('Received webhook:', event.event, event.simulation_id);
    
    // Process the webhook event
    switch (event.event) {
      case 'simulation.completed':
        console.log('Simulation completed:', event.data.results);
        break;
      case 'simulation.failed':
        console.log('Simulation failed:', event.data.error);
        break;
      default:
        console.log('Other event:', event.event);
    }
    
    res.status(200).json({ received: true });
  } catch (error) {
    console.error('Webhook error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.listen(3000, () => {
  console.log('Webhook server listening on port 3000');
});`,

    python: `# Flask webhook endpoint example
from flask import Flask, request, jsonify
import hmac
import hashlib
import json
import os

app = Flask(__name__)

@app.route('/webhooks/simulations', methods=['POST'])
def handle_webhook():
    try:
        signature = request.headers.get('X-SimApp-Signature', '')
        payload = request.get_data()
        secret = os.environ.get('WEBHOOK_SECRET', 'your-secret')
        
        # Verify signature
        expected_signature = 'sha256=' + hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        if signature != expected_signature:
            return jsonify({'error': 'Invalid signature'}), 401
        
        event = json.loads(payload)
        print(f"Received webhook: {event['event']} {event['simulation_id']}")
        
        # Process the webhook event
        if event['event'] == 'simulation.completed':
            print(f"Simulation completed: {event['data']['results']}")
        elif event['event'] == 'simulation.failed':
            print(f"Simulation failed: {event['data']['error']}")
        else:
            print(f"Other event: {event['event']}")
        
        return jsonify({'received': True})
    
    except Exception as e:
        print(f"Webhook error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)`,

    curl: `# Test webhook endpoint with curl
curl -X POST https://your-app.com/webhooks/simulations \\
  -H "Content-Type: application/json" \\
  -H "X-SimApp-Signature: sha256=generated_signature" \\
  -H "X-SimApp-Event: simulation.completed" \\
  -H "X-SimApp-Delivery: 12345" \\
  -H "X-SimApp-Timestamp: 2024-09-19T12:00:00Z" \\
  -d '{
    "event": "simulation.completed",
    "timestamp": "2024-09-19T12:00:00Z",
    "simulation_id": "sim_123",
    "data": {
      "status": "completed",
      "execution_time": "45.2s",
      "results": {
        "J25": {
          "mean": 1250.67,
          "var_95": 890.23
        }
      }
    }
  }'`
  };

  return examples[language] || examples.javascript;
};

export default {
  validateWebhookURL,
  generateTestPayload,
  generateHMACSignature,
  verifyWebhookSignature,
  eventDescriptions,
  getWebhookExamples
};
