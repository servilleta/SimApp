# 🔧 Stripe Webhook Secret Setup Guide

## Why This Is Important
The `STRIPE_WEBHOOK_SECRET` is critical for subscription synchronization. Without it, Stripe webhooks fail signature verification, causing subscription upgrades to not be reflected in the user interface.

## Step-by-Step Setup

### 1. Get Webhook Secret from Stripe

1. **Login to Stripe Dashboard**: https://dashboard.stripe.com/
2. **Navigate to Webhooks**: 
   - Go to **Developers** → **Webhooks**
3. **Find Your Webhook Endpoint**:
   - Look for webhook pointing to your domain: `https://yourdomain.com/api/webhooks/stripe`
   - If no webhook exists, create one with these events:
     - `customer.subscription.created`
     - `customer.subscription.updated` 
     - `customer.subscription.deleted`
     - `invoice.payment_succeeded`
     - `invoice.payment_failed`
4. **Get Signing Secret**:
   - Click on your webhook
   - Scroll down to **Signing secret**
   - Click **Reveal** and copy the secret (starts with `whsec_`)

### 2. Add to Environment

**Option A: Create .env file**
```bash
# In your project root
echo "STRIPE_WEBHOOK_SECRET=whsec_your_actual_secret_here" >> .env
```

**Option B: Export in shell**
```bash
export STRIPE_WEBHOOK_SECRET=whsec_your_actual_secret_here
```

**Option C: Add to docker-compose.yml directly**
```yaml
environment:
  - STRIPE_WEBHOOK_SECRET=whsec_your_actual_secret_here
```

### 3. Restart Services

```bash
# Restart the backend container to pick up new environment variable
docker-compose restart backend

# Or rebuild if needed
docker-compose down
docker-compose up -d
```

### 4. Test Webhook

**Test endpoint manually:**
```bash
curl -X POST http://localhost:9090/api/webhooks/stripe \
  -H "Content-Type: application/json" \
  -d '{"test": "webhook"}'
```

**Expected response before fix:**
```json
{"detail": "Webhook secret not configured"}
```

**Expected response after fix:**
```json
{"detail": "Missing Stripe signature"}
```

### 5. Verify in Logs

```bash
# Check backend logs
docker logs project-backend-1

# Look for:
# ✅ Good: "Processing Stripe webhook event: ..."
# ❌ Bad: "STRIPE_WEBHOOK_SECRET is not configured"
```

## Production Setup

For production, use:
- **Webhook URL**: `https://yourdomain.com/api/webhooks/stripe`
- **Events to listen for**:
  - `customer.subscription.created`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
  - `invoice.payment_succeeded`
  - `invoice.payment_failed`

## Troubleshooting

### Issue: "Invalid signature"
- Double-check webhook secret is correct
- Ensure webhook is pointing to correct URL
- Verify environment variable is loaded

### Issue: "Webhook secret not configured"
- Secret is empty or not loaded
- Restart backend service after adding secret

### Issue: Subscriptions still not syncing
- Use manual sync: `POST /api/admin/subscription-sync/user/{email}`
- Check logs for specific webhook errors

## Security Notes

- **Never commit webhook secrets to git**
- **Use different secrets for test/production**
- **Rotate secrets if compromised**
- **Verify webhook signatures always**
