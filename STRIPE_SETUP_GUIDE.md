# Stripe Integration Setup Guide

## Overview

This guide walks you through setting up Stripe payments for the Monte Carlo Platform's subscription system. The platform supports 5 pricing tiers with the following features:

| Plan | Price | Max Iterations | Concurrent Sims | File Size | Max Formulas | Projects | GPU Priority | API Calls |
|------|-------|----------------|-----------------|-----------|--------------|----------|--------------|-----------|
| **Free** | $0 | 5K | 1 | 10MB | 1K | 3 | Low | 0 |
| **Starter** | $19/mo | 50K | 3 | 25MB | 10K | 10 | Standard | 0 |
| **Professional** | $49/mo | 500K | 10 | 100MB | 50K | 50 | High | 1,000 |
| **Enterprise** | $149/mo | 2M | 25 | 500MB | 500K | Unlimited | Premium | Unlimited |
| **Ultra** | $299/mo | Unlimited | Unlimited | No limit | 1M+ | Unlimited | Dedicated | Unlimited |

## Prerequisites

1. **Stripe Account**: Create a Stripe account at [stripe.com](https://stripe.com)
2. **API Keys**: Get your Stripe secret and publishable keys
3. **Python Environment**: Ensure you have the backend environment set up

## Step 1: Environment Configuration

Add these environment variables to your `.env` file:

```env
# Stripe Configuration
STRIPE_PUBLISHABLE_KEY=pk_test_your_publishable_key_here
STRIPE_SECRET_KEY=sk_test_your_secret_key_here
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret_here
```

> **Note**: Use test keys for development (`pk_test_` and `sk_test_`). Switch to live keys for production.

## Step 2: Create Stripe Products and Prices

Run the setup script to create the subscription products in your Stripe account:

```bash
cd backend
python scripts/setup_stripe_products.py
```

This script will:
- Create 4 products (Starter, Professional, Enterprise, Ultra)
- Create monthly prices for each product
- Output the price IDs you need to configure

## Step 3: Update Price IDs

After running the setup script, update the `PRICE_IDS` in `backend/services/stripe_service.py`:

```python
PRICE_IDS = {
    "free": None,
    "starter": "price_1234567890_starter",      # Replace with actual price ID
    "professional": "price_1234567890_pro",     # Replace with actual price ID  
    "enterprise": "price_1234567890_enterprise", # Replace with actual price ID
    "ultra": "price_1234567890_ultra"           # Replace with actual price ID
}
```

## Step 4: Database Migration

Run the database migration to add the new subscription fields:

```bash
cd backend
alembic upgrade head
```

## Step 5: Webhook Configuration

### 5.1 Set up Webhook Endpoint

In your Stripe Dashboard:

1. Go to **Developers > Webhooks**
2. Click **Add endpoint**
3. Set endpoint URL: `https://yourdomain.com/api/webhooks/stripe`
4. Select these events:
   - `customer.subscription.created`
   - `customer.subscription.updated` 
   - `customer.subscription.deleted`
   - `invoice.payment_succeeded`
   - `invoice.payment_failed`
   - `customer.subscription.trial_will_end`

### 5.2 Configure Webhook Secret

Copy the webhook signing secret from Stripe Dashboard and add it to your environment:

```env
STRIPE_WEBHOOK_SECRET=whsec_your_actual_webhook_secret_here
```

## Step 6: Frontend Integration

### 6.1 Install Stripe SDK

```bash
npm install @stripe/stripe-js
```

### 6.2 Create Billing Components

Example React component for subscription management:

```jsx
import { loadStripe } from '@stripe/stripe-js';

const stripePromise = loadStripe(process.env.REACT_APP_STRIPE_PUBLISHABLE_KEY);

export function SubscriptionPlans() {
  const handleSubscribe = async (planTier) => {
    try {
      const response = await fetch('/api/billing/checkout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({
          plan: planTier,
          success_url: `${window.location.origin}/success`,
          cancel_url: `${window.location.origin}/pricing`
        })
      });
      
      const { checkout_url } = await response.json();
      window.location.href = checkout_url;
    } catch (error) {
      console.error('Subscription error:', error);
    }
  };

  return (
    <div className="pricing-grid">
      {/* Render pricing plans with subscribe buttons */}
    </div>
  );
}
```

## Step 7: API Endpoints

The following API endpoints are now available:

### Billing Endpoints

- `GET /api/billing/plans` - Get all available plans
- `GET /api/billing/subscription` - Get current user's subscription
- `POST /api/billing/checkout` - Create checkout session
- `POST /api/billing/portal` - Access billing portal
- `POST /api/billing/cancel` - Cancel subscription
- `GET /api/billing/usage` - Get usage information

### Webhook Endpoints

- `POST /api/webhooks/stripe` - Handle Stripe webhooks

## Step 8: Quota Enforcement

The `QuotaService` automatically enforces limits based on subscription tiers:

```python
from services.quota_service import QuotaService

# Check if user can run simulation
allowed, message = QuotaService.check_iteration_limit(db, user_id, 100000)
if not allowed:
    raise HTTPException(status_code=403, detail=message)

# Record usage after simulation
QuotaService.record_simulation_usage(db, user_id, iterations_run=50000)
```

## Step 9: Testing

### 9.1 Test Cards

Use Stripe's test cards for testing:

- **Success**: `4242424242424242`
- **Decline**: `4000000000000002`  
- **3D Secure**: `4000002500003155`

### 9.2 Test Workflow

1. Create a test user account
2. Subscribe to a paid plan using test card
3. Verify subscription status in database
4. Test quota enforcement
5. Test webhook events (subscription updates, cancellations)

## Step 10: Production Deployment

### 10.1 Switch to Live Keys

Replace test keys with live keys in production:

```env
STRIPE_PUBLISHABLE_KEY=pk_live_your_live_publishable_key
STRIPE_SECRET_KEY=sk_live_your_live_secret_key
```

### 10.2 Update Webhook URL

Update your webhook endpoint URL to point to your production domain.

### 10.3 Security Considerations

- ✅ Always verify webhook signatures
- ✅ Use HTTPS for all endpoints
- ✅ Validate user permissions before API calls
- ✅ Log all billing events for audit trail
- ✅ Implement rate limiting on billing endpoints

## Troubleshooting

### Common Issues

1. **Webhook signature verification failed**
   - Check webhook secret is correct
   - Ensure raw request body is used for verification

2. **Price ID not found**
   - Verify price IDs are correctly copied from Stripe Dashboard
   - Check that products are created in the correct Stripe account

3. **Database errors**
   - Ensure migration has been run
   - Check database connection and permissions

4. **CORS errors**
   - Update CORS settings to allow your frontend domain
   - Verify API endpoints are accessible

### Support

For additional support:
- Check Stripe documentation: [stripe.com/docs](https://stripe.com/docs)
- Review Stripe logs in Dashboard
- Check application logs for detailed error messages
- Test with Stripe CLI for webhook debugging

## Security Checklist

- [ ] Webhook signature verification implemented
- [ ] API keys stored securely (not in code)
- [ ] User authentication required for all billing operations
- [ ] Quota limits properly enforced
- [ ] Audit logging for all billing events
- [ ] Rate limiting on public endpoints
- [ ] HTTPS enabled in production
- [ ] Database access properly secured

## Next Steps

After successful setup:

1. **Monitor Usage**: Set up dashboards to monitor subscription metrics
2. **Analytics**: Implement conversion tracking and user behavior analytics  
3. **Notifications**: Add email notifications for billing events
4. **Customer Support**: Set up billing support workflows
5. **Scaling**: Monitor for performance and optimize as needed

Your Monte Carlo Platform is now ready with full Stripe subscription management! 🚀
