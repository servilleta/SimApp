# 🌐 SimApp.ai Domain Setup Guide

This guide will help you set up your `simapp.ai` domain with proper SSL certificates from Let's Encrypt.

## Prerequisites

1. **Domain purchased**: ✅ `simapp.ai` from GoDaddy
2. **Server running**: ✅ Paperspace server at `209.51.170.185`
3. **Docker installed**: ✅ Ready to deploy

## Step 1: Update DNS Records in GoDaddy

⚠️ **CRITICAL**: You must update your DNS records first!

### GoDaddy DNS Configuration:

1. **Log into GoDaddy**: Go to [GoDaddy.com](https://godaddy.com) → My Products → DNS
2. **Find your domain**: `simapp.ai`
3. **Update A Records**:

   | Type | Name | Value | TTL |
   |------|------|-------|-----|
   | A | @ | `209.51.170.185` | 1 Hour |
   | A | www | `209.51.170.185` | 1 Hour |

4. **Save changes** and wait for DNS propagation (5-30 minutes)

### Verify DNS Propagation:
```bash
# Check if DNS is working
nslookup simapp.ai
# Should return: 209.51.170.185
```

## Step 2: Deploy with SSL Certificates

Once DNS is propagated, run the deployment script:

```bash
# Navigate to project directory
cd /home/paperspace/PROJECT

# Run domain deployment (this will generate Let's Encrypt SSL certificates)
./scripts/deploy-domain.sh
```

### What the script does:
1. ✅ Checks DNS propagation
2. ✅ Generates Let's Encrypt SSL certificates
3. ✅ Deploys production services
4. ✅ Sets up automatic certificate renewal
5. ✅ Verifies everything is working

## Step 3: Access Your Secure Site

After successful deployment:

- **Primary Domain**: https://simapp.ai
- **WWW Domain**: https://www.simapp.ai
- **HTTP Redirect**: http://simapp.ai → https://simapp.ai

## Features Enabled

### 🔐 **Security Features**:
- ✅ **Let's Encrypt SSL** - Trusted by all browsers
- ✅ **HTTPS Redirect** - All HTTP traffic redirects to HTTPS
- ✅ **Security Headers** - HSTS, XSS Protection, etc.
- ✅ **Rate Limiting** - Protection against abuse
- ✅ **CORS Configuration** - Proper cross-origin settings

### 🚀 **Performance Features**:
- ✅ **HTTP/2** - Faster loading
- ✅ **Gzip Compression** - Reduced bandwidth
- ✅ **Caching** - Optimized static assets
- ✅ **Load Balancing** - Backend scaling ready

### 🔄 **Maintenance Features**:
- ✅ **Auto SSL Renewal** - Certificates renew automatically
- ✅ **Health Monitoring** - Service health checks
- ✅ **Logging** - Comprehensive access logs

## Troubleshooting

### If DNS isn't working:
```bash
# Check current DNS
nslookup simapp.ai

# Should return your server IP: 209.51.170.185
# If not, wait longer or check GoDaddy settings
```

### If SSL generation fails:
```bash
# Run in staging mode first (for testing)
STAGING=true ./scripts/deploy-domain.sh

# Then run production mode
./scripts/deploy-domain.sh
```

### If site isn't accessible:
```bash
# Check container status
docker-compose -f docker-compose.domain.yml ps

# Check nginx logs
docker-compose -f docker-compose.domain.yml logs nginx
```

## Manual SSL Renewal

SSL certificates auto-renew, but you can manually renew:

```bash
# Manual renewal
./scripts/renew-certificates.sh

# Check renewal logs
tail -f logs/ssl-renewal.log
```

## Production Checklist

- [ ] DNS records updated in GoDaddy
- [ ] DNS propagation verified
- [ ] SSL certificates generated
- [ ] Site accessible at https://simapp.ai
- [ ] HTTP redirects to HTTPS
- [ ] Auto-renewal configured
- [ ] All services healthy

## Support

If you encounter issues:

1. **Check DNS**: Ensure `simapp.ai` resolves to `209.51.170.185`
2. **Check Logs**: Look at nginx and certbot logs
3. **Verify Ports**: Ensure ports 80 and 443 are open
4. **Test Staging**: Use `STAGING=true` for testing

---

🎉 **Once complete, your Monte Carlo Platform will be live at https://simapp.ai with enterprise-grade security!** 