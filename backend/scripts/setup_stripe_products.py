#!/usr/bin/env python3
"""
Stripe Product Setup Script

This script creates the necessary Stripe products and prices for the
Monte Carlo Platform subscription tiers based on the pricing matrix.

Run this script once after setting up your Stripe account to create
the products and prices that will be referenced in the application.

Usage:
    python scripts/setup_stripe_products.py
"""

import stripe
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config import settings

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

def create_stripe_products():
    """
    Create Stripe products and prices for all subscription tiers
    """
    
    if not settings.STRIPE_SECRET_KEY:
        print("❌ Error: STRIPE_SECRET_KEY not set in environment variables")
        print("Please set STRIPE_SECRET_KEY before running this script")
        return False
    
    print("🚀 Setting up Stripe products for Monte Carlo Platform...")
    print(f"Using Stripe account: {stripe.api_key[:12]}...")
    
    products_to_create = [
        {
            "tier": "starter",
            "name": "Starter Plan",
            "description": "Perfect for small teams getting started with Monte Carlo simulations",
            "price_monthly": 1900,  # $19.00 in cents
            "features": [
                "50K max iterations per simulation",
                "3 concurrent simulations", 
                "25MB file size limit",
                "10K formulas maximum",
                "10 projects stored",
                "Standard GPU priority",
                "Email support"
            ]
        },
        {
            "tier": "professional", 
            "name": "Professional Plan",
            "description": "Advanced features for professional analysts and teams",
            "price_monthly": 4900,  # $49.00 in cents
            "features": [
                "500K max iterations per simulation",
                "10 concurrent simulations",
                "100MB file size limit", 
                "50K formulas maximum",
                "50 projects stored",
                "High GPU priority",
                "1,000 API calls per month",
                "Priority email support",
                "Advanced analytics"
            ]
        },
        {
            "tier": "enterprise",
            "name": "Enterprise Plan", 
            "description": "Enterprise-grade features for large organizations",
            "price_monthly": 14900,  # $149.00 in cents
            "features": [
                "2M max iterations per simulation",
                "25 concurrent simulations",
                "500MB file size limit",
                "500K formulas maximum", 
                "Unlimited projects stored",
                "Premium GPU priority",
                "Unlimited API calls",
                "24/7 priority support",
                "Advanced analytics & reporting",
                "Custom integrations",
                "Dedicated account manager"
            ]
        },
        {
            "tier": "ultra",
            "name": "Ultra Plan",
            "description": "Unlimited power for the most demanding simulations", 
            "price_monthly": 29900,  # $299.00 in cents
            "features": [
                "Unlimited iterations per simulation",
                "Unlimited concurrent simulations",
                "No file size limit",
                "1M+ formulas support",
                "Unlimited projects stored", 
                "Dedicated GPU resources",
                "Unlimited API calls",
                "24/7 dedicated support",
                "White-label options",
                "Custom development",
                "SLA guarantees"
            ]
        }
    ]
    
    created_products = {}
    
    for product_info in products_to_create:
        try:
            print(f"\n📦 Creating product: {product_info['name']}...")
            
            # Create the product
            product = stripe.Product.create(
                name=product_info["name"],
                description=product_info["description"],
                metadata={
                    "tier": product_info["tier"],
                    "platform": "monte_carlo_platform",
                    "features": ", ".join(product_info["features"][:3])  # First 3 features
                }
            )
            
            print(f"✅ Product created: {product.id}")
            
            # Create the monthly price
            price = stripe.Price.create(
                product=product.id,
                unit_amount=product_info["price_monthly"],
                currency="usd",
                recurring={"interval": "month"},
                metadata={
                    "tier": product_info["tier"],
                    "billing_period": "monthly"
                }
            )
            
            print(f"💰 Price created: {price.id} (${product_info['price_monthly']/100:.2f}/month)")
            
            created_products[product_info["tier"]] = {
                "product_id": product.id,
                "price_id": price.id,
                "price_monthly": product_info["price_monthly"]
            }
            
        except stripe.error.StripeError as e:
            print(f"❌ Error creating {product_info['name']}: {str(e)}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error creating {product_info['name']}: {str(e)}")
            return False
    
    # Print summary
    print(f"\n🎉 Successfully created {len(created_products)} products!")
    print("\n📋 IMPORTANT: Update your StripeService.PRICE_IDS with these values:")
    print("="*70)
    
    print("PRICE_IDS = {")
    print('    "free": None,  # Free tier has no Stripe price')
    
    for tier, info in created_products.items():
        print(f'    "{tier}": "{info["price_id"]}",')
    
    print("}")
    
    print("\n📋 Product Summary:")
    print("="*70)
    for tier, info in created_products.items():
        price_dollars = info["price_monthly"] / 100
        print(f"{tier.title():12} | ${price_dollars:6.2f}/mo | {info['price_id']}")
    
    print("\n🔧 Next Steps:")
    print("1. Copy the PRICE_IDS dictionary above")
    print("2. Update backend/services/stripe_service.py")
    print("3. Set your webhook endpoint in Stripe Dashboard")
    print("4. Update STRIPE_WEBHOOK_SECRET in your environment")
    print("5. Test the billing flow!")
    
    return True

def list_existing_products():
    """
    List existing Stripe products (useful for debugging)
    """
    try:
        print("📋 Existing Stripe products:")
        products = stripe.Product.list(limit=20)
        
        if not products.data:
            print("No products found.")
            return
        
        for product in products.data:
            print(f"  • {product.name} ({product.id})")
            
            # Get prices for this product
            prices = stripe.Price.list(product=product.id)
            for price in prices.data:
                if price.recurring:
                    amount = price.unit_amount / 100
                    interval = price.recurring.interval
                    print(f"    ${amount:.2f}/{interval} ({price.id})")
                    
    except stripe.error.StripeError as e:
        print(f"❌ Error listing products: {str(e)}")

def main():
    """
    Main function to set up Stripe products
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_existing_products()
        return
    
    print("🎯 Monte Carlo Platform - Stripe Setup")
    print("="*50)
    
    # Confirm before creating
    response = input("\n⚠️  This will create products in your Stripe account. Continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Setup cancelled")
        return
    
    success = create_stripe_products()
    
    if success:
        print("\n✅ Stripe setup completed successfully!")
        print("\n🔄 To see existing products, run:")
        print("python scripts/setup_stripe_products.py --list")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
