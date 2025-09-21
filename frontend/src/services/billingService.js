/**
 * Billing Service for Stripe Integration
 * 
 * Handles all billing-related API calls including subscription management,
 * plan information, and usage tracking.
 */

import { getToken } from './authService';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

class BillingService {
  
  /**
   * Get authentication headers for API requests
   */
  static getAuthHeaders() {
    const token = getToken();
    return {
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` })
    };
  }

  /**
   * Get all available subscription plans
   */
  static async getPlans() {
    try {
      const response = await fetch(`${API_BASE}/billing/plans`, {
        method: 'GET',
        headers: this.getAuthHeaders()
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch plans: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching plans:', error);
      throw error;
    }
  }

  /**
   * Get current user's subscription information
   */
  static async getCurrentSubscription() {
    try {
      const response = await fetch(`${API_BASE}/billing/subscription`, {
        method: 'GET',
        headers: this.getAuthHeaders()
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch subscription: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching subscription:', error);
      throw error;
    }
  }

  /**
   * Create a Stripe checkout session for subscription
   */
  static async createCheckoutSession(plan, successUrl, cancelUrl) {
    try {
      const response = await fetch(`${API_BASE}/billing/checkout`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({
          plan,
          success_url: successUrl,
          cancel_url: cancelUrl
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to create checkout session: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error creating checkout session:', error);
      throw error;
    }
  }

  /**
   * Create a Stripe billing portal session
   */
  static async createBillingPortalSession() {
    try {
      const response = await fetch(`${API_BASE}/billing/portal`, {
        method: 'POST',
        headers: this.getAuthHeaders()
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to create billing portal session: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error creating billing portal session:', error);
      throw error;
    }
  }

  /**
   * Cancel user's subscription
   */
  static async cancelSubscription() {
    try {
      const response = await fetch(`${API_BASE}/billing/cancel`, {
        method: 'POST',
        headers: this.getAuthHeaders()
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to cancel subscription: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error cancelling subscription:', error);
      throw error;
    }
  }

  /**
   * Get user's current usage and limits
   */
  static async getUsageInfo() {
    try {
      const response = await fetch(`${API_BASE}/billing/usage`, {
        method: 'GET',
        headers: this.getAuthHeaders()
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch usage info: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching usage info:', error);
      throw error;
    }
  }

  /**
   * Subscribe to a plan (handles the full flow)
   */
  static async subscribeToPlan(planTier) {
    try {
      // Map new plan tiers to backend expected names
      const planMapping = {
        'starter': 'starter',
        'professional': 'pro', // Backend might expect 'pro' instead of 'professional'
        'enterprise': 'enterprise'
      };

      const backendPlanName = planMapping[planTier.toLowerCase()] || planTier.toLowerCase();

      // Create success and cancel URLs
      const baseUrl = window.location.origin;
      const successUrl = `${baseUrl}/my-dashboard?subscription_success=true&plan=${planTier}`;
      const cancelUrl = `${baseUrl}/account?subscription_cancelled=true`;

      // Create checkout session
      const { checkout_url } = await this.createCheckoutSession(backendPlanName, successUrl, cancelUrl);
      
      // Redirect to Stripe checkout
      window.location.href = checkout_url;
    } catch (error) {
      console.error('Error subscribing to plan:', error);
      throw error;
    }
  }

  /**
   * Open Stripe billing portal
   */
  static async openBillingPortal() {
    try {
      const { portal_url } = await this.createBillingPortalSession();
      window.location.href = portal_url;
    } catch (error) {
      console.error('Error opening billing portal:', error);
      throw error;
    }
  }

  /**
   * Check if user can perform an action based on their plan limits
   */
  static async checkQuota(action, amount = 1) {
    try {
      const usage = await this.getUsageInfo();
      const { limits, current_usage } = usage;

      switch (action) {
        case 'simulation':
          if (limits.max_iterations === -1) return { allowed: true, message: 'Unlimited iterations' };
          return {
            allowed: amount <= limits.max_iterations,
            message: amount > limits.max_iterations 
              ? `Request exceeds limit of ${limits.max_iterations.toLocaleString()} iterations`
              : 'Within iteration limits'
          };

        case 'concurrent_simulation':
          if (limits.concurrent_simulations === -1) return { allowed: true, message: 'Unlimited concurrent simulations' };
          return {
            allowed: current_usage.concurrent_simulations < limits.concurrent_simulations,
            message: current_usage.concurrent_simulations >= limits.concurrent_simulations
              ? `Concurrent simulation limit of ${limits.concurrent_simulations} reached`
              : 'Can start new simulation'
          };

        case 'file_upload':
          if (limits.file_size_mb === -1) return { allowed: true, message: 'No file size limit' };
          return {
            allowed: amount <= limits.file_size_mb,
            message: amount > limits.file_size_mb
              ? `File size ${amount}MB exceeds limit of ${limits.file_size_mb}MB`
              : 'Within file size limits'
          };

        case 'api_call':
          if (limits.api_calls_per_month === -1) return { allowed: true, message: 'Unlimited API calls' };
          if (limits.api_calls_per_month === 0) return { allowed: false, message: 'API access not included in plan' };
          return {
            allowed: current_usage.api_calls < limits.api_calls_per_month,
            message: current_usage.api_calls >= limits.api_calls_per_month
              ? 'Monthly API call limit reached'
              : 'Within API call limits'
          };

        default:
          return { allowed: true, message: 'Unknown action' };
      }
    } catch (error) {
      console.error('Error checking quota:', error);
      return { allowed: false, message: 'Error checking quota limits' };
    }
  }

  /**
   * Format plan limits for display
   */
  static formatLimits(limits) {
    return {
      maxIterations: limits.max_iterations === -1 ? 'Unlimited' : limits.max_iterations.toLocaleString(),
      concurrentSimulations: limits.concurrent_simulations === -1 ? 'Unlimited' : limits.concurrent_simulations,
      fileSizeLimit: limits.file_size_mb === -1 ? 'No limit' : `${limits.file_size_mb}MB`,
      maxFormulas: limits.max_formulas === -1 ? 'Unlimited' : limits.max_formulas.toLocaleString(),
      projectsStored: limits.projects_stored === -1 ? 'Unlimited' : limits.projects_stored,
      gpuPriority: limits.gpu_priority,
      apiCallsPerMonth: limits.api_calls_per_month === -1 ? 'Unlimited' : 
                       limits.api_calls_per_month === 0 ? 'Not included' : 
                       limits.api_calls_per_month.toLocaleString()
    };
  }

  /**
   * Get plan recommendations based on usage
   */
  static async getPlanRecommendations() {
    try {
      const [plans, usage] = await Promise.all([
        this.getPlans(),
        this.getUsageInfo()
      ]);

      const { current_usage, limits } = usage;
      const recommendations = [];

      // Check if user is approaching limits
      if (limits.max_iterations !== -1 && current_usage.total_iterations > limits.max_iterations * 0.8) {
        recommendations.push({
          type: 'upgrade',
          reason: 'approaching_iteration_limit',
          message: 'You\'re approaching your monthly iteration limit. Consider upgrading for more capacity.'
        });
      }

      if (limits.concurrent_simulations !== -1 && current_usage.concurrent_simulations >= limits.concurrent_simulations) {
        recommendations.push({
          type: 'upgrade',
          reason: 'concurrent_limit_reached',
          message: 'You\'ve reached your concurrent simulation limit. Upgrade for more parallel processing.'
        });
      }

      if (limits.api_calls_per_month !== -1 && current_usage.api_calls > limits.api_calls_per_month * 0.8) {
        recommendations.push({
          type: 'upgrade',
          reason: 'approaching_api_limit',
          message: 'You\'re approaching your monthly API call limit. Consider upgrading for more API access.'
        });
      }

      return {
        plans,
        usage,
        recommendations
      };
    } catch (error) {
      console.error('Error getting plan recommendations:', error);
      throw error;
    }
  }
}

export default BillingService;
