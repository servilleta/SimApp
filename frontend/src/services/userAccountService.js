import apiClient from './api';

/**
 * User Account Management Service
 * Handles user profile, password, privacy settings, and account actions
 */

/**
 * Get current user profile
 */
export const getUserProfile = async () => {
  try {
    const response = await apiClient.get('/auth/me');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch user profile');
  }
};

/**
 * Update user profile information
 */
export const updateUserProfile = async (profileData) => {
  try {
    const response = await apiClient.patch('/auth/me', profileData);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to update profile');
  }
};

/**
 * Change user password
 */
export const changePassword = async (passwordData) => {
  try {
    const response = await apiClient.patch('/auth/me', {
      password: passwordData.new_password
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to change password');
  }
};

/**
 * Revoke all user sessions (logout from all devices)
 */
export const revokeAllSessions = async () => {
  try {
    const response = await apiClient.post('/auth/me/revoke-sessions');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to revoke sessions');
  }
};

/**
 * Delete user account
 */
export const deleteUserAccount = async () => {
  try {
    const response = await apiClient.delete('/auth/me');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to delete account');
  }
};

/**
 * Get user dashboard statistics (includes plan info)
 */
export const getUserDashboardStats = async () => {
  try {
    const response = await apiClient.get('/auth/dashboard/stats');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch dashboard stats');
  }
};

/**
 * Update privacy settings (placeholder - would need backend endpoint)
 */
export const updatePrivacySettings = async (privacyData) => {
  try {
    // For now, store in localStorage as backend endpoint doesn't exist yet
    localStorage.setItem('userPrivacySettings', JSON.stringify(privacyData));
    return { message: 'Privacy settings updated successfully' };
  } catch (error) {
    throw new Error('Failed to update privacy settings');
  }
};

/**
 * Get privacy settings (placeholder - would need backend endpoint)
 */
export const getPrivacySettings = async () => {
  try {
    // For now, get from localStorage as backend endpoint doesn't exist yet
    const settings = localStorage.getItem('userPrivacySettings');
    return settings ? JSON.parse(settings) : {
      email_notifications: true,
      marketing_emails: false,
      data_sharing: false
    };
  } catch (error) {
    // Return default settings if there's an error
    return {
      email_notifications: true,
      marketing_emails: false,
      data_sharing: false
    };
  }
};

/**
 * Get current plan information
 */
export const getCurrentPlan = async () => {
  try {
    const dashboardStats = await getUserDashboardStats();
    
    // Extract plan information from dashboard stats
    const subscription = dashboardStats.subscription || {};
    const limits = dashboardStats.limits || {};
    const usage = dashboardStats.usage || {};
    
    // Map subscription tier to plan details
    const planDetails = {
      free: {
        name: 'Free Plan',
        simulations_per_month: limits.simulations_per_month || 10,
        max_file_size: '50MB',
        priority_support: false,
        advanced_analytics: false,
        price: '$0/month'
      },
      basic: {
        name: 'Basic Plan',
        simulations_per_month: limits.simulations_per_month || 1000,
        max_file_size: '25MB',
        priority_support: false,
        advanced_analytics: true,
        price: '$19/month'
      },
      starter: {
        name: 'Starter Plan',
        simulations_per_month: limits.simulations_per_month || 100,
        max_file_size: '200MB',
        priority_support: false,
        advanced_analytics: true,
        price: '$29/month'
      },
      pro: {
        name: 'Pro Plan',
        simulations_per_month: limits.simulations_per_month || 10000,
        max_file_size: '10MB',
        priority_support: true,
        advanced_analytics: true,
        price: '$49/month'
      },
      professional: {
        name: 'Professional Plan',
        simulations_per_month: limits.simulations_per_month || 1000,
        max_file_size: '1GB',
        priority_support: true,
        advanced_analytics: true,
        price: '$99/month'
      },
      enterprise: {
        name: 'Enterprise Plan',
        simulations_per_month: -1, // Unlimited
        max_file_size: 'Unlimited',
        priority_support: true,
        advanced_analytics: true,
        price: 'Custom pricing'
      }
    };
    
    const currentTier = subscription.tier || 'free';
    const planInfo = planDetails[currentTier] || planDetails.free;
    
    return {
      ...planInfo,
      tier: currentTier,
      status: subscription.status || 'active',
      usage: {
        simulations_used: usage.simulations_this_month || 0,
        simulations_limit: planInfo.simulations_per_month,
        period_start: usage.period_start,
        current_month: usage.current_month
      }
    };
  } catch (error) {
    // Return default free plan if there's an error
    return {
      name: 'Free Plan',
      tier: 'free',
      simulations_per_month: 10,
      max_file_size: '50MB',
      priority_support: false,
      advanced_analytics: false,
      price: '$0/month',
      status: 'active',
      usage: {
        simulations_used: 0,
        simulations_limit: 10,
        period_start: new Date().toISOString(),
        current_month: new Date().toISOString().substring(0, 7)
      }
    };
  }
};

/**
 * Validate password strength
 */
export const validatePassword = (password) => {
  const minLength = 8;
  const hasUpperCase = /[A-Z]/.test(password);
  const hasLowerCase = /[a-z]/.test(password);
  const hasNumbers = /\d/.test(password);
  const hasSpecialChar = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password);
  
  const errors = [];
  
  if (password.length < minLength) {
    errors.push(`Password must be at least ${minLength} characters long`);
  }
  
  if (!hasUpperCase) {
    errors.push('Password must contain at least one uppercase letter');
  }
  
  if (!hasLowerCase) {
    errors.push('Password must contain at least one lowercase letter');
  }
  
  if (!hasNumbers) {
    errors.push('Password must contain at least one number');
  }
  
  if (!hasSpecialChar) {
    errors.push('Password must contain at least one special character');
  }
  
  return {
    isValid: errors.length === 0,
    errors,
    strength: errors.length === 0 ? 'strong' : errors.length <= 2 ? 'medium' : 'weak'
  };
};

/**
 * Format usage percentage
 */
export const formatUsagePercentage = (used, limit) => {
  if (limit === -1 || limit === 0) {
    return 0; // Unlimited or no limit
  }
  return Math.min(Math.round((used / limit) * 100), 100);
};

/**
 * Get usage status color
 */
export const getUsageStatusColor = (percentage) => {
  if (percentage >= 90) return '#ff6b6b'; // Red - Critical
  if (percentage >= 75) return '#ffa726'; // Orange - Warning
  if (percentage >= 50) return '#66bb6a'; // Green - Good
  return '#42a5f5'; // Blue - Excellent
};

export default {
  getUserProfile,
  updateUserProfile,
  changePassword,
  revokeAllSessions,
  deleteUserAccount,
  getUserDashboardStats,
  updatePrivacySettings,
  getPrivacySettings,
  getCurrentPlan,
  validatePassword,
  formatUsagePercentage,
  getUsageStatusColor
};

