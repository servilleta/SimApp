import apiClient from './api';

/**
 * Invoicing Service
 * Handles billing, invoices, subscriptions, and revenue tracking
 */

/**
 * Get revenue overview statistics
 */
export const getRevenueStats = async () => {
  try {
    const response = await apiClient.get('/admin/invoicing/stats');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch revenue statistics');
  }
};

/**
 * Get all invoices with optional filters
 */
export const getInvoices = async (filters = {}) => {
  try {
    const params = new URLSearchParams();
    
    if (filters.status) params.append('status', filters.status);
    if (filters.startDate) params.append('start_date', filters.startDate);
    if (filters.endDate) params.append('end_date', filters.endDate);
    if (filters.customerId) params.append('customer_id', filters.customerId);
    if (filters.page) params.append('page', filters.page);
    if (filters.limit) params.append('limit', filters.limit);

    const response = await apiClient.get(`/admin/invoicing/invoices?${params.toString()}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch invoices');
  }
};

/**
 * Get a specific invoice by ID
 */
export const getInvoice = async (invoiceId) => {
  try {
    const response = await apiClient.get(`/admin/invoicing/invoices/${invoiceId}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch invoice');
  }
};

/**
 * Create a new invoice
 */
export const createInvoice = async (invoiceData) => {
  try {
    const response = await apiClient.post('/admin/invoicing/invoices', invoiceData);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to create invoice');
  }
};

/**
 * Update an existing invoice
 */
export const updateInvoice = async (invoiceId, updateData) => {
  try {
    const response = await apiClient.patch(`/admin/invoicing/invoices/${invoiceId}`, updateData);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to update invoice');
  }
};

/**
 * Send payment reminder for an invoice
 */
export const sendPaymentReminder = async (invoiceId) => {
  try {
    const response = await apiClient.post(`/admin/invoicing/invoices/${invoiceId}/remind`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to send payment reminder');
  }
};

/**
 * Download invoice PDF
 */
export const downloadInvoicePDF = async (invoiceId) => {
  try {
    const response = await apiClient.get(`/admin/invoicing/invoices/${invoiceId}/pdf`, {
      responseType: 'blob'
    });
    
    // Create download link
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `invoice-${invoiceId}.pdf`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    
    return { success: true };
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to download invoice PDF');
  }
};

/**
 * Get all subscriptions with optional filters
 */
export const getSubscriptions = async (filters = {}) => {
  try {
    const params = new URLSearchParams();
    
    if (filters.status) params.append('status', filters.status);
    if (filters.plan) params.append('plan', filters.plan);
    if (filters.customerId) params.append('customer_id', filters.customerId);
    if (filters.page) params.append('page', filters.page);
    if (filters.limit) params.append('limit', filters.limit);

    const response = await apiClient.get(`/admin/invoicing/subscriptions?${params.toString()}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch subscriptions');
  }
};

/**
 * Get a specific subscription by ID
 */
export const getSubscription = async (subscriptionId) => {
  try {
    const response = await apiClient.get(`/admin/invoicing/subscriptions/${subscriptionId}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch subscription');
  }
};

/**
 * Update subscription status or plan
 */
export const updateSubscription = async (subscriptionId, updateData) => {
  try {
    const response = await apiClient.patch(`/admin/invoicing/subscriptions/${subscriptionId}`, updateData);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to update subscription');
  }
};

/**
 * Cancel a subscription
 */
export const cancelSubscription = async (subscriptionId, reason = '') => {
  try {
    const response = await apiClient.post(`/admin/invoicing/subscriptions/${subscriptionId}/cancel`, {
      reason
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to cancel subscription');
  }
};

/**
 * Get revenue reports with date range
 */
export const getRevenueReports = async (startDate, endDate, granularity = 'monthly') => {
  try {
    const params = new URLSearchParams({
      start_date: startDate,
      end_date: endDate,
      granularity
    });

    const response = await apiClient.get(`/admin/invoicing/reports/revenue?${params.toString()}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch revenue reports');
  }
};

/**
 * Get customer analysis reports
 */
export const getCustomerReports = async () => {
  try {
    const response = await apiClient.get('/admin/invoicing/reports/customers');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch customer reports');
  }
};

/**
 * Get plan performance reports
 */
export const getPlanReports = async () => {
  try {
    const response = await apiClient.get('/admin/invoicing/reports/plans');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch plan reports');
  }
};

/**
 * Export invoices to CSV
 */
export const exportInvoicesCSV = async (filters = {}) => {
  try {
    const params = new URLSearchParams();
    
    if (filters.status) params.append('status', filters.status);
    if (filters.startDate) params.append('start_date', filters.startDate);
    if (filters.endDate) params.append('end_date', filters.endDate);

    const response = await apiClient.get(`/admin/invoicing/export/invoices?${params.toString()}`, {
      responseType: 'blob'
    });
    
    // Create download link
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `invoices-${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    
    return { success: true };
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to export invoices');
  }
};

/**
 * Get payment methods analytics
 */
export const getPaymentMethodStats = async () => {
  try {
    const response = await apiClient.get('/admin/invoicing/stats/payment-methods');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch payment method statistics');
  }
};

/**
 * Get overdue invoices summary
 */
export const getOverdueInvoices = async () => {
  try {
    const response = await apiClient.get('/admin/invoicing/invoices/overdue');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch overdue invoices');
  }
};

/**
 * Process bulk payment reminders
 */
export const sendBulkPaymentReminders = async (invoiceIds) => {
  try {
    const response = await apiClient.post('/admin/invoicing/bulk/reminders', {
      invoice_ids: invoiceIds
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to send bulk payment reminders');
  }
};

/**
 * Calculate revenue projections
 */
export const getRevenueProjections = async (months = 12) => {
  try {
    const response = await apiClient.get(`/admin/invoicing/projections?months=${months}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to calculate revenue projections');
  }
};

/**
 * Utility function to format currency
 */
export const formatCurrency = (amount, currency = 'USD') => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: 2
  }).format(amount);
};

/**
 * Utility function to get status badge styling
 */
export const getInvoiceStatusStyle = (status) => {
  const styles = {
    paid: {
      backgroundColor: 'var(--color-success-bg)',
      color: 'var(--color-success)',
      border: '1px solid var(--color-success)'
    },
    pending: {
      backgroundColor: 'rgba(255, 214, 0, 0.1)',
      color: '#e6a700',
      border: '1px solid #e6a700'
    },
    overdue: {
      backgroundColor: 'var(--color-error-bg)',
      color: 'var(--color-error)',
      border: '1px solid var(--color-error)'
    },
    cancelled: {
      backgroundColor: 'var(--color-warm-white)',
      color: 'var(--color-text-secondary)',
      border: '1px solid var(--color-border-light)'
    }
  };
  
  return styles[status] || styles.pending;
};

/**
 * Utility function to calculate days overdue
 */
export const getDaysOverdue = (dueDate) => {
  const today = new Date();
  const due = new Date(dueDate);
  const timeDiff = today - due;
  const daysDiff = Math.ceil(timeDiff / (1000 * 60 * 60 * 24));
  return daysDiff > 0 ? daysDiff : 0;
};

export default {
  getRevenueStats,
  getInvoices,
  getInvoice,
  createInvoice,
  updateInvoice,
  sendPaymentReminder,
  downloadInvoicePDF,
  getSubscriptions,
  getSubscription,
  updateSubscription,
  cancelSubscription,
  getRevenueReports,
  getCustomerReports,
  getPlanReports,
  exportInvoicesCSV,
  getPaymentMethodStats,
  getOverdueInvoices,
  sendBulkPaymentReminders,
  getRevenueProjections,
  formatCurrency,
  getInvoiceStatusStyle,
  getDaysOverdue
};

