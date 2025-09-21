// Placeholder for utility functions related to data formatting

export const formatCurrency = (value, currency = 'USD') => {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(value);
};

export const formatDate = (dateString) => {
  if (!dateString) return '';
  try {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch (e) {
    return dateString; // Return original if parsing fails
  }
};

// Add other formatting utils as needed (e.g., for numbers, percentages)
export const formatNumber = (value, decimals = 2) => {
  if (typeof value !== 'number') return value;
  return value.toFixed(decimals);
}; 