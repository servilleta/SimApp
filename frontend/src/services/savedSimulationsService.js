import apiClient from './api';

export const saveSimulation = async (simulationData) => {
  const token = localStorage.getItem('authToken');
  if (!token) {
    throw new Error('Authentication token not found. Please log in.');
  }
  const response = await apiClient.post('/saved-simulations/save', simulationData, {
    headers: { 
      Authorization: `Bearer ${token}` 
    }
  });
  return response.data;
};

export const getSavedSimulations = async () => {
  const token = localStorage.getItem('authToken');
  if (!token) {
    throw new Error('Authentication token not found. Please log in.');
  }
  const response = await apiClient.get('/saved-simulations', {
    headers: { 
      Authorization: `Bearer ${token}` 
    }
  });
  return response.data;
};

export const loadSimulation = async (simulationId) => {
  const token = localStorage.getItem('authToken');
  if (!token) {
    throw new Error('Authentication token not found. Please log in.');
  }
  const response = await apiClient.get(`/saved-simulations/${simulationId}/load`, {
    headers: { 
      Authorization: `Bearer ${token}` 
    }
  });
  return response.data;
};

export const deleteSimulation = async (simulationId) => {
  const token = localStorage.getItem('authToken');
  if (!token) {
    throw new Error('Authentication token not found. Please log in.');
  }
  const response = await apiClient.delete(`/saved-simulations/${simulationId}`, {
    headers: { 
      Authorization: `Bearer ${token}` 
    }
  });
  return response.data;
}; 