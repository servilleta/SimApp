import React, { useEffect, useState } from 'react';
import { getToken } from '../services/authService';
import axios from 'axios';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import Modal from '../components/common/Modal';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
const USERS_API_URL = `${API_BASE_URL}/auth0/users`;

const AdminUsersPage = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [editUser, setEditUser] = useState(null);
  const [form, setForm] = useState({
    username: '',
    email: '',
    full_name: '',
    is_admin: false,
    password: '',
    disabled: false,
  });

  const fetchUsers = async () => {
    console.log('AdminUsersPage: Fetching users...');
    setLoading(true);
    setError('');
    try {
      const res = await axios.get(USERS_API_URL, {
        headers: { Authorization: `Bearer ${getToken()}` },
      });
      console.log('AdminUsersPage: Successfully fetched users:', res.data.length, 'users');
      setUsers(res.data);
    } catch (err) {
      console.error('AdminUsersPage: Error fetching users:', err.response?.data || err.message);
      setError(err.response?.data?.detail || 'Failed to fetch users');
    }
    setLoading(false);
  };

  useEffect(() => {
    console.log('AdminUsersPage: Component mounted, fetching users');
    fetchUsers();
  }, []);

  const openModal = (user = null) => {
    // Prevent opening modal if it's already open
    if (showModal) {
      console.log('AdminUsersPage: Modal already open, ignoring duplicate request');
      return;
    }
    
    console.log('AdminUsersPage: Opening modal for user:', user ? `edit ${user.username}` : 'create new');
    console.log('AdminUsersPage: Current showModal state:', showModal);
    
    setEditUser(user);
    setForm(
      user
        ? { ...user, password: '' }
        : { username: '', email: '', full_name: '', is_admin: false, password: '', disabled: false }
    );
    setShowModal(true);
    
    console.log('AdminUsersPage: Modal state set to true');
  };

  const closeModal = () => {
    console.log('AdminUsersPage: Closing modal');
    setShowModal(false);
    setEditUser(null);
    setForm({ username: '', email: '', full_name: '', is_admin: false, password: '', disabled: false });
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : value;
    console.log(`AdminUsersPage: Form field changed - ${name}:`, newValue);
    setForm((prev) => ({ ...prev, [name]: newValue }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('AdminUsersPage: Submitting form for', editUser ? 'edit' : 'create', 'with data:', form);
    setError('');
    try {
      if (editUser) {
        console.log('AdminUsersPage: Updating user', editUser.id);
        const updateData = { ...form };
        if (!updateData.password) delete updateData.password;
        console.log('AdminUsersPage: Update data being sent:', updateData);
        await axios.patch(`${USERS_API_URL}/${editUser.id}`, updateData, {
          headers: { Authorization: `Bearer ${getToken()}` },
        });
        console.log('AdminUsersPage: User updated successfully');
      } else {
        console.log('AdminUsersPage: Creating new user with data:', form);
        await axios.post(USERS_API_URL, form, {
          headers: { Authorization: `Bearer ${getToken()}` },
        });
        console.log('AdminUsersPage: User created successfully');
      }
      fetchUsers();
      closeModal();
    } catch (err) {
      console.error('AdminUsersPage: Error saving user:', err.response?.data || err.message);
      console.error('AdminUsersPage: Full error object:', err);
      setError(err.response?.data?.detail || 'Failed to save user');
    }
  };

  const handleDelete = async (userId) => {
    const deleteFromAuth0 = window.confirm(
      'Do you want to delete this user from both the local database AND Auth0?\n\n' +
      'Click OK to delete from both (recommended for complete removal)\n' +
      'Click Cancel to delete only from local database'
    );
    
    if (!window.confirm(`Are you sure you want to delete this user?${deleteFromAuth0 ? ' This will permanently remove them from Auth0 as well.' : ''}`)) return;
    
    console.log('AdminUsersPage: Deleting user', userId, 'from Auth0:', deleteFromAuth0);
    
    try {
      const response = await axios.delete(`${USERS_API_URL}/${userId}?delete_from_auth0=${deleteFromAuth0}`, {
        headers: { Authorization: `Bearer ${getToken()}` },
      });
      
      console.log('AdminUsersPage: User deleted successfully', response.data);
      
      // Show detailed result message
      const result = response.data;
      let message = result.message;
      if (result.deletion_results?.errors?.length > 0) {
        message += `\n\nErrors: ${result.deletion_results.errors.join(', ')}`;
      }
      alert(message);
      
      fetchUsers();
    } catch (err) {
      console.error('AdminUsersPage: Error deleting user:', err.response?.data || err.message);
      const errorDetail = err.response?.data?.detail || err.response?.data?.message || 'Failed to delete user';
      alert(typeof errorDetail === 'string' ? errorDetail : JSON.stringify(errorDetail));
    }
  };

  const handleSyncFromAuth0 = async () => {
    if (loading) {
      console.log('AdminUsersPage: Sync already in progress, ignoring request');
      return;
    }
    
    if (!window.confirm('This will sync all users from Auth0 to the local database. Continue?')) return;
    
    console.log('AdminUsersPage: Syncing users from Auth0...');
    setLoading(true);
    
    try {
      const response = await axios.get(`${USERS_API_URL}/sync`, {
        headers: { Authorization: `Bearer ${getToken()}` },
      });
      
      console.log('AdminUsersPage: Sync completed', response.data);
      alert(response.data.message);
      await fetchUsers(); // Wait for users to be fetched
    } catch (err) {
      console.error('AdminUsersPage: Error syncing users:', err.response?.data || err.message);
      alert(err.response?.data?.detail || 'Failed to sync users from Auth0');
    } finally {
      setLoading(false);
    }
  };

  const handleBlockUser = async (userId, block = true) => {
    const action = block ? 'block' : 'unblock';
    if (!window.confirm(`Are you sure you want to ${action} this user in Auth0?`)) return;
    
    console.log(`AdminUsersPage: ${action}ing user`, userId);
    
    try {
      const response = await axios.post(`${USERS_API_URL}/${userId}/${action}`, {}, {
        headers: { Authorization: `Bearer ${getToken()}` },
      });
      
      console.log(`AdminUsersPage: User ${action}ed successfully`, response.data);
      alert(response.data.message);
      fetchUsers();
    } catch (err) {
      console.error(`AdminUsersPage: Error ${action}ing user:`, err.response?.data || err.message);
      alert(err.response?.data?.detail || `Failed to ${action} user`);
    }
  };

  return (
    <div className="page-container">
      {/* Header Section */}
      <div className="page-header">
        <h1 className="page-title">User Management</h1>
        <p className="page-subtitle">
          Manage user accounts, permissions, and access controls for your simulation platform
        </p>
      </div>

      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '2rem'
      }}>
        <h2 style={{ color: 'var(--color-charcoal)', margin: 0 }}>All Users</h2>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <button 
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              handleSyncFromAuth0();
            }} 
            className="btn-braun-secondary"
            disabled={loading}
            title="Sync all users from Auth0 to local database"
            style={{
              padding: '0.75rem 1.5rem',
              fontSize: '0.9rem',
              fontWeight: '500',
              opacity: loading ? 0.6 : 1,
              cursor: loading ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? '🔄 Syncing...' : '🔄 Sync from Auth0'}
          </button>
          <button 
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              openModal();
            }} 
            className="btn-braun-primary"
            style={{
              padding: '0.75rem 1.5rem',
              fontSize: '0.9rem',
              fontWeight: '500'
            }}
          >
            ➕ Add New User
          </button>
        </div>
      </div>

      {loading ? (
        <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
          <p style={{ color: 'var(--color-medium-grey)' }}>Loading users...</p>
        </div>
      ) : error ? (
        <div className="card error-card" style={{ marginBottom: '1rem' }}>
          <strong>Error:</strong> {typeof error === 'string' ? error : error?.message || 'An error occurred'}
        </div>
      ) : (
        <div className="card" style={{ padding: '1.5rem', overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ 
                borderBottom: '2px solid var(--color-border-light)',
                backgroundColor: 'var(--color-warm-white)'
              }}>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>ID</th>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>Username</th>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>Email</th>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>Full Name</th>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>Admin</th>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>Auth0 User</th>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>Status</th>
                <th style={{ 
                  padding: '1rem', 
                  textAlign: 'left', 
                  color: 'var(--color-charcoal)',
                  fontWeight: '600'
                }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((user) => (
                <tr key={user.id} style={{ 
                  borderBottom: '1px solid var(--color-border-light)',
                  '&:hover': { backgroundColor: 'var(--color-warm-white)' }
                }}>
                  <td style={{ padding: '1rem', color: 'var(--color-medium-grey)' }}>{user.id}</td>
                  <td style={{ padding: '1rem', color: 'var(--color-charcoal)', fontWeight: '500' }}>
                    {user.username}
                  </td>
                  <td style={{ padding: '1rem', color: 'var(--color-medium-grey)' }}>{user.email}</td>
                  <td style={{ padding: '1rem', color: 'var(--color-medium-grey)' }}>{user.full_name}</td>
                  <td style={{ padding: '1rem' }}>
                    <span style={{
                      padding: '0.25rem 0.75rem',
                      borderRadius: '20px',
                      fontSize: '0.8rem',
                      fontWeight: '600',
                      backgroundColor: user.is_admin ? 'var(--color-braun-orange)' : 'var(--color-light-grey)',
                      color: user.is_admin ? 'white' : 'var(--color-medium-grey)',
                    }}>
                      {user.is_admin ? 'Admin' : 'User'}
                    </span>
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <span style={{
                      padding: '0.25rem 0.75rem',
                      borderRadius: '20px',
                      fontSize: '0.8rem',
                      fontWeight: '600',
                      backgroundColor: user.auth0_user_id ? 'var(--color-braun-orange)' : 'var(--color-light-grey)',
                      color: user.auth0_user_id ? 'white' : 'var(--color-medium-grey)',
                    }}>
                      {user.auth0_user_id ? 'Yes' : 'Local'}
                    </span>
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <span style={{
                      padding: '0.25rem 0.75rem',
                      borderRadius: '20px',
                      fontSize: '0.8rem',
                      fontWeight: '600',
                      backgroundColor: user.disabled ? 'var(--color-error)' : 'var(--color-success)',
                      color: 'white',
                    }}>
                      {user.disabled ? 'Disabled' : 'Active'}
                    </span>
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <div style={{ 
                      display: 'flex', 
                      gap: '0.5rem', 
                      alignItems: 'center',
                      justifyContent: 'flex-start',
                      flexWrap: 'nowrap',
                      minWidth: '200px'
                    }}>
                      <button 
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          openModal(user);
                        }} 
                        className="btn-braun-secondary"
                        style={{ 
                          fontSize: '0.8rem', 
                          padding: '0.4rem 0.8rem',
                          minWidth: '60px',
                          whiteSpace: 'nowrap'
                        }}
                      >
                        Edit
                      </button>
                      {user.auth0_user_id && (
                        <button 
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            handleBlockUser(user.id, true);
                          }} 
                          style={{
                            fontSize: '0.8rem',
                            padding: '0.4rem 0.8rem',
                            backgroundColor: 'transparent',
                            border: '1px solid #ff8c00',
                            color: '#ff8c00',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            minWidth: '60px',
                            whiteSpace: 'nowrap',
                            transition: 'all 0.2s ease'
                          }}
                          onMouseOver={(e) => {
                            e.target.style.backgroundColor = '#ff8c0020';
                          }}
                          onMouseOut={(e) => {
                            e.target.style.backgroundColor = 'transparent';
                          }}
                          title="Block user in Auth0"
                        >
                          Block
                        </button>
                      )}
                      <button 
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          handleDelete(user.id);
                        }} 
                        style={{
                          fontSize: '0.8rem',
                          padding: '0.4rem 0.8rem',
                          backgroundColor: 'transparent',
                          border: '1px solid #dc3545',
                          color: '#dc3545',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          minWidth: '60px',
                          whiteSpace: 'nowrap',
                          transition: 'all 0.2s ease'
                        }}
                        onMouseOver={(e) => {
                          e.target.style.backgroundColor = '#dc354520';
                        }}
                        onMouseOut={(e) => {
                          e.target.style.backgroundColor = 'transparent';
                        }}
                        title={user.auth0_user_id ? 'Delete from local database and Auth0' : 'Delete from local database'}
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Modal for Add/Edit User */}
      <Modal isOpen={showModal} onClose={closeModal}>
          <div style={{ padding: '2rem', maxWidth: '500px' }}>
            <h3 style={{ 
              color: 'var(--color-charcoal)', 
              marginBottom: '1.5rem',
              fontSize: '1.5rem'
            }}>
              {editUser ? 'Edit User' : 'Add New User'}
            </h3>
            
            {error && (
              <div className="card error-card" style={{ marginBottom: '1rem' }}>
                {error}
              </div>
            )}
            
            <form onSubmit={handleSubmit}>
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem',
                  color: 'var(--color-charcoal)',
                  fontWeight: '500'
                }}>
                  Username:
                </label>
                <Input
                  type="text"
                  name="username"
                  value={form.username}
                  onChange={handleChange}
                  required
                />
              </div>
              
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem',
                  color: 'var(--color-charcoal)',
                  fontWeight: '500'
                }}>
                  Email:
                </label>
                <Input
                  type="email"
                  name="email"
                  value={form.email}
                  onChange={handleChange}
                  required
                />
              </div>
              
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem',
                  color: 'var(--color-charcoal)',
                  fontWeight: '500'
                }}>
                  Full Name:
                </label>
                <Input
                  type="text"
                  name="full_name"
                  value={form.full_name}
                  onChange={handleChange}
                />
              </div>
              
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ 
                  display: 'block', 
                  marginBottom: '0.5rem',
                  color: 'var(--color-charcoal)',
                  fontWeight: '500'
                }}>
                  Password {editUser && '(leave blank to keep current)'}:
                </label>
                <Input
                  type="password"
                  name="password"
                  value={form.password}
                  onChange={handleChange}
                  required={!editUser}
                />
              </div>
              
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ 
                  display: 'flex', 
                  alignItems: 'center',
                  color: 'var(--color-charcoal)',
                  fontWeight: '500'
                }}>
                  <input
                    type="checkbox"
                    name="is_admin"
                    checked={form.is_admin}
                    onChange={handleChange}
                    style={{
                      width: '18px',
                      height: '18px',
                      marginRight: '8px',
                      accentColor: 'var(--color-braun-orange)',
                    }}
                  />
                  Admin User
                </label>
              </div>
              
              <div style={{ marginBottom: '1.5rem' }}>
                <label style={{ 
                  display: 'flex', 
                  alignItems: 'center',
                  color: 'var(--color-charcoal)',
                  fontWeight: '500'
                }}>
                  <input
                    type="checkbox"
                    name="disabled"
                    checked={form.disabled}
                    onChange={handleChange}
                    style={{
                      width: '18px',
                      height: '18px',
                      marginRight: '8px',
                      accentColor: 'var(--color-braun-orange)',
                    }}
                  />
                  Disabled
                </label>
              </div>
              
              <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
                <button 
                  type="button" 
                  onClick={closeModal}
                  className="btn-braun-secondary"
                >
                  Cancel
                </button>
                <button 
                  type="submit"
                  className="btn-braun-primary"
                >
                  {editUser ? 'Update User' : 'Create User'}
                </button>
              </div>
            </form>
          </div>
      </Modal>
    </div>
  );
};

export default AdminUsersPage; 