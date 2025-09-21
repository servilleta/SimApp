// 🔄 ADMIN STATUS REFRESH SCRIPT
// Run this in the browser console to refresh your admin status

console.log('🔄 Refreshing admin status...');

// Clear Redux store admin state
if (window.__REDUX_DEVTOOLS_EXTENSION__ && window.store) {
    console.log('🔄 Clearing Redux auth state...');
    window.store.dispatch({ type: 'auth/clearUser' });
}

// Clear localStorage admin cache
Object.keys(localStorage).forEach(key => {
    if (key.includes('admin') || key.includes('user') || key.includes('auth')) {
        console.log(`🗑️ Removing ${key} from localStorage`);
        localStorage.removeItem(key);
    }
});

// Clear sessionStorage 
Object.keys(sessionStorage).forEach(key => {
    if (key.includes('admin') || key.includes('user') || key.includes('auth')) {
        console.log(`🗑️ Removing ${key} from sessionStorage`);
        sessionStorage.removeItem(key);
    }
});

console.log('✅ Admin status cache cleared!');
console.log('🔄 Please refresh the page (Ctrl+R) to load updated admin status');

// Auto refresh after 2 seconds
setTimeout(() => {
    console.log('🔄 Auto-refreshing page...');
    window.location.reload();
}, 2000);
