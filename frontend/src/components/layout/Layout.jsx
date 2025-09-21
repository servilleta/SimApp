import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';

const Layout = ({ children }) => {
  const { logout } = useAuth0();

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <img className="h-8 w-auto" src="/logo.svg" alt="SimApp.ai" />
              </div>
            </div>
            <div className="flex items-center">
              <button
                onClick={() => logout({ returnTo: window.location.origin })}
                className="ml-4 px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  );
};

export default Layout; 