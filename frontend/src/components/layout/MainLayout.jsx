import React, { Suspense, useState } from 'react';
import { Outlet } from 'react-router-dom';
import Footer from './Footer';
import Sidebar from './Sidebar';

const Loading = () => (
  <div style={{ 
    display: 'flex', 
    alignItems: 'center', 
    justifyContent: 'center', 
    height: '100%',
    color: '#6b7280',
    fontSize: '14px'
  }}>
    Loading...
  </div>
);

const MainLayout = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="app-container" style={{ 
      display: 'flex', 
      minHeight: '100vh',
      backgroundColor: '#ffffff',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", sans-serif'
    }}>
      <Sidebar collapsed={sidebarCollapsed} setCollapsed={setSidebarCollapsed} />
      <main style={{ 
        flexGrow: 1, 
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
        backgroundColor: '#ffffff'
      }}>
        <div style={{ 
          flex: 1, 
          padding: '24px',
          overflow: 'auto'
        }}>
          <Suspense fallback={<Loading />}>
            <Outlet context={{ setSidebarCollapsed }} />
          </Suspense>
        </div>
        <Footer />
      </main>
    </div>
  );
};

export default MainLayout; 