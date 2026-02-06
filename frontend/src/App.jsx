import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { LayoutProvider } from './contexts/LayoutContext'
import { DataProvider } from './contexts/DataContext'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Inventory from './pages/Inventory'
import Forecasting from './pages/Forecasting'
import Pricing from './pages/Pricing'
import Analytics from './pages/Analytics'
import Copilot from './pages/Copilot'

function App() {
  return (
    <LayoutProvider>
      <DataProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/inventory" element={<Inventory />} />
            <Route path="/forecasting" element={<Forecasting />} />
            <Route path="/pricing" element={<Pricing />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/copilot" element={<Copilot />} />
          </Routes>
        </Layout>
      </DataProvider>
    </LayoutProvider>
  )
}

export default App
