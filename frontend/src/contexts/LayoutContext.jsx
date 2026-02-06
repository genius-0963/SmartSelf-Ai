import React, { createContext, useContext, useState } from 'react'

const LayoutContext = createContext()

export function LayoutProvider({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <LayoutContext.Provider value={{ sidebarOpen, setSidebarOpen }}>
      {children}
    </LayoutContext.Provider>
  )
}

export function useLayout() {
  const context = useContext(LayoutContext)
  if (!context) {
    throw new Error('useLayout must be used within a LayoutProvider')
  }
  return context
}
