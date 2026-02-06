import React, { createContext, useContext, useState, useEffect } from 'react'

const DataContext = createContext()

export function DataProvider({ children }) {
  const [data, setData] = useState({
    products: [],
    sales: [],
    inventory: [],
    forecasts: [],
    loading: true
  })

  useEffect(() => {
    // Simulate initial data loading
    setTimeout(() => {
      setData(prev => ({
        ...prev,
        loading: false
      }))
    }, 1000)
  }, [])

  return (
    <DataContext.Provider value={{ data, setData }}>
      {children}
    </DataContext.Provider>
  )
}

export function useData() {
  const context = useContext(DataContext)
  if (!context) {
    throw new Error('useData must be used within a DataProvider')
  }
  return context
}
