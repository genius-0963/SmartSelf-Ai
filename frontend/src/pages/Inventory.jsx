import React, { useState, useEffect } from 'react'
import {
  Package,
  AlertTriangle,
  TrendingDown,
  Search,
  Filter,
  Download,
  RefreshCw
} from 'lucide-react'

export default function Inventory() {
  const [inventory, setInventory] = useState([])
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      setInventory([
        { id: 1, name: 'Laptop Pro', sku: 'ELE001', stock: 45, optimal: 50, status: 'optimal' },
        { id: 2, name: 'Wireless Mouse', sku: 'ELE002', stock: 8, optimal: 30, status: 'low' },
        { id: 3, name: 'Office Chair', sku: 'FUR001', stock: 0, optimal: 25, status: 'out' },
        { id: 4, name: 'Desk Lamp', sku: 'FUR002', stock: 67, optimal: 40, status: 'overstock' },
        { id: 5, name: 'USB Cable', sku: 'ELE003', stock: 23, optimal: 20, status: 'optimal' },
      ])
      setAlerts([
        { id: 1, product: 'Office Chair', type: 'stockout', severity: 'critical', message: 'Product out of stock' },
        { id: 2, product: 'Wireless Mouse', type: 'low_stock', severity: 'high', message: 'Only 8 units remaining' },
      ])
      setLoading(false)
    }, 1000)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner w-8 h-8"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Inventory Management</h1>
          <p className="text-gray-600">Monitor and manage your product inventory</p>
        </div>
        <div className="flex space-x-2">
          <button className="btn-secondary flex items-center space-x-2">
            <Download className="w-4 h-4" />
            <span>Export</span>
          </button>
          <button className="btn-primary flex items-center space-x-2">
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-lg font-semibold text-gray-900">Active Alerts</h2>
          {alerts.map(alert => (
            <div key={alert.id} className={`p-4 rounded-lg border ${
              alert.severity === 'critical' ? 'bg-red-50 border-red-200' : 'bg-yellow-50 border-yellow-200'
            }`}>
              <div className="flex items-center space-x-3">
                <AlertTriangle className={`w-5 h-5 ${
                  alert.severity === 'critical' ? 'text-red-600' : 'text-yellow-600'
                }`} />
                <div className="flex-1">
                  <p className="font-medium text-gray-900">{alert.product}</p>
                  <p className="text-sm text-gray-600">{alert.message}</p>
                </div>
                <span className={`px-2 py-1 text-xs font-medium rounded ${
                  alert.severity === 'critical' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'
                }`}>
                  {alert.severity}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Inventory Table */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center space-x-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search products..."
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg w-full"
              />
            </div>
            <button className="btn-secondary flex items-center space-x-2">
              <Filter className="w-4 h-4" />
              <span>Filter</span>
            </button>
          </div>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Product
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  SKU
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Current Stock
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Optimal Stock
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {inventory.map(item => (
                <tr key={item.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <Package className="w-5 h-5 text-gray-400 mr-3" />
                      <span className="font-medium text-gray-900">{item.name}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {item.sku}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`font-medium ${
                      item.stock === 0 ? 'text-red-600' : 
                      item.stock < item.optimal * 0.3 ? 'text-yellow-600' : 
                      'text-green-600'
                    }`}>
                      {item.stock}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {item.optimal}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 text-xs font-medium rounded ${
                      item.status === 'optimal' ? 'bg-green-100 text-green-800' :
                      item.status === 'low' ? 'bg-yellow-100 text-yellow-800' :
                      item.status === 'out' ? 'bg-red-100 text-red-800' :
                      'bg-blue-100 text-blue-800'
                    }`}>
                      {item.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button className="text-blue-600 hover:text-blue-900 mr-3">Reorder</button>
                    <button className="text-gray-600 hover:text-gray-900">Details</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
