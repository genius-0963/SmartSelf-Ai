import React from 'react'
import { AlertTriangle, Package } from 'lucide-react'

export default function InventoryAlerts() {
  const alerts = [
    { id: 1, product: 'Office Chair', type: 'stockout', severity: 'critical' },
    { id: 2, product: 'Wireless Mouse', type: 'low_stock', severity: 'high' },
    { id: 3, product: 'Desk Lamp', type: 'overstock', severity: 'medium' },
  ]

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Inventory Alerts</h2>
      <div className="space-y-3">
        {alerts.map(alert => (
          <div key={alert.id} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
            <Package className="w-5 h-5 text-gray-400" />
            <div className="flex-1">
              <p className="font-medium text-gray-900">{alert.product}</p>
              <p className="text-sm text-gray-600">{alert.type.replace('_', ' ')}</p>
            </div>
            <span className={`px-2 py-1 text-xs font-medium rounded ${
              alert.severity === 'critical' ? 'bg-red-100 text-red-800' :
              alert.severity === 'high' ? 'bg-yellow-100 text-yellow-800' :
              'bg-blue-100 text-blue-800'
            }`}>
              {alert.severity}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
