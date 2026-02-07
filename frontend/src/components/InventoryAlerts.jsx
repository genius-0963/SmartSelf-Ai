import React from 'react'
import { AlertTriangle, Package, ArrowRight } from 'lucide-react'

export default function InventoryAlerts() {
  const alerts = [
    { id: 1, product: 'Office Chair', type: 'stockout', severity: 'critical' },
    { id: 2, product: 'Wireless Mouse', type: 'low_stock', severity: 'high' },
    { id: 3, product: 'Desk Lamp', type: 'overstock', severity: 'medium' },
  ]

  return (
    <div className="bg-card border border-border p-6 rounded-lg">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-foreground">Inventory Alerts</h2>
        <button className="text-sm text-green-400 hover:text-green-300 inline-flex items-center">
          View all
          <ArrowRight className="w-4 h-4 ml-1" />
        </button>
      </div>
      <div className="space-y-3">
        {alerts.map(alert => (
          <div key={alert.id} className="flex items-center space-x-3 p-3 bg-gray-950/50 border border-border rounded-lg hover:border-green-500/30 transition-colors">
            <Package className="w-5 h-5 text-muted-foreground" />
            <div className="flex-1">
              <p className="font-medium text-foreground">{alert.product}</p>
              <p className="text-sm text-muted-foreground">{alert.type.replace('_', ' ')}</p>
            </div>
            <span className={`px-2 py-1 text-xs font-medium rounded ${
              alert.severity === 'critical' ? 'bg-red-500/15 text-red-200 border border-red-500/20' :
              alert.severity === 'high' ? 'bg-yellow-500/15 text-yellow-200 border border-yellow-500/20' :
              'bg-green-500/15 text-green-200 border border-green-500/20'
            }`}>
              {alert.severity}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
