import React from 'react'
import { TrendingUp } from 'lucide-react'

export default function TopProducts() {
  const products = [
    { id: 1, name: 'Laptop Pro', revenue: 125450, units: 125 },
    { id: 2, name: 'Wireless Mouse', revenue: 89750, units: 2995 },
    { id: 3, name: 'Office Chair', revenue: 67230, units: 338 },
    { id: 4, name: 'Desk Lamp', revenue: 45670, units: 1522 },
    { id: 5, name: 'USB Cable', revenue: 28423, units: 1895 },
  ]

  return (
    <div className="space-y-4">
      {products.map((product, index) => {
        const pct = Math.max(12, Math.min(100, Math.round((product.revenue / products[0].revenue) * 100)))
        const growth = Math.floor(Math.random() * 20 + 5)

        return (
          <div key={product.id} className="rounded-lg border border-border bg-gray-950/40 p-4 hover:border-green-500/30 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold bg-green-500/10 text-green-300 border border-green-500/20">
                  {index + 1}
                </div>
                <div>
                  <p className="font-medium text-foreground">{product.name}</p>
                  <p className="text-sm text-muted-foreground">{product.units.toLocaleString()} units</p>
                </div>
              </div>

              <div className="text-right">
                <p className="font-semibold text-foreground">${product.revenue.toLocaleString()}</p>
                <p className="text-sm text-green-400 flex items-center justify-end">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  +{growth}%
                </p>
              </div>
            </div>

            <div className="mt-3">
              <div className="h-2 rounded-full bg-gray-900 border border-border overflow-hidden">
                <div className="h-full bg-green-500/70" style={{ width: `${pct}%` }} />
              </div>
              <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                <span>Revenue share</span>
                <span>{pct}%</span>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
