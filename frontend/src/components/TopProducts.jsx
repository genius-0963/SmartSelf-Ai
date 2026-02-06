import React from 'react'
import { TrendingUp, Package } from 'lucide-react'

export default function TopProducts() {
  const products = [
    { id: 1, name: 'Laptop Pro', revenue: 125450, units: 125 },
    { id: 2, name: 'Wireless Mouse', revenue: 89750, units: 2995 },
    { id: 3, name: 'Office Chair', revenue: 67230, units: 338 },
    { id: 4, name: 'Desk Lamp', revenue: 45670, units: 1522 },
    { id: 5, name: 'USB Cable', revenue: 28423, units: 1895 },
  ]

  return (
    <div className="space-y-3">
      {products.map((product, index) => (
        <div key={product.id} className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              index === 0 ? 'bg-yellow-100 text-yellow-800' :
              index === 1 ? 'bg-gray-100 text-gray-800' :
              index === 2 ? 'bg-orange-100 text-orange-800' :
              'bg-blue-100 text-blue-800'
            }`}>
              {index + 1}
            </div>
            <div>
              <p className="font-medium text-gray-900">{product.name}</p>
              <p className="text-sm text-gray-600">{product.units} units</p>
            </div>
          </div>
          <div className="text-right">
            <p className="font-medium text-gray-900">${product.revenue.toLocaleString()}</p>
            <p className="text-sm text-green-600 flex items-center">
              <TrendingUp className="w-3 h-3 mr-1" />
              +{Math.floor(Math.random() * 20 + 5)}%
            </p>
          </div>
        </div>
      ))}
    </div>
  )
}
