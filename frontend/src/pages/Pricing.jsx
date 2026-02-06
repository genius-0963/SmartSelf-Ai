import React, { useState, useEffect } from 'react'
import { DollarSign, TrendingUp, TrendingDown, Target } from 'lucide-react'

export default function Pricing() {
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => {
      setRecommendations([
        { id: 1, product: 'Laptop Pro', current: 999, recommended: 1099, impact: 245.50, confidence: 87 },
        { id: 2, product: 'Wireless Mouse', current: 29.99, recommended: 27.99, impact: 189.25, confidence: 92 },
        { id: 3, product: 'Office Chair', current: 199, recommended: 189, impact: 156.75, confidence: 78 },
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
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Pricing Optimization</h1>
        <p className="text-gray-600">AI-powered pricing recommendations and analysis</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center space-x-3 mb-4">
            <DollarSign className="w-6 h-6 text-green-600" />
            <h3 className="font-semibold">Total Impact</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">$592</p>
          <p className="text-sm text-gray-600">Potential revenue increase</p>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center space-x-3 mb-4">
            <Target className="w-6 h-6 text-blue-600" />
            <h3 className="font-semibold">Opportunities</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">8</p>
          <p className="text-sm text-gray-600">Products to optimize</p>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center space-x-3 mb-4">
            <TrendingUp className="w-6 h-6 text-purple-600" />
            <h3 className="font-semibold">Avg. Price Change</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">5.2%</p>
          <p className="text-sm text-gray-600">Recommended adjustment</p>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center space-x-3 mb-4">
            <Target className="w-6 h-6 text-orange-600" />
            <h3 className="font-semibold">Confidence</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">86%</p>
          <p className="text-sm text-gray-600">Average confidence score</p>
        </div>
      </div>

      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Pricing Recommendations</h2>
        <div className="space-y-4">
          {recommendations.map(rec => (
            <div key={rec.id} className="border rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium text-gray-900">{rec.product}</h3>
                  <div className="flex items-center space-x-4 mt-2">
                    <span className="text-sm text-gray-600">Current: ${rec.current}</span>
                    <span className="text-sm text-gray-600">Recommended: ${rec.recommended}</span>
                    <span className="text-sm text-gray-600">Impact: ${rec.impact}</span>
                    <span className="text-sm text-gray-600">Confidence: {rec.confidence}%</span>
                  </div>
                </div>
                <div className="text-right">
                  <span className={`inline-flex items-center px-2 py-1 text-xs font-medium rounded ${
                    rec.recommended > rec.current ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                  }`}>
                    {rec.recommended > rec.current ? 'Increase' : 'Decrease'}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
