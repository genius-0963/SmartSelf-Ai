import React, { useState, useEffect } from 'react'
import { TrendingUp, Calendar, Target, BarChart3 } from 'lucide-react'

export default function Forecasting() {
  const [forecasts, setForecasts] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => {
      setForecasts([
        { id: 1, product: 'Laptop Pro', current: 45, predicted: 52, confidence: 87, trend: 'up' },
        { id: 2, product: 'Wireless Mouse', current: 23, predicted: 28, confidence: 92, trend: 'up' },
        { id: 3, product: 'Office Chair', current: 15, predicted: 18, confidence: 78, trend: 'stable' },
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
        <h1 className="text-2xl font-bold text-gray-900">Demand Forecasting</h1>
        <p className="text-gray-600">AI-powered demand predictions and insights</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center space-x-3 mb-4">
            <Target className="w-6 h-6 text-blue-600" />
            <h3 className="font-semibold">Forecast Accuracy</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">87%</p>
          <p className="text-sm text-gray-600">Average across all products</p>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center space-x-3 mb-4">
            <TrendingUp className="w-6 h-6 text-green-600" />
            <h3 className="font-semibold">Growth Prediction</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">+15%</p>
          <p className="text-sm text-gray-600">Expected next month</p>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center space-x-3 mb-4">
            <Calendar className="w-6 h-6 text-purple-600" />
            <h3 className="font-semibold">Forecast Horizon</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">30</p>
          <p className="text-sm text-gray-600">Days ahead</p>
        </div>
      </div>

      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Product Forecasts</h2>
        <div className="space-y-4">
          {forecasts.map(forecast => (
            <div key={forecast.id} className="border rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium text-gray-900">{forecast.product}</h3>
                  <div className="flex items-center space-x-4 mt-2">
                    <span className="text-sm text-gray-600">Current: {forecast.current}</span>
                    <span className="text-sm text-gray-600">Predicted: {forecast.predicted}</span>
                    <span className="text-sm text-gray-600">Confidence: {forecast.confidence}%</span>
                  </div>
                </div>
                <div className="text-right">
                  <span className={`inline-flex items-center px-2 py-1 text-xs font-medium rounded ${
                    forecast.trend === 'up' ? 'bg-green-100 text-green-800' :
                    forecast.trend === 'down' ? 'bg-red-100 text-red-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {forecast.trend}
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
