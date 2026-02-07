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
        <h1 className="text-2xl font-bold text-foreground">Demand Forecasting</h1>
        <p className="text-muted-foreground">AI-powered demand predictions and insights</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-card border border-border p-6 rounded-lg">
          <div className="flex items-center space-x-3 mb-4">
            <Target className="w-6 h-6 text-green-400" />
            <h3 className="font-semibold text-foreground">Forecast Accuracy</h3>
          </div>
          <p className="text-2xl font-bold text-foreground">87%</p>
          <p className="text-sm text-muted-foreground">Average across all products</p>
        </div>

        <div className="bg-card border border-border p-6 rounded-lg">
          <div className="flex items-center space-x-3 mb-4">
            <TrendingUp className="w-6 h-6 text-green-400" />
            <h3 className="font-semibold text-foreground">Growth Prediction</h3>
          </div>
          <p className="text-2xl font-bold text-foreground">+15%</p>
          <p className="text-sm text-muted-foreground">Expected next month</p>
        </div>

        <div className="bg-card border border-border p-6 rounded-lg">
          <div className="flex items-center space-x-3 mb-4">
            <Calendar className="w-6 h-6 text-green-400" />
            <h3 className="font-semibold text-foreground">Forecast Horizon</h3>
          </div>
          <p className="text-2xl font-bold text-foreground">30</p>
          <p className="text-sm text-muted-foreground">Days ahead</p>
        </div>
      </div>

      <div className="bg-card border border-border rounded-lg p-6">
        <h2 className="text-lg font-semibold text-foreground mb-4">Product Forecasts</h2>
        <div className="space-y-4">
          {forecasts.map(forecast => (
            <div key={forecast.id} className="border border-border rounded-lg p-4 hover:border-green-500/30 transition-colors">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium text-foreground">{forecast.product}</h3>
                  <div className="flex items-center space-x-4 mt-2">
                    <span className="text-sm text-muted-foreground">Current: {forecast.current}</span>
                    <span className="text-sm text-muted-foreground">Predicted: {forecast.predicted}</span>
                    <span className="text-sm text-green-400">Confidence: {forecast.confidence}%</span>
                  </div>
                </div>
                <div className="text-right">
                  <span className={`inline-flex items-center px-2 py-1 text-xs font-medium rounded ${
                    forecast.trend === 'up' ? 'bg-green-500/15 text-green-200 border border-green-500/20' :
                    forecast.trend === 'down' ? 'bg-red-500/15 text-red-200 border border-red-500/20' :
                    'bg-gray-500/15 text-gray-200 border border-gray-500/20'
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
