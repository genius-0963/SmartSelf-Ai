import React, { useState, useEffect } from 'react'
import { BarChart3, TrendingUp, PieChart, Activity } from 'lucide-react'

export default function Analytics() {
  const [analytics, setAnalytics] = useState({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => {
      setAnalytics({
        revenue: 609523,
        growth: 12.5,
        topCategory: 'Electronics',
        conversionRate: 3.2,
        avgOrderValue: 45.67
      })
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
        <h1 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h1>
        <p className="text-gray-600">Comprehensive business intelligence and insights</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center space-x-3 mb-4">
            <BarChart3 className="w-6 h-6 text-blue-600" />
            <h3 className="font-semibold">Total Revenue</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">${analytics.revenue.toLocaleString()}</p>
          <p className="text-sm text-green-600">+{analytics.growth}% growth</p>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center space-x-3 mb-4">
            <TrendingUp className="w-6 h-6 text-green-600" />
            <h3 className="font-semibold">Growth Rate</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">{analytics.growth}%</p>
          <p className="text-sm text-gray-600">Month over month</p>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center space-x-3 mb-4">
            <PieChart className="w-6 h-6 text-purple-600" />
            <h3 className="font-semibold">Top Category</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">{analytics.topCategory}</p>
          <p className="text-sm text-gray-600">By revenue</p>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center space-x-3 mb-4">
            <Activity className="w-6 h-6 text-orange-600" />
            <h3 className="font-semibold">Conversion Rate</h3>
          </div>
          <p className="text-2xl font-bold text-gray-900">{analytics.conversionRate}%</p>
          <p className="text-sm text-gray-600">Visitors to customers</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Revenue by Category</h2>
          <div className="h-64 flex items-center justify-center bg-gray-50 rounded">
            <p className="text-gray-500">Chart placeholder</p>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Sales Trend</h2>
          <div className="h-64 flex items-center justify-center bg-gray-50 rounded">
            <p className="text-gray-500">Chart placeholder</p>
          </div>
        </div>
      </div>
    </div>
  )
}
