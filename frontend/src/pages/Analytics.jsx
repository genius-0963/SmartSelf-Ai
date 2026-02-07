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
        <h1 className="text-2xl font-bold text-foreground">Analytics Dashboard</h1>
        <p className="text-muted-foreground">Comprehensive business intelligence and insights</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="bg-card border border-border p-6 rounded-lg">
          <div className="flex items-center space-x-3 mb-4">
            <BarChart3 className="w-6 h-6 text-green-400" />
            <h3 className="font-semibold text-foreground">Total Revenue</h3>
          </div>
          <p className="text-2xl font-bold text-foreground">${analytics.revenue.toLocaleString()}</p>
          <p className="text-sm text-green-400">+{analytics.growth}% growth</p>
        </div>

        <div className="bg-card border border-border p-6 rounded-lg">
          <div className="flex items-center space-x-3 mb-4">
            <TrendingUp className="w-6 h-6 text-green-400" />
            <h3 className="font-semibold text-foreground">Growth Rate</h3>
          </div>
          <p className="text-2xl font-bold text-foreground">{analytics.growth}%</p>
          <p className="text-sm text-muted-foreground">Month over month</p>
        </div>

        <div className="bg-card border border-border p-6 rounded-lg">
          <div className="flex items-center space-x-3 mb-4">
            <PieChart className="w-6 h-6 text-green-400" />
            <h3 className="font-semibold text-foreground">Top Category</h3>
          </div>
          <p className="text-2xl font-bold text-foreground">{analytics.topCategory}</p>
          <p className="text-sm text-muted-foreground">By revenue</p>
        </div>

        <div className="bg-card border border-border p-6 rounded-lg">
          <div className="flex items-center space-x-3 mb-4">
            <Activity className="w-6 h-6 text-green-400" />
            <h3 className="font-semibold text-foreground">Conversion Rate</h3>
          </div>
          <p className="text-2xl font-bold text-foreground">{analytics.conversionRate}%</p>
          <p className="text-sm text-muted-foreground">Visitors to customers</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-card border border-border p-6 rounded-lg">
          <h2 className="text-lg font-semibold text-foreground mb-4">Revenue by Category</h2>
          <div className="h-64 flex items-center justify-center bg-gray-950/50 border border-border rounded">
            <p className="text-muted-foreground">Chart coming soon</p>
          </div>
        </div>

        <div className="bg-card border border-border p-6 rounded-lg">
          <h2 className="text-lg font-semibold text-foreground mb-4">Sales Trend</h2>
          <div className="h-64 flex items-center justify-center bg-gray-950/50 border border-border rounded">
            <p className="text-muted-foreground">Chart coming soon</p>
          </div>
        </div>
      </div>
    </div>
  )
}
