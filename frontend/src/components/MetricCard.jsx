import React from 'react'
import { ArrowUp, ArrowDown, Minus } from 'lucide-react'

export default function MetricCard({ title, value, change, trend, icon: Icon, format, color }) {
  const formatValue = (val) => {
    if (format === 'currency') {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
      }).format(val)
    }
    return new Intl.NumberFormat('en-US').format(val)
  }

  const getTrendIcon = () => {
    if (trend === 'up') return <ArrowUp className="w-4 h-4" />
    if (trend === 'down') return <ArrowDown className="w-4 h-4" />
    return <Minus className="w-4 h-4" />
  }

  const getTrendColor = () => {
    if (trend === 'up') return 'text-green-600'
    if (trend === 'down') return 'text-red-600'
    return 'text-gray-600'
  }

  const getIconColor = () => {
    const colors = {
      green: 'text-green-600 bg-green-100',
      blue: 'text-blue-600 bg-blue-100',
      purple: 'text-purple-600 bg-purple-100',
      red: 'text-red-600 bg-red-100',
    }
    return colors[color] || 'text-gray-600 bg-gray-100'
  }

  return (
    <div className="metric-card">
      <div className="flex items-center justify-between">
        <div>
          <p className="metric-label">{title}</p>
          <p className="metric-value">{formatValue(value)}</p>
        </div>
        <div className={`p-3 rounded-lg ${getIconColor()}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
      <div className="flex items-center mt-4">
        <span className={`metric-change ${getTrendColor()} flex items-center`}>
          {getTrendIcon()}
          {Math.abs(change)}%
        </span>
        <span className="text-sm text-gray-500 ml-2">vs last period</span>
      </div>
    </div>
  )
}
