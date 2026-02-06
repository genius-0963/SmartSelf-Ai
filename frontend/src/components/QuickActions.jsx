import React from 'react'
import { Link } from 'react-router-dom'
import { Package, TrendingUp, DollarSign, Bot } from 'lucide-react'

export default function QuickActions() {
  const actions = [
    { title: 'Check Inventory', description: 'View stock levels', icon: Package, href: '/inventory', color: 'blue' },
    { title: 'View Forecasts', description: 'Demand predictions', icon: TrendingUp, href: '/forecasting', color: 'green' },
    { title: 'Pricing Insights', description: 'Optimize prices', icon: DollarSign, href: '/pricing', color: 'purple' },
    { title: 'Ask AI Copilot', description: 'Get assistance', icon: Bot, href: '/copilot', color: 'orange' },
  ]

  const getColorClasses = (color) => {
    const colors = {
      blue: 'bg-blue-100 text-blue-700 hover:bg-blue-200',
      green: 'bg-green-100 text-green-700 hover:bg-green-200',
      purple: 'bg-purple-100 text-purple-700 hover:bg-purple-200',
      orange: 'bg-orange-100 text-orange-700 hover:bg-orange-200',
    }
    return colors[color] || 'bg-gray-100 text-gray-700 hover:bg-gray-200'
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
      <div className="space-y-2">
        {actions.map(action => (
          <Link
            key={action.title}
            to={action.href}
            className={`flex items-center space-x-3 p-3 rounded-lg transition-colors ${getColorClasses(action.color)}`}
          >
            <action.icon className="w-5 h-5" />
            <div>
              <p className="font-medium">{action.title}</p>
              <p className="text-sm opacity-75">{action.description}</p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
