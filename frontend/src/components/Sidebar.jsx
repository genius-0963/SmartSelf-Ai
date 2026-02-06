import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Package,
  TrendingUp,
  DollarSign,
  BarChart3,
  Bot,
  X,
  Brain,
  ShoppingCart,
  AlertTriangle
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Inventory', href: '/inventory', icon: Package },
  { name: 'Forecasting', href: '/forecasting', icon: TrendingUp },
  { name: 'Pricing', href: '/pricing', icon: DollarSign },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'AI Copilot', href: '/copilot', icon: Bot },
]

export default function Sidebar({ onClose }) {
  const location = useLocation()

  return (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-2">
          <Brain className="w-8 h-8 text-blue-600" />
          <div>
            <h1 className="text-lg font-bold text-gray-900">SmartShelf</h1>
            <p className="text-xs text-gray-500">AI Analytics</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-md lg:hidden hover:bg-gray-100"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href
          return (
            <NavLink
              key={item.name}
              to={item.href}
              onClick={onClose}
              className={`flex items-center space-x-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-blue-50 text-blue-700 border-r-2 border-blue-700'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
              }`}
            >
              <item.icon className="w-5 h-5" />
              <span>{item.name}</span>
            </NavLink>
          )
        })}
      </nav>

      {/* Quick Stats */}
      <div className="p-4 border-t">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
          Quick Stats
        </h3>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Active Products</span>
            <span className="text-sm font-medium text-gray-900">50</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Alerts</span>
            <span className="text-sm font-medium text-red-600">5</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Revenue (30d)</span>
            <span className="text-sm font-medium text-green-600">$609K</span>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t">
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span className="text-xs text-gray-500">System Online</span>
        </div>
        <p className="text-xs text-gray-400 mt-1">Version 1.0.0</p>
      </div>
    </div>
  )
}
