#!/usr/bin/env python3
"""
SmartShelf AI - ML Model Training Script

Trains all ML models for the retail analytics platform:
- Demand forecasting with Prophet
- Pricing optimization models
- Inventory intelligence models

Usage:
    python train_models.py --model all --data_dir data/raw --output_dir data/models
"""

import sys
import os
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json

# Add ML models to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ml_models"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_data(data_dir: Path) -> dict:
    """Load all training data from CSV files."""
    logger.info("Loading training data...")
    
    data = {}
    
    # Load products
    products_path = data_dir / "products.csv"
    if products_path.exists():
        data['products'] = pd.read_csv(products_path)
        data['products']['created_date'] = pd.to_datetime(data['products']['created_date'])
        logger.info(f"Loaded {len(data['products'])} products")
    
    # Load sales
    sales_path = data_dir / "sales.csv"
    if sales_path.exists():
        data['sales'] = pd.read_csv(sales_path)
        data['sales']['date'] = pd.to_datetime(data['sales']['date'])
        logger.info(f"Loaded {len(data['sales'])} sales records")
    
    # Load inventory
    inventory_path = data_dir / "inventory.csv"
    if inventory_path.exists():
        data['inventory'] = pd.read_csv(inventory_path)
        data['inventory']['date'] = pd.to_datetime(data['inventory']['date'])
        logger.info(f"Loaded {len(data['inventory'])} inventory records")
    
    # Load competitor pricing
    competitor_path = data_dir / "competitor_pricing.csv"
    if competitor_path.exists():
        data['competitor_pricing'] = pd.read_csv(competitor_path)
        data['competitor_pricing']['last_updated'] = pd.to_datetime(data['competitor_pricing']['last_updated'])
        logger.info(f"Loaded {len(data['competitor_pricing'])} competitor pricing records")
    
    return data


def train_demand_forecasting_models(data: dict, output_dir: Path) -> dict:
    """Train demand forecasting models."""
    logger.info("Training demand forecasting models...")
    
    from forecasting.prophet_model import DemandForecaster
    
    results = {}
    
    if 'sales' not in data:
        logger.error("No sales data available for demand forecasting")
        return results
    
    sales_data = data['sales'].copy()
    
    # Train models for top products by revenue
    top_products = sales_data.groupby('product_id')['total_revenue'].sum().nlargest(10).index.tolist()
    
    for product_id in top_products:
        logger.info(f"Training demand forecast model for product {product_id}")
        
        # Filter data for this product
        product_sales = sales_data[sales_data['product_id'] == product_id].copy()
        
        if len(product_sales) < 30:  # Need minimum data for Prophet
            logger.warning(f"Insufficient data for product {product_id}: {len(product_sales)} records")
            continue
        
        try:
            # Initialize and train model
            config = {
                'yearly_seasonality': True if len(product_sales) >= 365 else False,
                'weekly_seasonality': True,
                'seasonality_mode': 'multiplicative',
                'interval_width': 0.95,
                'include_holidays': True,
                'include_promotions': True
            }
            
            forecaster = DemandForecaster(config)
            
            # Prepare data
            X, y = forecaster.prepare_data(product_sales)
            
            # Train model
            training_metrics = forecaster.train(X, y)
            
            # Save model
            model_path = output_dir / f"demand_forecast_product_{product_id}.pkl"
            forecaster.save_model(str(model_path))
            
            # Generate sample forecast
            future_forecast = forecaster.forecast_future(periods=30)
            
            results[f'product_{product_id}'] = {
                'model_path': str(model_path),
                'training_metrics': training_metrics,
                'forecast_sample': future_forecast.head(7).to_dict('records'),
                'model_info': forecaster.get_model_summary()
            }
            
            logger.info(f"Successfully trained model for product {product_id}")
            
        except Exception as e:
            logger.error(f"Error training model for product {product_id}: {e}")
            continue
    
    # Train aggregate demand model (all products combined)
    logger.info("Training aggregate demand forecasting model...")
    
    try:
        aggregate_sales = sales_data.groupby('date').agg({
            'quantity_sold': 'sum',
            'total_revenue': 'sum',
            'promotion_active': 'any',
            'promotion_discount': 'mean'
        }).reset_index()
        
        if len(aggregate_sales) >= 30:
            aggregate_forecaster = DemandForecaster({
                'yearly_seasonality': True if len(aggregate_sales) >= 365 else False,
                'weekly_seasonality': True,
                'seasonality_mode': 'multiplicative',
                'interval_width': 0.95,
                'include_holidays': True,
                'include_promotions': True
            })
            
            X, y = aggregate_forecaster.prepare_data(aggregate_sales)
            training_metrics = aggregate_forecaster.train(X, y)
            
            model_path = output_dir / "demand_forecast_aggregate.pkl"
            aggregate_forecaster.save_model(str(model_path))
            
            future_forecast = aggregate_forecaster.forecast_future(periods=30)
            
            results['aggregate'] = {
                'model_path': str(model_path),
                'training_metrics': training_metrics,
                'forecast_sample': future_forecast.head(7).to_dict('records'),
                'model_info': aggregate_forecaster.get_model_summary()
            }
            
            logger.info("Successfully trained aggregate demand forecasting model")
    
    except Exception as e:
        logger.error(f"Error training aggregate demand model: {e}")
    
    return results


def train_pricing_optimization_models(data: dict, output_dir: Path) -> dict:
    """Train pricing optimization models."""
    logger.info("Training pricing optimization models...")
    
    results = {}
    
    if 'sales' not in data or 'products' not in data:
        logger.error("Missing sales or products data for pricing optimization")
        return results
    
    sales_data = data['sales'].copy()
    products_data = data['products'].copy()
    
    try:
        # Calculate price elasticity for products
        logger.info("Calculating price elasticity...")
        
        elasticity_results = {}
        
        # Get products with sufficient sales history
        product_sales_counts = sales_data.groupby('product_id').size()
        valid_products = product_sales_counts[product_sales_counts >= 10].index
        
        for product_id in valid_products[:20]:  # Top 20 products
            product_sales = sales_data[sales_data['product_id'] == product_id].copy()
            
            # Group by date to get daily price and quantity
            daily_data = product_sales.groupby('date').agg({
                'quantity_sold': 'sum',
                'unit_price': 'mean'
            }).reset_index()
            
            # Calculate price elasticity (simplified)
            if len(daily_data) >= 5:
                price_changes = daily_data['unit_price'].pct_change()
                quantity_changes = daily_data['quantity_sold'].pct_change()
                
                # Remove infinite and NaN values
                valid_idx = (price_changes != 0) & (~np.isinf(price_changes)) & (~np.isnan(price_changes)) & \
                           (quantity_changes != 0) & (~np.isinf(quantity_changes)) & (~np.isnan(quantity_changes))
                
                if valid_idx.sum() >= 3:
                    elasticity = (quantity_changes[valid_idx] / price_changes[valid_idx]).mean()
                    
                    elasticity_results[f'product_{product_id}'] = {
                        'elasticity': float(elasticity),
                        'data_points': int(valid_idx.sum()),
                        'avg_price': float(daily_data['unit_price'].mean()),
                        'avg_quantity': float(daily_data['quantity_sold'].mean())
                    }
        
        # Create pricing optimization model (simplified rule-based approach)
        logger.info("Creating pricing optimization model...")
        
        pricing_model = {
            'elasticity_analysis': elasticity_results,
            'optimization_rules': {
                'elastic_threshold': -1.0,  # Products more elastic than this
                'min_margin': 0.2,  # Minimum 20% margin
                'max_price_change': 0.3,  # Maximum 30% price change
                'competitor_sensitivity': 0.5
            },
            'model_type': 'rule_based_elasticity',
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Save pricing model
        pricing_model_path = output_dir / "pricing_optimization_model.json"
        with open(pricing_model_path, 'w') as f:
            json.dump(pricing_model, f, indent=2)
        
        results['pricing_optimization'] = {
            'model_path': str(pricing_model_path),
            'products_analyzed': len(elasticity_results),
            'model_type': 'rule_based_elasticity'
        }
        
        logger.info(f"Successfully analyzed pricing for {len(elasticity_results)} products")
        
    except Exception as e:
        logger.error(f"Error training pricing optimization model: {e}")
    
    return results


def train_inventory_intelligence_models(data: dict, output_dir: Path) -> dict:
    """Train inventory intelligence models."""
    logger.info("Training inventory intelligence models...")
    
    results = {}
    
    if 'inventory' not in data:
        logger.error("No inventory data available")
        return results
    
    inventory_data = data['inventory'].copy()
    
    try:
        # Analyze inventory patterns
        logger.info("Analyzing inventory patterns...")
        
        # Get daily snapshots for analysis
        daily_snapshots = inventory_data[inventory_data['transaction_type'] == 'daily_snapshot'].copy()
        
        if daily_snapshots.empty:
            logger.warning("No daily inventory snapshots found")
            return results
        
        # Calculate inventory metrics by product
        product_metrics = daily_snapshots.groupby('product_id').agg({
            'stock_level_after': ['mean', 'std', 'min', 'max'],
            'days_of_supply': ['mean', 'std', 'min'],
            'inventory_value': 'mean'
        }).round(2)
        
        # Flatten column names
        product_metrics.columns = ['_'.join(col).strip() for col in product_metrics.columns]
        product_metrics = product_metrics.reset_index()
        
        # Calculate reorder points (simplified)
        product_metrics['reorder_point'] = product_metrics['stock_level_after_mean'] * 0.3
        product_metrics['safety_stock'] = product_metrics['stock_level_after_std'] * 1.5
        product_metrics['optimal_stock'] = product_metrics['stock_level_after_mean'] * 1.5
        
        # Identify stockout risks
        stockout_risks = daily_snapshots[daily_snapshots['stock_level_after'] <= 5].groupby('product_id').size()
        product_metrics['stockout_frequency'] = product_metrics['product_id'].map(stockout_risks).fillna(0)
        
        # Create inventory intelligence model
        inventory_model = {
            'product_metrics': product_metrics.to_dict('records'),
            'model_parameters': {
                'reorder_threshold': 0.3,
                'safety_stock_multiplier': 1.5,
                'optimal_stock_multiplier': 1.5,
                'stockout_threshold': 5
            },
            'model_type': 'statistical_analysis',
            'analysis_period': {
                'start_date': daily_snapshots['date'].min().isoformat(),
                'end_date': daily_snapshots['date'].max().isoformat(),
                'total_days': len(daily_snapshots['date'].unique())
            },
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Save inventory model
        inventory_model_path = output_dir / "inventory_intelligence_model.json"
        with open(inventory_model_path, 'w') as f:
            json.dump(inventory_model, f, indent=2)
        
        results['inventory_intelligence'] = {
            'model_path': str(inventory_model_path),
            'products_analyzed': len(product_metrics),
            'model_type': 'statistical_analysis',
            'analysis_days': len(daily_snapshots['date'].unique())
        }
        
        logger.info(f"Successfully analyzed inventory for {len(product_metrics)} products")
        
    except Exception as e:
        logger.error(f"Error training inventory intelligence model: {e}")
    
    return results


def evaluate_models(data: dict, output_dir: Path) -> dict:
    """Evaluate all trained models."""
    logger.info("Evaluating trained models...")
    
    evaluation_results = {}
    
    # Evaluate demand forecasting models
    try:
        from forecasting.prophet_model import DemandForecaster
        
        # Load aggregate model for evaluation
        aggregate_model_path = output_dir / "demand_forecast_aggregate.pkl"
        if aggregate_model_path.exists():
            forecaster = DemandForecaster()
            forecaster.load_model(str(aggregate_model_path))
            
            # Get recent data for evaluation
            if 'sales' in data:
                sales_data = data['sales'].copy()
                recent_sales = sales_data.groupby('date').agg({
                    'quantity_sold': 'sum',
                    'total_revenue': 'sum',
                    'promotion_active': 'any',
                    'promotion_discount': 'mean'
                }).reset_index()
                
                # Use last 30 days for evaluation
                eval_data = recent_sales.tail(30)
                if len(eval_data) >= 7:
                    X_eval, y_eval = forecaster.prepare_data(eval_data)
                    eval_metrics = forecaster.evaluate(X_eval, y_eval)
                    
                    evaluation_results['demand_forecast_aggregate'] = eval_metrics
                    logger.info(f"Demand forecast evaluation - MAPE: {eval_metrics.get('mape', 'N/A'):.2f}%")
    
    except Exception as e:
        logger.error(f"Error evaluating demand forecasting models: {e}")
    
    return evaluation_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train SmartShelf AI ML models')
    parser.add_argument('--model', choices=['all', 'demand', 'pricing', 'inventory'], 
                       default='all', help='Model type to train')
    parser.add_argument('--data_dir', type=str, default='data/raw', 
                       help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='data/models', 
                       help='Directory to save trained models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    data = load_training_data(data_dir)
    
    if not data:
        logger.error("No training data found")
        return
    
    # Train models
    training_results = {}
    
    if args.model in ['all', 'demand']:
        training_results['demand_forecasting'] = train_demand_forecasting_models(data, output_dir)
    
    if args.model in ['all', 'pricing']:
        training_results['pricing_optimization'] = train_pricing_optimization_models(data, output_dir)
    
    if args.model in ['all', 'inventory']:
        training_results['inventory_intelligence'] = train_inventory_intelligence_models(data, output_dir)
    
    # Evaluate models
    evaluation_results = evaluate_models(data, output_dir)
    
    # Save training summary
    training_summary = {
        'training_timestamp': datetime.utcnow().isoformat(),
        'data_summary': {
            'products': len(data.get('products', [])),
            'sales_records': len(data.get('sales', [])),
            'inventory_records': len(data.get('inventory', [])),
            'competitor_prices': len(data.get('competitor_pricing', []))
        },
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'config': {
            'model_type': args.model,
            'data_dir': str(data_dir),
            'output_dir': str(output_dir)
        }
    }
    
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING SUMMARY")
    logger.info("="*50)
    
    for model_type, results in training_results.items():
        logger.info(f"\n{model_type.upper()}:")
        for model_name, model_info in results.items():
            if isinstance(model_info, dict) and 'model_path' in model_info:
                logger.info(f"  - {model_name}: {model_info['model_path']}")
    
    if evaluation_results:
        logger.info(f"\nEVALUATION RESULTS:")
        for model_name, metrics in evaluation_results.items():
            logger.info(f"  - {model_name}:")
            for metric_name, value in metrics.items():
                logger.info(f"    {metric_name}: {value:.4f}")
    
    logger.info(f"\nTraining completed! Models saved to: {output_dir}")
    logger.info(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
