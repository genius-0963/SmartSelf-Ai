"""
SmartShelf AI - NLP Intent Engine

Advanced intent recognition and entity extraction for natural language queries.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Intent types for retail queries."""
    FORECAST_DEMAND = "forecast_demand"
    ANALYZE_REVENUE = "analyze_revenue"
    CHECK_INVENTORY = "check_inventory"
    PRICING_STRATEGY = "pricing_strategy"
    CUSTOMER_SENTIMENT = "customer_sentiment"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    GENERAL_QUERY = "general_query"


class EntityType(Enum):
    """Entity types for extraction."""
    PRODUCT = "product"
    CATEGORY = "category"
    TIME = "time"
    METRIC = "metric"
    MONEY = "money"
    QUANTITY = "quantity"
    BRAND = "brand"
    SUPPLIER = "supplier"
    COMPARISON = "comparison"


@dataclass
class Entity:
    """Extracted entity from query."""
    type: EntityType
    value: str
    confidence: float
    start_pos: int
    end_pos: int
    normalized_value: Optional[str] = None


@dataclass
class IntentResult:
    """Intent recognition result."""
    primary_intent: IntentType
    secondary_intents: List[IntentType]
    confidence: float
    entities: List[Entity]
    query_type: str
    required_functions: List[str]
    disambiguation_needed: bool


class NLPIntentEngine:
    """Advanced NLP intent recognition and entity extraction."""
    
    def __init__(self):
        """Initialize NLP intent engine."""
        self.nlp = None
        self.intent_classifier = None
        
        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy NLP model loaded")
            except OSError:
                logger.warning("SpaCy English model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Transformer-based intent classification is intentionally optional.
        # Current implementation uses robust rule-based scoring to avoid
        # pulling large models during local demo/dev.
        if TRANSFORMERS_AVAILABLE:
            self.intent_classifier = None
        
        # Product and category mappings
        self.product_mappings = {
            "milk": {"product_id": "PRD-001", "category": "dairy"},
            "eggs": {"product_id": "PRD-015", "category": "dairy"},
            "bread": {"product_id": "PRD-023", "category": "bakery"},
            "coffee": {"product_id": "PRD-008", "category": "beverages"},
            "organic milk": {"product_id": "PRD-024", "category": "dairy"},
            "sourdough": {"product_id": "PRD-025", "category": "bakery"},
        }
        
        self.category_mappings = {
            "dairy": ["milk", "cheese", "yogurt", "eggs", "butter"],
            "bakery": ["bread", "croissants", "muffins", "cakes", "pastries"],
            "beverages": ["coffee", "tea", "juice", "soda", "water"],
            "produce": ["apples", "bananas", "carrots", "lettuce", "tomatoes"],
            "electronics": ["phones", "laptops", "tablets", "headphones", "chargers"],
            "clothing": ["shirts", "pants", "dresses", "shoes", "jackets"],
        }
        
        # Time expression patterns
        self.time_patterns = {
            "today": lambda: datetime.now().strftime("%Y-%m-%d"),
            "yesterday": lambda: (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "last week": lambda: (datetime.now() - timedelta(weeks=1)).strftime("%Y-%W"),
            "last month": lambda: (datetime.now() - timedelta(days=30)).strftime("%Y-%m"),
            "last quarter": lambda: f"Q{((datetime.now().month - 1) // 3) - 1 or 4}",
            "this month": lambda: datetime.now().strftime("%Y-%m"),
            "next week": lambda: (datetime.now() + timedelta(weeks=1)).strftime("%Y-%W"),
            "next month": lambda: (datetime.now() + timedelta(days=30)).strftime("%Y-%m"),
        }
        
        # Metric patterns
        self.metric_patterns = {
            "revenue": ["revenue", "sales", "income", "turnover", "earnings"],
            "profit": ["profit", "margin", "gain", "net income"],
            "inventory": ["inventory", "stock", "stockout", "supply"],
            "demand": ["demand", "forecast", "prediction", "projection"],
            "price": ["price", "cost", "pricing", "amount"],
            "volume": ["volume", "quantity", "units", "items sold"],
        }
        
        logger.info("NLP Intent Engine initialized")
    
    async def parse_query(self, raw_query: str, context_history: List[Dict] = None) -> IntentResult:
        """
        Parse user query and extract intent and entities.
        
        Args:
            raw_query: Raw user query
            context_history: Previous conversation context
            
        Returns:
            IntentResult with parsed information
        """
        try:
            # Preprocess query
            processed_query = self._preprocess_query(raw_query)
            
            # Extract entities
            entities = await self._extract_entities(processed_query)
            
            # Classify intent
            intent_result = await self._classify_intent(processed_query, entities)
            
            # Determine required functions
            required_functions = self._determine_required_functions(intent_result, entities)
            
            # Check if disambiguation is needed
            disambiguation_needed = self._needs_disambiguation(intent_result, entities)
            
            # Apply context if available
            if context_history:
                intent_result = self._apply_context(intent_result, context_history)
            
            result = IntentResult(
                primary_intent=intent_result["primary_intent"],
                secondary_intents=intent_result["secondary_intents"],
                confidence=intent_result["confidence"],
                entities=entities,
                query_type=intent_result["query_type"],
                required_functions=required_functions,
                disambiguation_needed=disambiguation_needed
            )
            
            logger.info(f"Query parsed: {result.primary_intent.value} (confidence: {result.confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            # Fallback to general query
            return IntentResult(
                primary_intent=IntentType.GENERAL_QUERY,
                secondary_intents=[],
                confidence=0.5,
                entities=[],
                query_type="general",
                required_functions=["general_search"],
                disambiguation_needed=False
            )
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better analysis."""
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Expand common contractions
        contractions = {
            "what's": "what is",
            "how's": "how is",
            "where's": "where is",
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
        }
        
        for contraction, expansion in contractions.items():
            query = query.replace(contraction, expansion)
        
        return query
    
    async def _extract_entities(self, query: str) -> List[Entity]:
        """Extract entities from query."""
        entities = []
        
        # Use spaCy if available for NER
        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                entity_type = self._map_spacy_entity(ent.label_)
                if entity_type:
                    entities.append(Entity(
                        type=entity_type,
                        value=ent.text,
                        confidence=0.8,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        normalized_value=self._normalize_entity_value(entity_type, ent.text)
                    ))
        
        # Custom entity extraction patterns
        entities.extend(self._extract_product_entities(query))
        entities.extend(self._extract_time_entities(query))
        entities.extend(self._extract_metric_entities(query))
        entities.extend(self._extract_money_entities(query))
        entities.extend(self._extract_quantity_entities(query))
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.start_pos)
        
        return entities
    
    def _map_spacy_entity(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our entity types."""
        mapping = {
            "PRODUCT": EntityType.PRODUCT,
            "ORG": EntityType.BRAND,
            "MONEY": EntityType.MONEY,
            "QUANTITY": EntityType.QUANTITY,
            "DATE": EntityType.TIME,
            "TIME": EntityType.TIME,
            "CARDINAL": EntityType.QUANTITY,
        }
        return mapping.get(spacy_label)
    
    def _extract_product_entities(self, query: str) -> List[Entity]:
        """Extract product and category entities."""
        entities = []
        
        # Check for exact product matches
        for product_name, product_info in self.product_mappings.items():
            if product_name in query:
                start_pos = query.find(product_name)
                entities.append(Entity(
                    type=EntityType.PRODUCT,
                    value=product_name,
                    confidence=0.9,
                    start_pos=start_pos,
                    end_pos=start_pos + len(product_name),
                    normalized_value=product_info["product_id"]
                ))
        
        # Check for category matches
        for category, products in self.category_mappings.items():
            if category in query:
                start_pos = query.find(category)
                entities.append(Entity(
                    type=EntityType.CATEGORY,
                    value=category,
                    confidence=0.85,
                    start_pos=start_pos,
                    end_pos=start_pos + len(category),
                    normalized_value=category
                ))
            
            # Check for products within categories
            for product in products:
                if product in query:
                    start_pos = query.find(product)
                    entities.append(Entity(
                        type=EntityType.PRODUCT,
                        value=product,
                        confidence=0.8,
                        start_pos=start_pos,
                        end_pos=start_pos + len(product),
                        normalized_value=product
                    ))
        
        return entities
    
    def _extract_time_entities(self, query: str) -> List[Entity]:
        """Extract time-related entities."""
        entities = []
        
        for time_expr, time_func in self.time_patterns.items():
            if time_expr in query:
                start_pos = query.find(time_expr)
                entities.append(Entity(
                    type=EntityType.TIME,
                    value=time_expr,
                    confidence=0.9,
                    start_pos=start_pos,
                    end_pos=start_pos + len(time_expr),
                    normalized_value=time_func()
                ))
        
        # Pattern for specific dates (YYYY-MM-DD, MM/DD, etc.)
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{1,2}/\d{1,2}',  # MM/DD
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                entities.append(Entity(
                    type=EntityType.TIME,
                    value=match.group(),
                    confidence=0.95,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=match.group()
                ))
        
        return entities
    
    def _extract_metric_entities(self, query: str) -> List[Entity]:
        """Extract metric-related entities."""
        entities = []
        
        for metric_type, keywords in self.metric_patterns.items():
            for keyword in keywords:
                if keyword in query:
                    start_pos = query.find(keyword)
                    entities.append(Entity(
                        type=EntityType.METRIC,
                        value=keyword,
                        confidence=0.8,
                        start_pos=start_pos,
                        end_pos=start_pos + len(keyword),
                        normalized_value=metric_type
                    ))
        
        return entities
    
    def _extract_money_entities(self, query: str) -> List[Entity]:
        """Extract money-related entities."""
        entities = []
        
        # Pattern for currency amounts
        money_patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # $1,234.56
            r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd)',  # 1,234.56 dollars
            r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:cents?)',  # 56 cents
        ]
        
        for pattern in money_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                entities.append(Entity(
                    type=EntityType.MONEY,
                    value=match.group(),
                    confidence=0.95,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=match.group().replace('$', '').replace(',', '')
                ))
        
        return entities
    
    def _extract_quantity_entities(self, query: str) -> List[Entity]:
        """Extract quantity-related entities."""
        entities = []
        
        # Pattern for quantities
        quantity_patterns = [
            r'\d+\s*(?:units?|items?|pieces?)',
            r'\d+\s*(?:%|percent)',
            r'\d+\s*(?:times?)',
        ]
        
        for pattern in quantity_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                entities.append(Entity(
                    type=EntityType.QUANTITY,
                    value=match.group(),
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=match.group()
                ))
        
        return entities
    
    def _normalize_entity_value(self, entity_type: EntityType, value: str) -> str:
        """Normalize entity values."""
        if entity_type == EntityType.PRODUCT:
            return self.product_mappings.get(value.lower(), {}).get("product_id", value)
        elif entity_type == EntityType.CATEGORY:
            return value.lower()
        elif entity_type == EntityType.MONEY:
            return value.replace('$', '').replace(',', '')
        else:
            return value.lower()
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.type, entity.value.lower())
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    async def _classify_intent(self, query: str, entities: List[Entity]) -> Dict[str, Any]:
        """Classify user intent."""
        # Rule-based intent classification
        intent_scores = {
            IntentType.FORECAST_DEMAND: 0.0,
            IntentType.ANALYZE_REVENUE: 0.0,
            IntentType.CHECK_INVENTORY: 0.0,
            IntentType.PRICING_STRATEGY: 0.0,
            IntentType.CUSTOMER_SENTIMENT: 0.0,
            IntentType.COMPETITOR_ANALYSIS: 0.0,
            IntentType.PRODUCT_RECOMMENDATION: 0.0,
            IntentType.TREND_ANALYSIS: 0.0,
            IntentType.ANOMALY_DETECTION: 0.0,
            IntentType.GENERAL_QUERY: 0.1,  # Base score
        }
        
        # Keyword-based scoring
        intent_keywords = {
            IntentType.FORECAST_DEMAND: ["forecast", "predict", "demand", "future", "expect", "projection"],
            IntentType.ANALYZE_REVENUE: ["revenue", "sales", "income", "earnings", "turnover", "performance"],
            IntentType.CHECK_INVENTORY: ["inventory", "stock", "stockout", "supply", "available", "quantity"],
            IntentType.PRICING_STRATEGY: ["price", "pricing", "cost", "discount", "margin", "profit"],
            IntentType.CUSTOMER_SENTIMENT: ["customer", "review", "feedback", "sentiment", "complaint", "satisfaction"],
            IntentType.COMPETITOR_ANALYSIS: ["competitor", "competition", "market", "compare", "vs", "versus"],
            IntentType.PRODUCT_RECOMMENDATION: ["recommend", "suggest", "best", "top", "popular", "show me"],
            IntentType.TREND_ANALYSIS: ["trend", "pattern", "change", "increase", "decrease", "growth"],
            IntentType.ANOMALY_DETECTION: ["anomaly", "unusual", "weird", "strange", "unexpected", "wrong"],
        }
        
        query_words = query.split()
        
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    intent_scores[intent] += 0.3
        
        # Entity-based scoring
        for entity in entities:
            if entity.type == EntityType.METRIC:
                if entity.normalized_value in ["revenue", "sales"]:
                    intent_scores[IntentType.ANALYZE_REVENUE] += 0.4
                elif entity.normalized_value in ["inventory", "stock"]:
                    intent_scores[IntentType.CHECK_INVENTORY] += 0.4
                elif entity.normalized_value in ["price", "cost"]:
                    intent_scores[IntentType.PRICING_STRATEGY] += 0.4
                elif entity.normalized_value in ["demand", "forecast"]:
                    intent_scores[IntentType.FORECAST_DEMAND] += 0.4
        
        # Find primary and secondary intents
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        primary_intent, primary_score = sorted_intents[0]
        
        secondary_intents = []
        for intent, score in sorted_intents[1:3]:  # Top 2 secondary
            if score > 0.2:
                secondary_intents.append(intent)
        
        # Determine query type
        query_type = self._determine_query_type(query, entities, primary_intent)
        
        return {
            "primary_intent": primary_intent,
            "secondary_intents": secondary_intents,
            "confidence": min(primary_score, 1.0),
            "query_type": query_type
        }
    
    def _determine_query_type(self, query: str, entities: List[Entity], primary_intent: IntentType) -> str:
        """Determine query type based on content."""
        if "?" in query or any(word in query for word in ["what", "how", "why", "when", "where"]):
            return "question"
        elif any(word in query for word in ["show", "display", "list", "get"]):
            return "retrieval"
        elif any(word in query for word in ["analyze", "compare", "explain"]):
            return "analysis"
        elif any(word in query for word in ["recommend", "suggest", "should"]):
            return "recommendation"
        elif primary_intent == IntentType.GENERAL_QUERY:
            return "general"
        else:
            return "task"
    
    def _determine_required_functions(self, intent_result: Dict[str, Any], entities: List[Entity]) -> List[str]:
        """Determine required functions based on intent and entities."""
        functions = []
        
        primary_intent = intent_result["primary_intent"]
        
        # Map intents to functions
        intent_functions = {
            IntentType.FORECAST_DEMAND: ["get_demand_forecast", "get_historical_sales"],
            IntentType.ANALYZE_REVENUE: ["get_sales_analytics", "get_revenue_trends"],
            IntentType.CHECK_INVENTORY: ["get_inventory_status", "get_stockout_alerts"],
            IntentType.PRICING_STRATEGY: ["get_pricing_analysis", "get_competitor_prices"],
            IntentType.CUSTOMER_SENTIMENT: ["analyze_customer_sentiment", "get_customer_reviews"],
            IntentType.COMPETITOR_ANALYSIS: ["get_competitor_analysis", "get_market_data"],
            IntentType.PRODUCT_RECOMMENDATION: ["semantic_product_search", "get_product_analytics"],
            IntentType.TREND_ANALYSIS: ["get_trend_analysis", "get_time_series_data"],
            IntentType.ANOMALY_DETECTION: ["detect_anomalies", "get_performance_metrics"],
            IntentType.GENERAL_QUERY: ["general_search", "get_business_overview"],
        }
        
        functions.extend(intent_functions.get(primary_intent, ["general_search"]))
        
        # Add entity-specific functions
        for entity in entities:
            if entity.type == EntityType.PRODUCT:
                functions.append(f"get_product_details(product_id={entity.normalized_value})")
            elif entity.type == EntityType.CATEGORY:
                functions.append(f"get_category_analytics(category={entity.normalized_value})")
            elif entity.type == EntityType.TIME:
                functions.append(f"filter_by_time_period(period={entity.normalized_value})")
        
        return list(set(functions))  # Remove duplicates
    
    def _needs_disambiguation(self, intent_result: Dict[str, Any], entities: List[Entity]) -> bool:
        """Check if query needs disambiguation."""
        # Low confidence intent
        if intent_result["confidence"] < 0.6:
            return True
        
        # Multiple high-confidence secondary intents
        if len(intent_result["secondary_intents"]) > 1:
            return True
        
        # Vague time references
        time_entities = [e for e in entities if e.type == EntityType.TIME]
        if not time_entities and any(word in intent_result["query_type"] for word in ["analyze", "forecast"]):
            return True
        
        # No product/category context for product-specific queries
        product_entities = [e for e in entities if e.type in [EntityType.PRODUCT, EntityType.CATEGORY]]
        if intent_result["primary_intent"] in [IntentType.PRODUCT_RECOMMENDATION, IntentType.CHECK_INVENTORY] and not product_entities:
            return True
        
        return False
    
    def _apply_context(self, intent_result: Dict[str, Any], context_history: List[Dict]) -> Dict[str, Any]:
        """Apply conversation context to intent result."""
        if not context_history:
            return intent_result
        
        # Get last few user queries
        recent_queries = [msg["content"].lower() for msg in context_history[-3:] if msg["role"] == "user"]
        
        # Check for entity references (pronouns, "it", "that", etc.)
        if any(word in " ".join(recent_queries) for word in ["milk", "dairy", "inventory"]):
            # User might be referring to previous topics
            if intent_result["primary_intent"] == IntentType.GENERAL_QUERY:
                # Upgrade to more specific intent based on context
                intent_result["primary_intent"] = IntentType.ANALYZE_REVENUE
                intent_result["confidence"] = 0.7
        
        return intent_result
