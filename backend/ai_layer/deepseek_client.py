"""
DeepSeek LLM Client for AI Layer
Provides standardized interface to DeepSeek API for intelligent analysis
"""

import logging
import json
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
import time

logger = logging.getLogger(__name__)

class DeepSeekClient:
    """
    Client for DeepSeek LLM API with async support and error handling
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"
        self.session = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        logger.info("✅ DeepSeek client initialized")
    
    async def get_session(self, extended_timeout: bool = False):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            # Use longer timeout for comprehensive analysis
            timeout_config = aiohttp.ClientTimeout(
                total=120 if extended_timeout else 5,  # 2 minutes for comprehensive analysis
                connect=10 if extended_timeout else 2,  # Connection timeout
                sock_connect=10 if extended_timeout else 2,  # Socket connection timeout
                sock_read=110 if extended_timeout else 5  # Socket read timeout
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout_config,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self.session
    
    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from markdown code blocks or return response as-is
        
        Args:
            response: Raw response text that might contain ```json blocks
            
        Returns:
            Clean JSON string
        """
        # Remove markdown code blocks if present
        if "```json" in response:
            # Find the start and end of the JSON block
            start_marker = "```json"
            end_marker = "```"
            
            start_idx = response.find(start_marker)
            if start_idx != -1:
                start_idx += len(start_marker)
                end_idx = response.find(end_marker, start_idx)
                if end_idx != -1:
                    json_content = response[start_idx:end_idx].strip()
                    logger.debug(f"🔧 [DEEPSEEK] Extracted JSON from code block: {len(json_content)} chars")
                    return json_content
        
        # Return original response if no code blocks found
        return response.strip()
    
    async def chat_completion(self, 
                            messages: List[Dict[str, str]], 
                            max_tokens: int = 150,
                            temperature: float = 0.3,
                            extended_timeout: bool = False) -> Optional[str]:
        """
        Send chat completion request to DeepSeek API
        
        Args:
            messages: List of messages in OpenAI format
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0.0-1.0)
            
        Returns:
            Response text or None if failed
        """
        try:
            # Rate limiting
            now = time.time()
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
            
            session = await self.get_session(extended_timeout=extended_timeout)
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            logger.debug(f"🤖 [DEEPSEEK] Sending request: {len(messages)} messages, max_tokens={max_tokens}")
            
            async with session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                self.last_request_time = time.time()
                
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    logger.debug(f"✅ [DEEPSEEK] Response received: {len(content)} characters")
                    return content.strip()
                else:
                    error_text = await response.text()
                    logger.warning(f"❌ [DEEPSEEK] API error {response.status}: {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning("⏰ [DEEPSEEK] Request timeout")
            return None
        except Exception as e:
            logger.error(f"❌ [DEEPSEEK] Request failed: {e}")
            return None
    
    async def analyze_excel_context(self, 
                                  cell_address: str,
                                  formula: str,
                                  cell_type: str,
                                  sheet_name: str) -> Optional[str]:
        """Generate business context for Excel cell"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a business analyst expert at interpreting Excel models. Provide concise, business-focused descriptions."
            },
            {
                "role": "user", 
                "content": f"""Analyze this Excel cell and provide a brief business context description:

Sheet: {sheet_name}
Cell: {cell_address}
Formula: {formula}
Cell Type: {cell_type}

Provide a 1-2 sentence description of what this cell likely represents in a business model.
Focus on the business purpose, not the technical formula details."""
            }
        ]
        
        return await self.chat_completion(messages, max_tokens=100)
    
    async def suggest_distribution_parameters(self,
                                            cell_address: str,
                                            current_value: Any,
                                            context: str,
                                            description: str) -> Optional[Dict[str, Any]]:
        """Get AI-powered distribution parameter suggestions"""
        
        logger.info(f"🎯 [DEEPSEEK] Requesting distribution parameters for {cell_address}")
        
        messages = [
            {
                "role": "system",
                "content": "You are a financial modeling expert. Suggest probability distribution parameters for Monte Carlo simulation based on business context. Respond only with valid JSON."
            },
            {
                "role": "user",
                "content": f"""Suggest probability distribution parameters for this Monte Carlo variable:

Cell: {cell_address}
Current Value: {current_value}
Context: {context}
Description: {description}

Consider:
- Business context and typical variation for this type of variable
- Current value as baseline
- Appropriate distribution type (triangular, normal, uniform, beta)

Respond with JSON only:
{{
    "distribution_type": "triangular|normal|uniform|beta",
    "parameters": {{
        "min_value": number,
        "max_value": number,
        "most_likely": number,
        "mean": number,
        "std_dev": number
    }},
    "reasoning": "brief explanation"
}}"""
            }
        ]
        
        response = await self.chat_completion(messages, max_tokens=200)
        if response:
            try:
                logger.debug(f"📝 [DEEPSEEK] Raw response for {cell_address}: {response[:200]}...")
                # Extract JSON from markdown code blocks if present
                json_text = self._extract_json_from_response(response)
                result = json.loads(json_text)
                logger.info(f"✅ [DEEPSEEK] Successfully parsed parameters for {cell_address}")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ [DEEPSEEK] Invalid JSON response for {cell_address}: {e}")
                logger.debug(f"Raw response: {response}")
        else:
            logger.warning(f"🚫 [DEEPSEEK] No response received for {cell_address}")
        return None
    
    async def generate_business_justification(self,
                                            cell_address: str,
                                            current_value: Any,
                                            context: str,
                                            distribution_type: str) -> Optional[str]:
        """Generate business justification for variable selection"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a business analyst. Provide concise justifications for Monte Carlo simulation variables."
            },
            {
                "role": "user",
                "content": f"""Provide a brief business justification for including this variable in Monte Carlo simulation:

Cell: {cell_address}
Value: {current_value}
Context: {context}
Distribution: {distribution_type}

Explain in 1-2 sentences why this variable should be varied and how it impacts the model."""
            }
        ]
        
        return await self.chat_completion(messages, max_tokens=80)
    
    async def generate_simulation_insights(self,
                                         stats: Dict[str, float],
                                         risk_metrics: Dict[str, float],
                                         distribution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate business insights from simulation results"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a financial analyst. Analyze Monte Carlo simulation results and provide business insights. Respond only with valid JSON array."
            },
            {
                "role": "user",
                "content": f"""Analyze these Monte Carlo simulation results and provide business insights:

Statistics:
- Mean: {stats.get('mean', 0):.2f}
- Median: {stats.get('median', 0):.2f}
- Standard Deviation: {stats.get('std', 0):.2f}
- Coefficient of Variation: {stats.get('cv', 0):.1%}

Risk Metrics:
- 5th Percentile: {risk_metrics.get('p5', 0):.2f}
- 95th Percentile: {risk_metrics.get('p95', 0):.2f}
- Value at Risk (5%): {risk_metrics.get('var_5', 0):.2f}

Distribution:
- Shape: {distribution.get('shape', 'unknown')}
- Skewness: {distribution.get('skewness', 0):.2f}
- Kurtosis: {distribution.get('kurtosis', 0):.2f}

Provide 1-2 key business insights in JSON format:
[
    {{
        "title": "Insight Title",
        "description": "Brief description",
        "business_impact": "Impact on business",
        "risk_level": "high|medium|low"
    }}
]"""
            }
        ]
        
        response = await self.chat_completion(messages, max_tokens=300)
        if response:
            try:
                # Extract JSON from markdown code blocks if present
                json_text = self._extract_json_from_response(response)
                return json.loads(json_text)
            except json.JSONDecodeError:
                logger.warning(f"⚠️ [DEEPSEEK] Invalid JSON response: {response}")
        return []
    
    async def generate_executive_summary(self,
                                       stats: Dict[str, float],
                                       risk_metrics: Dict[str, float],
                                       success_prob: float,
                                       key_insights: List[str]) -> Optional[str]:
        """Generate executive summary of results"""
        
        insights_text = "; ".join(key_insights[:3])
        
        messages = [
            {
                "role": "system",
                "content": "You are a business executive's assistant. Write concise executive summaries of Monte Carlo simulation results."
            },
            {
                "role": "user",
                "content": f"""Write a concise executive summary for Monte Carlo simulation results:

Key Results:
- Mean outcome: {stats.get('mean', 0):.2f}
- Success probability: {success_prob:.1%}
- Risk range: {risk_metrics.get('p5', 0):.2f} to {risk_metrics.get('p95', 0):.2f} (90% confidence)
- Key insights: {insights_text}

Write 2-3 sentences suitable for executive presentation."""
            }
        ]
        
        return await self.chat_completion(messages, max_tokens=120)
    
    async def analyze_complete_excel_model(self,
                                         workbook_data: Dict[str, Any],
                                         file_id: str,
                                         sheet_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Comprehensive Excel model analysis by DeepSeek
        Sends entire model structure for intelligent analysis
        """
        
        logger.info(f"🎯 [DEEPSEEK] Analyzing complete Excel model for {file_id}")
        
        # Extract model structure
        model_structure = self._extract_model_structure(workbook_data, sheet_name)
        
        messages = [
            {
                "role": "system",
                "content": """You are an Excel model analyzer. Analyze the model structure and identify input/output cells for Monte Carlo simulation.

Focus on technical analysis:
1. Count sheets, cells, formulas, and dependencies
2. Identify input cells (non-formula cells that feed into calculations)
3. Identify output cells (formula cells that represent key results)
4. Suggest appropriate probability distributions based on data patterns

Respond ONLY with valid JSON matching the exact structure provided."""
            },
            {
                "role": "user", 
                "content": f"""Analyze this Excel model structure:

MODEL STRUCTURE:
{json.dumps(model_structure, indent=2)}

Provide analysis with this JSON structure:
{{
    "model_kpis": {{
        "active_sheets": 0,
        "total_cells": 0,
        "input_cells": 0,
        "output_cells": 0,
        "formula_cells": 0,
        "structure_description": "Brief technical description of model structure and calculation flow"
    }},
    "input_variables": [
        {{
            "cell_address": "A1",
            "sheet_name": "Sheet1",
            "variable_name": "Descriptive name from nearby labels",
            "current_value": 100,
            "data_type": "number|percentage|currency",
            "distribution": {{
                "type": "triangular|normal|uniform|beta|lognormal",
                "parameters": {{
                    "min_value": 80,
                    "max_value": 120,
                    "most_likely": 100
                }},
                "reasoning": "Statistical justification for distribution choice"
            }},
            "referenced_by": ["C2", "D5"]
        }}
    ],
    "output_targets": [
        {{
            "cell_address": "Z10",
            "sheet_name": "Sheet1",
            "variable_name": "Descriptive name from nearby labels",
            "current_value": 50000,
            "formula": "=X10*Y10",
            "data_type": "number|percentage|currency",
            "depends_on": ["X10", "Y10"]
        }}
    ]
}}"""
            }
        ]
        
        try:
            # Comprehensive analysis needs more time than individual requests
            response = await asyncio.wait_for(
                self.chat_completion(messages, max_tokens=2000, extended_timeout=True),
                timeout=120.0  # 2 minute timeout for comprehensive analysis
            )
            if response:
                logger.debug(f"📝 [DEEPSEEK] Raw model analysis response: {response[:300]}...")
                # Extract JSON from markdown code blocks if present
                json_text = self._extract_json_from_response(response)
                result = json.loads(json_text)
                logger.info(f"✅ [DEEPSEEK] Successfully analyzed complete model for {file_id}")
                return result
        except asyncio.TimeoutError:
            logger.warning(f"⏰ [DEEPSEEK] Comprehensive model analysis timed out after 2 minutes for {file_id}")
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ [DEEPSEEK] Invalid JSON in model analysis: {e}")
            logger.debug(f"Raw response: {response}")
        except Exception as e:
            logger.error(f"❌ [DEEPSEEK] Model analysis failed: {e}")
        
        return None
    
    def _extract_model_structure(self, workbook_data: Dict[str, Any], target_sheet: str = None) -> Dict[str, Any]:
        """Extract comprehensive model structure for DeepSeek analysis"""
        
        structure = {
            "sheets": {},
            "model_metadata": {
                "total_sheets": 0,
                "total_formulas": 0,
                "total_values": 0
            }
        }
        
        sheets_data = workbook_data.get('sheets', {})
        
        # Focus on target sheet if specified, otherwise analyze all
        target_sheets = {target_sheet: sheets_data[target_sheet]} if target_sheet and target_sheet in sheets_data else sheets_data
        
        for sheet_name, sheet_data in target_sheets.items():
            sheet_structure = {
                "name": sheet_name,
                "cells": {},
                "formulas": [],
                "values": [],
                "statistics": {
                    "formula_count": 0,
                    "value_count": 0
                }
            }
            
            # Process grid data if available
            if 'data' in sheet_data and sheet_data['data']:
                grid_data = sheet_data['data']
                for row_idx, row in enumerate(grid_data):
                    for col_idx, cell_value in enumerate(row):
                        if cell_value is not None and cell_value != '':
                            cell_address = f"{chr(65 + col_idx)}{row_idx + 1}"
                            
                            # Extract actual value from CellData object if needed
                            actual_value = cell_value
                            formula = None
                            
                            if hasattr(cell_value, 'value'):
                                # This is a CellData object
                                actual_value = cell_value.value
                                if hasattr(cell_value, 'formula') and cell_value.formula:
                                    formula = cell_value.formula
                            elif isinstance(cell_value, dict):
                                # This is a dictionary representation
                                actual_value = cell_value.get('value', cell_value.get('display_value'))
                                formula = cell_value.get('formula')
                            
                            # Check if it's a formula cell
                            if formula and isinstance(formula, str) and formula.startswith('='):
                                sheet_structure["formulas"].append({
                                    "cell": cell_address,
                                    "formula": formula,
                                    "type": "formula",
                                    "calculated_value": actual_value
                                })
                                sheet_structure["statistics"]["formula_count"] += 1
                                
                                sheet_structure["cells"][cell_address] = {
                                    "value": actual_value,
                                    "formula": formula,
                                    "is_formula": True
                                }
                            else:
                                # Regular value cell
                                sheet_structure["values"].append({
                                    "cell": cell_address,
                                    "value": actual_value,
                                    "type": type(actual_value).__name__
                                })
                                sheet_structure["statistics"]["value_count"] += 1
                                
                                sheet_structure["cells"][cell_address] = {
                                    "value": actual_value,
                                    "is_formula": False
                                }
            
            # Also check sheet_data object if available
            if 'sheet_data' in sheet_data:
                sd = sheet_data['sheet_data']
                if hasattr(sd, 'grid_data'):
                    # Additional processing if needed
                    pass
            
            structure["sheets"][sheet_name] = sheet_structure
            structure["model_metadata"]["total_formulas"] += sheet_structure["statistics"]["formula_count"]
            structure["model_metadata"]["total_values"] += sheet_structure["statistics"]["value_count"]
        
        structure["model_metadata"]["total_sheets"] = len(structure["sheets"])
        
        return structure

# Global DeepSeek client instance
_deepseek_client = None

def get_deepseek_client() -> Optional[DeepSeekClient]:
    """Get the global DeepSeek client instance"""
    global _deepseek_client
    return _deepseek_client

def initialize_deepseek_client(api_key: str) -> DeepSeekClient:
    """Initialize the global DeepSeek client"""
    global _deepseek_client
    _deepseek_client = DeepSeekClient(api_key)
    return _deepseek_client

async def cleanup_deepseek_client():
    """Cleanup the DeepSeek client session"""
    global _deepseek_client
    if _deepseek_client:
        await _deepseek_client.close()
        _deepseek_client = None
