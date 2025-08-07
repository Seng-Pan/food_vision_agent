
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Optional, List, Dict, Any
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
import wikipedia
import requests
import base64
import json
import os
from PIL import Image
import hashlib
import io
import time
from datetime import datetime

# Configure API keys
OPENAI_API_KEY = "Your OpenAI API Key" 
TAVILY_API_KEY = "Your Tavily Search api key" 
CHAT_MODEL = "gpt-4o-mini"

# Set up Tavily Search API 
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm = ChatOpenAI(model=CHAT_MODEL, openai_api_key=os.environ["OPENAI_API_KEY"])

# ===== Data Models =====
class FoodAnalysis(BaseModel):
    """Structured response for food image analysis"""
    dish_name: str = Field(description="Name of the identified dish")
    description: str = Field(description="Detailed description of the food")
    main_ingredients: List[str] = Field(description="List of main ingredients identified")
    cuisine_type: Optional[str] = Field(description="Type of cuisine (e.g., Italian, Chinese)")
    confidence_score: float = Field(description="Confidence in identification (0-1)")

class NutritionInfo(BaseModel):
    """Nutrition information for food items"""
    calories_per_serving: int = Field(description="Estimated calories per serving")
    protein_g: float = Field(description="Protein content in grams")
    carbs_g: float = Field(description="Carbohydrate content in grams")
    fat_g: float = Field(description="Fat content in grams")
    fiber_g: Optional[float] = Field(description="Fiber content in grams")
    sugar_g: Optional[float] = Field(description="Sugar content in grams")
    serving_size: str = Field(description="Estimated serving size")

class DietCompatibility(BaseModel):
    """Diet compatibility analysis"""
    is_vegan: bool = Field(description="Whether the dish is vegan")
    is_vegetarian: bool = Field(description="Whether the dish is vegetarian")
    is_gluten_free: bool = Field(description="Whether the dish is gluten-free")
    is_keto_friendly: bool = Field(description="Whether the dish is keto-friendly")
    is_diabetic_friendly: bool = Field(description="Whether the dish is suitable for diabetics")
    allergens: List[str] = Field(description="List of common allergens present")
    dietary_notes: str = Field(description="Additional dietary considerations")

class ImageContext(BaseModel):
    """Context for each analyzed image"""
    image_hash: str = Field(description="Hash of the image for identification")
    analysis: FoodAnalysis = Field(description="Food analysis results")
    nutrition: Optional[NutritionInfo] = Field(description="Nutrition information")
    diet_info: Optional[DietCompatibility] = Field(description="Diet compatibility info")
    upload_timestamp: str = Field(description="When the image was uploaded")

class ChatState(TypedDict):
    """State management for the chatbot"""
    messages: Annotated[list, add_messages]
    current_image: Optional[str]
    current_image_hash: Optional[str]
    image_contexts: Dict[str, ImageContext]
    user_query_type: str

# ====== UTILITY FUNCTIONS ======
def get_image_hash(image_base64: str) -> str:
    """Generate a hash for image identification"""
    return hashlib.md5(image_base64.encode()).hexdigest()[:12]

def format_timestamp() -> str:
    """Get current timestamp"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def search_with_tavily(query: str) -> str:
    """Search using Tavily API for food history information"""
    try:
        if TAVILY_API_KEY == "your_tavily_api_key_here":
            # Fallback to Wikipedia if Tavily not configured
            return search_with_wikipedia(query)
        
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": 3,
            "include_domains": ["wikipedia.org", "britannica.com", "history.com", "smithsonianmag.com"]
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            
            # Extract the answer or combine results
            if data.get("answer"):
                return data["answer"]
            
            # Combine top results if no direct answer
            results = data.get("results", [])
            if results:
                combined = ""
                for result in results[:2]:  # Use top 2 results
                    if result.get("content"):
                        combined += f"{result['content']}\n\n"
                return combined.strip()
        
        # Fallback to Wikipedia
        return search_with_wikipedia(query)
        
    except Exception as e:
        print(f"Tavily search error: {e}")
        return search_with_wikipedia(query)

def search_with_wikipedia(query: str) -> str:
    """Fallback Wikipedia search for food history"""
    try:
        # Search for the dish
        search_results = wikipedia.search(f"{query} food dish history", results=3)
        
        if search_results:
            # Try to get the page for the first result
            page = wikipedia.page(search_results[0])
            
            # Get a summary that's informative but not too long
            summary = wikipedia.summary(search_results[0], sentences=8)
            
            return f"**Source: Wikipedia**\n\n{summary}"
        else:
            return f"I couldn't find specific historical information about {query}. This might be a regional dish or a variation with limited documentation."
            
    except wikipedia.exceptions.DisambiguationError as e:
        # Try the first option from disambiguation
        try:
            page = wikipedia.page(e.options[0])
            summary = wikipedia.summary(e.options[0], sentences=6)
            return f"**Source: Wikipedia**\n\n{summary}"
        except:
            return f"I found multiple entries for {query} but couldn't retrieve specific information. Could you be more specific about the type or region?"
    
    except wikipedia.exceptions.PageError:
        return f"I couldn't find detailed information about {query}. This might be a regional dish or a modern variation."
    
    except Exception as e:
        return f"I encountered an issue searching for information about {query}. Please try rephrasing your question."

# ===== Agents =====
class ImageAnalyzerAgent:
    """Enhanced agent for analyzing food images using GPT-4o vision"""
    
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """You are an expert food analyst with deep knowledge of global cuisines, including ethnic and regional dishes. When analyzing food images:

1. Look carefully at visual details: ingredients, cooking methods, presentation style, garnishes
2. Consider regional and ethnic cuisine markers (spices, cooking vessels, traditional preparations)
3. For unfamiliar dishes, describe what you see accurately rather than guessing
4. Pay attention to cultural food presentation styles
5. Be specific about ingredients you can identify visually
6. If unsure about exact dish name, provide a descriptive name based on visible elements
7. Consider traditional cooking methods that might affect identification

Be especially careful with:
- Southeast Asian cuisines (Thai, Vietnamese, Burmese, Malaysian, etc.)
- Regional Indian dishes
- African cuisines
- Lesser-known ethnic foods
- Traditional preparation methods

Your confidence score should reflect your actual certainty about the identification."""
    
    def analyze_food_image(self, image_base64: str) -> FoodAnalysis:
        """Analyze a food image and return structured analysis"""
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": "Please analyze this food image carefully. Pay special attention to ethnic and regional cuisine characteristics:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ])
            ]
            
            structured_llm = self.llm.with_structured_output(FoodAnalysis)
            result = structured_llm.invoke(messages)
            return result
        except Exception as e:
            return FoodAnalysis(
                dish_name="Unknown dish",
                description=f"Could not analyze image: {str(e)}",
                main_ingredients=["Unknown"],
                cuisine_type=None,
                confidence_score=0.0
            )

class CalorieEstimatorAgent:
    """Agent for estimating nutrition information"""
    
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """You are a nutrition expert with knowledge of global cuisines. Based on the food analysis provided, 
        estimate detailed nutrition information including calories, macronutrients, and serving size.
        
        Consider:
        - Traditional cooking methods and their impact on nutrition
        - Regional ingredients and their nutritional profiles
        - Typical portion sizes for different cuisines
        - Oil, sugar, and salt content common in various ethnic cuisines
        
        Be realistic in your estimates and account for preparation methods."""
    
    def estimate_nutrition(self, food_analysis: FoodAnalysis) -> NutritionInfo:
        """Estimate nutrition information based on food analysis"""
        try:
            prompt = f"""
            Food: {food_analysis.dish_name}
            Description: {food_analysis.description}
            Main ingredients: {', '.join(food_analysis.main_ingredients)}
            Cuisine: {food_analysis.cuisine_type or 'Unknown'}
            
            Please provide detailed nutrition estimates for a typical serving of this dish.
            Consider traditional preparation methods and regional cooking styles.
            """
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            structured_llm = self.llm.with_structured_output(NutritionInfo)
            result = structured_llm.invoke(messages)
            return result
        except Exception as e:
            return NutritionInfo(
                calories_per_serving=0,
                protein_g=0.0,
                carbs_g=0.0,
                fat_g=0.0,
                serving_size="Unknown"
            )

class DietCompatibilityAgent:
    """Agent for checking diet compatibility"""
    
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """You are a dietary specialist with knowledge of global food traditions. 
        Analyze the provided food information and determine its compatibility with various diets.
        
        Consider:
        - Traditional ingredients used in regional cuisines
        - Hidden ingredients common in ethnic preparations
        - Religious and cultural dietary restrictions
        - Traditional cooking methods (use of ghee, fish sauce, etc.)
        
        Be conservative in your assessments - when in doubt, assume not compatible."""
    
    def check_diet_compatibility(self, food_analysis: FoodAnalysis) -> DietCompatibility:
        """Check diet compatibility based on food analysis"""
        try:
            prompt = f"""
            Food: {food_analysis.dish_name}
            Description: {food_analysis.description}
            Main ingredients: {', '.join(food_analysis.main_ingredients)}
            Cuisine: {food_analysis.cuisine_type or 'Unknown'}
            
            Analyze this food for dietary compatibility, considering traditional preparation methods
            and common hidden ingredients in this type of cuisine.
            """
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            structured_llm = self.llm.with_structured_output(DietCompatibility)
            result = structured_llm.invoke(messages)
            return result
        except Exception as e:
            return DietCompatibility(
                is_vegan=False,
                is_vegetarian=False,
                is_gluten_free=False,
                is_keto_friendly=False,
                is_diabetic_friendly=False,
                allergens=["Unknown"],
                dietary_notes=f"Could not analyze dietary compatibility: {str(e)}"
            )

class FoodHistoryAgent:
    """Enhanced agent for food history with web search capabilities"""
    
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """You are a food historian and cultural expert. Using the provided search results,
        create comprehensive, engaging information about dishes including:
        
        1. Origins and historical background
        2. Cultural significance and traditions
        3. Regional variations if applicable
        4. Interesting facts or stories
        5. Evolution of the dish over time
        
        Structure your response well and cite the information appropriately.
        If the search results don't contain enough information, be honest about limitations."""
    
    def get_food_history(self, dish_name: str, cuisine_type: Optional[str] = None) -> str:
        """Get detailed historical and cultural information about a dish using web search"""
        try:
            # Create search query
            search_query = f"{dish_name} food history origin culture"
            if cuisine_type:
                search_query += f" {cuisine_type} cuisine"
            
            # Get search results
            search_results = search_with_tavily(search_query)
            
            # Process with LLM to create structured response
            prompt = f"""
            Based on the following search results about {dish_name}, create an engaging and informative
            response about the dish's history, culture, and significance:
            
            Search Results:
            {search_results}
            
            Dish: {dish_name}
            Cuisine Type: {cuisine_type or 'Not specified'}
            
            Create a well-structured response that includes:
            - Historical origins and development
            - Cultural significance and traditions
            - Regional variations (if mentioned)
            - Interesting facts or stories
            - How it became popular globally (if applicable)
            
            If the search results are limited, be honest about what information is available.
            """
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"I apologize, but I encountered an issue retrieving information about {dish_name}. This might be a very regional dish or the search service is temporarily unavailable. Could you provide more details about the cuisine type or region?"

class QueryAnalyzer:
    """Improved query analysis"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def analyze_query(self, user_text: str, has_current_image: bool, current_analysis_exists: bool) -> Dict[str, Any]:
        """Analyze user query and determine required actions"""
        
        analysis_prompt = f"""
        Analyze this user query and determine what actions are needed:
        
        User Query: "{user_text}"
        Has Image Available: {has_current_image}
        Food Already Identified: {current_analysis_exists}
        
        Determine which of these actions are needed (can be multiple):
        1. identify_food - User specifically asks to identify the food (only if not already done or if new image)
        2. get_nutrition - User wants calorie/nutrition information
        3. check_diet - User wants dietary compatibility (vegan, keto, etc.)
        4. get_history - User wants cultural/historical information about the dish
        5. general_knowledge - General food questions not requiring image analysis
        
        IMPORTANT: If the food is already identified, DO NOT include "identify_food" unless the user specifically asks for re-identification.
        
        Return a JSON with:
        - "actions": list of required actions
        - "needs_identification_first": boolean if food needs to be identified before other actions
        - "is_greeting": boolean if this is just a greeting/hello
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a query analyzer. Return only valid JSON."),
                HumanMessage(content=analysis_prompt)
            ])
            
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_analysis(user_text, has_current_image, current_analysis_exists)
                
        except Exception:
            return self._fallback_analysis(user_text, has_current_image, current_analysis_exists)
    
    def _fallback_analysis(self, user_text: str, has_current_image: bool, current_analysis_exists: bool) -> Dict[str, Any]:
        """Fallback analysis using simple keyword matching"""
        text_lower = user_text.lower()
        actions = []
        
        is_greeting = any(word in text_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"])
        
        if is_greeting:
            return {
                "actions": ["general_knowledge"],
                "needs_identification_first": False,
                "is_greeting": True
            }
        
        identify_keywords = ["what is", "identify", "what dish", "what food", "what's this", "tell me what"]
        if any(keyword in text_lower for keyword in identify_keywords) and (not current_analysis_exists or "again" in text_lower):
            actions.append("identify_food")
        
        if any(word in text_lower for word in ["calories", "nutrition", "protein", "carbs", "fat", "nutrients", "kcal"]):
            actions.append("get_nutrition")
        
        if any(word in text_lower for word in ["vegan", "vegetarian", "gluten", "keto", "diabetic", "allergen", "diet"]):
            actions.append("check_diet")
        
        if any(word in text_lower for word in ["history", "origin", "culture", "traditional", "where from", "invented", "background", "story"]):
            actions.append("get_history")
        
        if not actions:
            actions.append("general_knowledge")
        
        needs_identification = False
        if has_current_image and not current_analysis_exists and any(action in ["get_nutrition", "check_diet", "get_history"] for action in actions):
            needs_identification = True
        
        return {
            "actions": actions,
            "needs_identification_first": needs_identification,
            "is_greeting": False
        }

class FoodChatbotAgent:
    """Main orchestrating agent with improved context management"""
    
    def __init__(self):
        self.llm = llm
        self.image_analyzer = ImageAnalyzerAgent(self.llm)
        self.calorie_estimator = CalorieEstimatorAgent(self.llm)
        self.diet_checker = DietCompatibilityAgent(self.llm)
        self.food_historian = FoodHistoryAgent(self.llm)
        self.query_analyzer = QueryAnalyzer(self.llm)
        
    def process_message(self, state: ChatState) -> ChatState:
        """Process user message and coordinate appropriate agents"""
        last_message = state["messages"][-1]
        
        user_text, image_base64 = self._extract_message_content(last_message)
        
        if image_base64:
            image_hash = get_image_hash(image_base64)
            state["current_image"] = image_base64
            state["current_image_hash"] = image_hash
            
            if image_hash not in state.get("image_contexts", {}):
                if "image_contexts" not in state:
                    state["image_contexts"] = {}
        
        if image_base64 and not user_text.strip():
            response = (
                "📸 New image uploaded successfully!\n\n"
                "What would you like to know about this food? I can help you with:\n\n"
                "🔍 Identify the dish - \"What is this food?\"\n"
                "📊 Nutrition info - \"How many calories does this have?\"\n"
                "🥗 Diet compatibility - \"Is this vegan?\" or \"Can diabetics eat this?\"\n"
                "📚 Food history - \"Tell me about the origin of this dish\"\n"
                "💡 Or combine requests - \"Identify this food and tell me its calories\"\n\n"
                "What interests you most?"
            )
            
            state["messages"].append(AIMessage(content=response))
            return state
        
        current_context = None
        if state.get("current_image_hash"):
            current_context = state.get("image_contexts", {}).get(state["current_image_hash"])
        
        has_current_image = state.get("current_image") is not None
        current_analysis_exists = current_context is not None and hasattr(current_context, 'analysis')
        
        query_analysis = self.query_analyzer.analyze_query(user_text, has_current_image, current_analysis_exists)
        
        response = self._execute_actions(state, query_analysis, user_text)
        
        if isinstance(response, str):
            state["messages"].append(AIMessage(content=response))
        else:
            state["messages"].append(AIMessage(content="I apologize, but I encountered an issue processing your request. Please try again."))
        
        return state
    
    def _extract_message_content(self, message):
        """Extract text and image from message"""
        user_text = ""
        image_base64 = None
        
        if isinstance(message, HumanMessage):
            if isinstance(message.content, str):
                user_text = message.content
            elif isinstance(message.content, list):
                for content in message.content:
                    if isinstance(content, dict):
                        if content.get("type") == "text":
                            user_text = content["text"]
                        elif content.get("type") == "image_url":
                            image_url = content["image_url"]["url"]
                            if "base64," in image_url:
                                image_base64 = image_url.split("base64,")[1]
        
        return user_text, image_base64
    
    def _get_or_create_analysis(self, state: ChatState) -> Optional[FoodAnalysis]:
        """Get existing analysis or create new one for current image"""
        if not state.get("current_image_hash"):
            return None
        
        image_hash = state["current_image_hash"]
        
        if image_hash in state.get("image_contexts", {}):
            return state["image_contexts"][image_hash].analysis
        
        if state.get("current_image"):
            analysis = self.image_analyzer.analyze_food_image(state["current_image"])
            
            if "image_contexts" not in state:
                state["image_contexts"] = {}
            
            state["image_contexts"][image_hash] = ImageContext(
                image_hash=image_hash,
                analysis=analysis,
                nutrition=None,
                diet_info=None,
                upload_timestamp=format_timestamp()
            )
            
            return analysis
        
        return None
    
    def _execute_actions(self, state: ChatState, query_analysis: Dict, user_text: str) -> str:
        """Execute the determined actions and compile response"""
        actions = query_analysis.get("actions", [])
        needs_identification = query_analysis.get("needs_identification_first", False)
        is_greeting = query_analysis.get("is_greeting", False)
        
        response_parts = []
        
        if is_greeting:
            return ("👋 Welcome to Food AI Chatbot!\n\n"
                    "I can help you with:\n"
                    "🔍 Food Identification - Upload an image and I'll tell you what dish it is\n"
                    "📊 Nutrition Analysis - Get calorie and nutrient information\n"
                    "🥗 Dietary Compatibility - Check if food fits various diets\n"
                    "📚 Food History - Learn about the cultural background of dishes\n"
                    "❓ Food Questions - Ask me anything about food and nutrition\n\n"
                    "Just upload a food image and tell me what you'd like to know!")
        
        if needs_identification or "identify_food" in actions:
            analysis = self._get_or_create_analysis(state)
            if not analysis and "identify_food" in actions:
                return "❌ I need an image to identify the food. Please upload a food image first!"
        
        for action in actions:
            try:
                if action == "identify_food":
                    result = self._handle_food_identification(state)
                    if result:
                        response_parts.append(str(result))
                
                elif action == "get_nutrition":
                    result = self._handle_nutrition_request(state)
                    if result:
                        response_parts.append(str(result))
                
                elif action == "check_diet":
                    result = self._handle_diet_check(state)
                    if result:
                        response_parts.append(str(result))
                
                elif action == "get_history":
                    result = self._handle_history_request(state, user_text)
                    if result:
                        response_parts.append(str(result))
                
                elif action == "general_knowledge":
                    result = self._handle_general_query(user_text)
                    if result:
                        response_parts.append(str(result))
            
            except Exception as e:
                response_parts.append(f"❌ Error processing {action}: {str(e)}")
        
        if len(response_parts) > 1:
            final_response = "\n\n---\n\n".join(str(part) for part in response_parts if part)
        elif response_parts:
            final_response = str(response_parts[0])
        else:
            final_response = "I'm not sure how to help with that. Could you try rephrasing your question?"
        
        return final_response
    
    def _handle_food_identification(self, state: ChatState) -> str:
        """Handle food identification requests"""
        if not state.get("current_image"):
            return "❌ I need an image to identify the food. Please upload a food image first!"
        
        analysis = self._get_or_create_analysis(state)
        if not analysis:
            return "❌ Could not analyze the image. Please try uploading the image again."
        
        return (f"🍽️ **Food Identification**\n\n"
                f"📛 **Dish:** {analysis.dish_name}\n"
                f"📝 **Description:** {analysis.description}\n"
                f"🥘 **Main Ingredients:** {', '.join(analysis.main_ingredients)}\n"
                f"🌍 **Cuisine Type:** {analysis.cuisine_type or 'Not specified'}\n"
                f"📈 **Confidence:** {analysis.confidence_score:.1%}")
    
    def _handle_nutrition_request(self, state: ChatState) -> str:
        """Handle nutrition information requests"""
        analysis = self._get_or_create_analysis(state)
        if not analysis:
            return "❌ I need to identify the food first. Please upload an image or ask me to identify the dish."
        
        image_hash = state["current_image_hash"]
        context = state["image_contexts"][image_hash]
        
        if not context.nutrition:
            nutrition = self.calorie_estimator.estimate_nutrition(analysis)
            context.nutrition = nutrition
        else:
            nutrition = context.nutrition
        
        response = (f"📊 **Nutrition Information for {analysis.dish_name}**\n\n"
                    f"🍽️ **Serving Size:** {nutrition.serving_size}\n"
                    f"🔥 **Calories:** {nutrition.calories_per_serving} kcal\n"
                    f"💪 **Protein:** {nutrition.protein_g}g\n"
                    f"🍞 **Carbohydrates:** {nutrition.carbs_g}g\n"
                    f"🧈 **Fat:** {nutrition.fat_g}g")
        
        if nutrition.fiber_g:
            response += f"\n🌾 **Fiber:** {nutrition.fiber_g}g"
        if nutrition.sugar_g:
            response += f"\n🍯 **Sugar:** {nutrition.sugar_g}g"
        
        return response
    
    def _handle_diet_check(self, state: ChatState) -> str:
        """Handle dietary compatibility checks"""
        analysis = self._get_or_create_analysis(state)
        if not analysis:
            return "❌ I need to identify the food first. Please upload an image or ask me to identify the dish."
        
        image_hash = state["current_image_hash"]
        context = state["image_contexts"][image_hash]
        
        if not context.diet_info:
            diet_info = self.diet_checker.check_diet_compatibility(analysis)
            context.diet_info = diet_info
        else:
            diet_info = context.diet_info
        
        return (f"🥗 **Dietary Compatibility for {analysis.dish_name}**\n\n"
                f"🌱 **Vegan:** {'✅ Yes' if diet_info.is_vegan else '❌ No'}\n"
                f"🥕 **Vegetarian:** {'✅ Yes' if diet_info.is_vegetarian else '❌ No'}\n"
                f"🌾 **Gluten-Free:** {'✅ Yes' if diet_info.is_gluten_free else '❌ No'}\n"
                f"🥩 **Keto-Friendly:** {'✅ Yes' if diet_info.is_keto_friendly else '❌ No'}\n"
                f"🩺 **Diabetic-Friendly:** {'✅ Yes' if diet_info.is_diabetic_friendly else '❌ No'}\n\n"
                f"⚠️ **Potential Allergens:** {', '.join(diet_info.allergens) if diet_info.allergens else 'None identified'}\n\n"
                f"📝 **Notes:** {diet_info.dietary_notes}")
    
    def _handle_history_request(self, state: ChatState, user_text: str) -> str:
        """Handle requests for food history/cultural information with web search"""
        analysis = self._get_or_create_analysis(state)
        
        if analysis:
            dish_name = analysis.dish_name
            cuisine_type = analysis.cuisine_type
        else:
            dish_name = self._extract_dish_name_from_query(user_text)
            cuisine_type = None
        
        if not dish_name or dish_name.lower() in ["unknown", "this dish"]:
            return "❌ I need to know what dish you're asking about. Please upload an image first or specify the dish name in your question."
        
        # Get detailed history using web search
        history_info = self.food_historian.get_food_history(dish_name, cuisine_type)
        
        return f"📚 **History & Culture of {dish_name}**\n\n{history_info}"
    
    def _extract_dish_name_from_query(self, user_text: str) -> str:
        """Extract dish name from user query for history requests"""
        text_lower = user_text.lower()
        
        if " of " in text_lower:
            parts = text_lower.split(" of ")
            if len(parts) > 1:
                return parts[1].split()[0].title()
        
        if " about " in text_lower:
            parts = text_lower.split(" about ")
            if len(parts) > 1:
                return parts[1].split()[0].title()
        
        return "this dish"
    
    def _handle_general_query(self, user_text: str) -> str:
        """Handle general food knowledge queries"""
        try:
            messages = [
                SystemMessage(content="You are a food knowledge expert. Answer the query using your extensive knowledge about food, nutrition, and cooking. Be helpful and informative."),
                HumanMessage(content=f"Query: {user_text}")
            ]
            
            response = self.llm.invoke(messages)
            return f"🤖 **Food Knowledge**\n\n{response.content}"
        except Exception as e:
            return f"I'm sorry, I couldn't process your question at the moment. Could you try rephrasing it?"

# ===== LangGraph WorkFlow =====
def create_food_chatbot_graph():
    """Create the LangGraph workflow for the food chatbot"""
    
    chatbot_agent = FoodChatbotAgent()
    
    graph_builder = StateGraph(ChatState)
    
    def chatbot_node(state: ChatState) -> ChatState:
        """Main chatbot processing node"""
        return chatbot_agent.process_message(state)
    
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph


