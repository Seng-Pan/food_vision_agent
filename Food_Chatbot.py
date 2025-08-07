from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.prompts import PromptTemplate
from typing import Annotated, Optional, List, Dict, Any
from Agent import create_food_chatbot_graph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
import wikipedia
import requests
import base64
import json
import os
from PIL import Image
import hashlib
import io
import time
import re

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your_tavily_api_key_here")

def is_tavily_configured():
    """Check if Tavily API is properly configured"""
    return (TAVILY_API_KEY and 
            TAVILY_API_KEY != "your_tavily_api_key_here" and
            len(TAVILY_API_KEY) > 10)


def create_custom_css():
    """Create custom CSS for beautiful UI"""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        font-weight: 400;
        margin-bottom: 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        font-weight: 500;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
        font-weight: 500;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 1rem 1.5rem;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .typing-dots {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #667eea;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stats-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """

def show_typing_indicator(container):
    """Show typing indicator"""
    typing_html = """
    <div class="typing-indicator">
        <span style="margin-right: 10px;">🤖 AI is thinking</span>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    </div>
    """
    return container.markdown(typing_html, unsafe_allow_html=True)

def create_feature_cards():
    """Create beautiful feature cards for sidebar"""
    features = [
        {
            "icon": "🔍",
            "title": "Food Identification",
            "description": "AI-powered dish recognition from photos"
        },
        {
            "icon": "📊",
            "title": "Nutrition Analysis", 
            "description": "Detailed calorie and nutrient breakdown"
        },
        {
            "icon": "🥗",
            "title": "Diet Compatibility",
            "description": "Check vegan, keto, gluten-free options"
        },
        {
            "icon": "📚",
            "title": "Food History",
            "description": "Cultural background and origins with web search"
        }
    ]
    
    for feature in features:
        st.markdown(f"""
        <div class="feature-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{feature['icon']}</div>
            <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem; color: #374151;">{feature['title']}</div>
            <div style="color: #6b7280; font-size: 0.9rem;">{feature['description']}</div>
        </div>
        """, unsafe_allow_html=True)

def display_stats(chat_state):
    """Display usage statistics"""
    if chat_state.get("image_contexts"):
        images_analyzed = len(chat_state["image_contexts"])
        total_messages = len([msg for msg in chat_state["messages"] if isinstance(msg, (HumanMessage, AIMessage))])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{images_analyzed}</div>
                <div class="stats-label">Images Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{total_messages}</div>
                <div class="stats-label">Total Messages</div>
            </div>
            """, unsafe_allow_html=True)

def display_message_with_animation(message, message_type="assistant"):
    """Display message with typing animation"""
    if message_type == "user":
        st.markdown(f"""
        <div class="user-message">
            {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        message_container = st.empty()
        
        show_typing_indicator(message_container)
        time.sleep(1)
        
        words = message.split(' ')
        displayed_text = ""
        
        for i, word in enumerate(words):
            displayed_text += word + " "
            message_container.markdown(f"""
            <div class="assistant-message">
                {displayed_text}
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.05)
        
        message_container.markdown(f"""
        <div class="assistant-message">
            {message}
        </div>
        """, unsafe_allow_html=True)

def create_streamlit_app():
    """Create the enhanced Streamlit web interface with session state fixes"""
    
    st.set_page_config(
        page_title="🍽️ Food AI Chatbot",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown(create_custom_css(), unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🍽️ Food AI Chatbot</div>
        <div class="main-subtitle">Your intelligent culinary companion powered by AI with web search</div>
    </div>
    """, unsafe_allow_html=True)
    
    
    # FIXED: Better session state initialization with reset handling
    def initialize_session_state():
        """Initialize session state variables properly"""
        
        # Handle conversation reset first
        if st.session_state.get("resetting_conversation", False):
            import uuid
            
            # Clear ALL conversation-related state
            for key in list(st.session_state.keys()):
                if key in ["chat_state", "config", "show_welcome", "last_upload_key", 
                          "conversation_initialized", "image_contexts"]:
                    del st.session_state[key]
            
            # Create fresh state
            st.session_state.chat_state = {
                "messages": [SystemMessage(content="You are a helpful food analysis assistant.")],
                "current_image": None,
                "current_image_hash": None,
                "image_contexts": {},
                "user_query_type": "general"
            }
            
            # Generate new thread ID for LangGraph
            thread_id = str(uuid.uuid4())
            st.session_state.config = {"configurable": {"thread_id": thread_id}}
            
            # Reset other state variables
            st.session_state.show_welcome = True
            st.session_state.last_upload_key = None
            st.session_state.conversation_initialized = True
            
            # Clear the reset flag
            st.session_state.resetting_conversation = False
            
            return  # Exit early to prevent old state from being processed
        
        # Normal initialization
        if "graph" not in st.session_state:
            st.session_state.graph = create_food_chatbot_graph()
        
        if "chat_state" not in st.session_state:
            st.session_state.chat_state = {
                "messages": [SystemMessage(content="You are a helpful food analysis assistant.")],
                "current_image": None,
                "current_image_hash": None,
                "image_contexts": {},
                "user_query_type": "general"
            }
        
        if "config" not in st.session_state:
            # Use a unique thread ID based on session
            import uuid
            thread_id = str(uuid.uuid4())
            st.session_state.config = {"configurable": {"thread_id": thread_id}}
        
        if "show_welcome" not in st.session_state:
            st.session_state.show_welcome = True
        
        if "last_upload_key" not in st.session_state:
            st.session_state.last_upload_key = None
        
        if "conversation_initialized" not in st.session_state:
            st.session_state.conversation_initialized = True
        
        if "resetting_conversation" not in st.session_state:
            st.session_state.resetting_conversation = False

    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #374151; margin-bottom: 2rem;">✨ Features</h2>
        </div>
        """, unsafe_allow_html=True)
        
        create_feature_cards()
        
        st.markdown("---")
        
        # Upload section
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h3 style="color: #374151;">📸 Upload Food Image</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a food image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of food for analysis",
            label_visibility="collapsed"
        )
        
        # Stats section
        if st.session_state.chat_state.get("image_contexts"):
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; margin: 2rem 0;">
                <h3 style="color: #374151;">📈 Your Stats</h3>
            </div>
            """, unsafe_allow_html=True)
            display_stats(st.session_state.chat_state)
        
        # Quick actions
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h3 style="color: #374151;">🚀 Quick Actions</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 New Conversation", use_container_width=True):
            # FIXED: Set reset flag FIRST to prevent flash
            st.session_state.resetting_conversation = True
            st.rerun()
        
        # Configuration section
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <h4 style="color: #374151;">⚙️ Configuration</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if TAVILY_API_KEY == "your_tavily_api_key_here":
            st.warning("🔑 Tavily API not configured. Using Wikipedia fallback for food history.")
            st.markdown("To enable enhanced search, get a free API key from [tavily.com](https://tavily.com)")
        else:
            st.success("✅ Tavily Search API configured")
    
    # Main chat area
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        # FIXED: Don't display anything if we're resetting
        if st.session_state.get("resetting_conversation", False):
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔄</div>
                <h2 style="color: #000; margin-bottom: 1rem;">Starting New Conversation...</h2>
            </div>
            """, unsafe_allow_html=True)
            return  # Don't render the rest of the chat
        
        # Welcome message - FIXED: Only show for truly new conversations
        if (st.session_state.show_welcome and 
            len(st.session_state.chat_state["messages"]) == 1 and
            not st.session_state.chat_state.get("image_contexts")):
            
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">👋</div>
                <h2 style="color: #000; margin-bottom: 1rem;">Welcome to Food AI!</h2>
                <p style="color: #fff; font-size: 1.1rem; margin-bottom: 2rem;">
                    I'm here to help you discover, analyze, and learn about food! 
                    Upload an image or ask me anything about nutrition, recipes, or food culture.
                </p>
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                           color: white; padding: 1rem; border-radius: 15px; margin: 1rem 0;">
                    💡 <strong>Tip:</strong> Try uploading a food photo and asking "What is this and how many calories does it have?"
                </div>
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 1rem; border-radius: 15px; margin: 1rem 0;">
                    🌍 <strong>New:</strong> Enhanced food history with web search for better accuracy on ethnic cuisines!
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat history - FIXED: Skip system message properly
        chat_messages = st.session_state.chat_state["messages"][1:]  # Skip system message
        
        for message in chat_messages:
            if isinstance(message, HumanMessage):
                if isinstance(message.content, list):
                    has_text = False
                    for content in message.content:
                        if isinstance(content, dict):
                            if content.get("type") == "text":
                                display_message_with_animation(content["text"], "user")
                                has_text = True
                            elif content.get("type") == "image_url":
                                if not has_text:
                                    display_message_with_animation("🖼️ *Image uploaded*", "user")
                else:
                    display_message_with_animation(message.content, "user")
            
            elif isinstance(message, AIMessage):
                display_message_with_animation(message.content, "assistant")
    
    # Handle image upload - FIXED: Better upload handling and reset check
    if uploaded_file is not None and not st.session_state.get("resetting_conversation", False):
        upload_key = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.type}"
        
        # Only process if this is a new upload
        if st.session_state.last_upload_key != upload_key:
            st.session_state.last_upload_key = upload_key
            st.session_state.show_welcome = False
            
            try:
                # Convert image to base64
                image = Image.open(uploaded_file)
                
                # Resize if too large (optional optimization)
                max_size = (1024, 1024)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=85)
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Create image message
                image_message = HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ])
                
                st.session_state.chat_state["messages"].append(image_message)
                
                # Display uploaded image
                with col2:
                    st.markdown("""
                    <div style="text-align: center; margin: 1rem 0;">
                        <div class="user-message">🖼️ Image uploaded successfully!</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.image(image, caption="Your uploaded food image", 
                            use_column_width=True, clamp=True)
                
                # Process with graph
                with col2:
                    typing_container = st.empty()
                    show_typing_indicator(typing_container)
                    
                    try:
                        result = st.session_state.graph.invoke(
                            st.session_state.chat_state,
                            st.session_state.config
                        )
                        st.session_state.chat_state = result
                        
                        typing_container.empty()
                        
                        response = st.session_state.chat_state["messages"][-1].content
                        display_message_with_animation(response, "assistant")
                        
                    except Exception as e:
                        typing_container.empty()
                        st.error(f"Error processing image: {str(e)}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing uploaded image: {str(e)}")
    
    # Chat input at the bottom - FIXED: Disable during reset
    if not st.session_state.get("resetting_conversation", False):
        with col2:
            st.markdown("""
            <div style="margin-top: 2rem; padding: 1rem; 
                        background: rgba(255, 255, 255, 0.9); 
                        border-radius: 20px; 
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
            """, unsafe_allow_html=True)
            
            prompt = st.chat_input("Ask me about food, nutrition, or dietary restrictions...", key="main_input")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Handle text input - FIXED: Better error handling and reset check
    if prompt and not st.session_state.get("resetting_conversation", False):
        st.session_state.show_welcome = False
        
        user_message = HumanMessage(content=prompt)
        st.session_state.chat_state["messages"].append(user_message)
        
        with col2:
            display_message_with_animation(prompt, "user")
            
            typing_container = st.empty()
            show_typing_indicator(typing_container)
            
            try:
                result = st.session_state.graph.invoke(
                    st.session_state.chat_state,
                    st.session_state.config
                )
                st.session_state.chat_state = result
                
                typing_container.empty()
                
                response = st.session_state.chat_state["messages"][-1].content
                display_message_with_animation(response, "assistant")
                
            except Exception as e:
                typing_container.empty()
                st.error(f"Error processing message: {str(e)}")
                
                # Add error response to maintain conversation flow
                error_message = AIMessage(content="I apologize, but I encountered an issue processing your request. Please try again or rephrase your question.")
                st.session_state.chat_state["messages"].append(error_message)
        
        st.rerun()

if __name__ == "__main__":
    create_streamlit_app()
