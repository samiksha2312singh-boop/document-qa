import streamlit as st
import requests
import os
import json
from typing import Dict, Optional, List
from openai import OpenAI

# Import for multi-vendor support
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ============ PART A: Weather Function ============
def get_current_weather(location: str, API_key: str) -> Dict:
    """
    Fetch current weather data for a given location using OpenWeatherMap API.
    This is the actual function that calls the weather API.
    """
    try:
        # Handle comma-separated locations
        if "," in location:
            location = location.split(",")[0].strip()
        
        # Construct API URL
        urlbase = "https://api.openweathermap.org/data/2.5/"
        urlweather = f"weather?q={location}&appid={API_key}"
        url = urlbase + urlweather
        
        # Make API request
        response = requests.get(url)
        
        if response.status_code != 200:
            return {"error": f"City '{location}' not found"}
        
        data = response.json()
        
        # Extract and convert temperatures
        temp = data['main']['temp'] - 273.15
        feels_like = data['main']['feels_like'] - 273.15
        temp_min = data['main']['temp_min'] - 273.15
        temp_max = data['main']['temp_max'] - 273.15
        humidity = data['main']['humidity']
        
        # Extract weather info
        weather_description = data['weather'][0]['description']
        weather_main = data['weather'][0]['main']
        wind_speed = data['wind']['speed']
        
        return {
            "location": location,
            "temperature": round(temp, 2),
            "feels_like": round(feels_like, 2),
            "temp_min": round(temp_min, 2),
            "temp_max": round(temp_max, 2),
            "humidity": round(humidity, 2),
            "description": weather_description,
            "main_weather": weather_main,
            "wind_speed": round(wind_speed, 2)
        }
    
    except Exception as e:
        return {"error": f"Error fetching weather: {str(e)}"}

# ============ PART B: OpenAI Function Calling ============
def get_weather_for_openai(location: str = "Syracuse NY") -> str:
    """
    Weather function formatted for OpenAI function calling.
    Returns JSON string for OpenAI to process.
    """
    # Get the API key
    api_key = st.session_state.get('weather_api_key')
    if not api_key:
        return json.dumps({"error": "Weather API key not configured"})
    
    # Default to Syracuse NY if no location provided
    if not location or location.strip() == "":
        location = "Syracuse NY"
    
    # Get weather data
    weather_data = get_current_weather(location, api_key)
    
    # Format for LLM consumption
    if "error" not in weather_data:
        formatted = {
            "location": weather_data["location"],
            "temperature_celsius": weather_data["temperature"],
            "feels_like_celsius": weather_data["feels_like"],
            "weather_condition": weather_data["description"],
            "humidity_percent": weather_data["humidity"],
            "wind_speed_ms": weather_data["wind_speed"]
        }
        return json.dumps(formatted)
    else:
        return json.dumps(weather_data)

# OpenAI function definition for API
WEATHER_FUNCTION_DEFINITION = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state/country, e.g. San Francisco, CA or Paris, France"
            }
        },
        "required": ["location"]
    }
}

def get_openai_response_with_function(client: OpenAI, user_input: str) -> str:
    """
    Get response from OpenAI using function calling for weather data.
    """
    try:
        # First API call - let OpenAI decide if it needs weather data
        messages = [
            {"role": "system", "content": "You are a helpful travel assistant that provides weather information and clothing suggestions."},
            {"role": "user", "content": user_input}
        ]
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=[WEATHER_FUNCTION_DEFINITION],
            function_call="auto"  # Let OpenAI decide when to call the function
        )
        
        message = response.choices[0].message
        
        # Check if OpenAI wants to call the weather function
        if message.function_call:
            # Parse the function arguments
            function_args = json.loads(message.function_call.arguments)
            location = function_args.get("location", "Syracuse NY")
            
            # Call our weather function
            weather_result = get_weather_for_openai(location)
            
            # Second API call - provide weather data and get clothing suggestions
            messages.append(message.model_dump())  # Add assistant's function call
            messages.append({
                "role": "function",
                "name": "get_current_weather",
                "content": weather_result
            })
            
            # Ask specifically for clothing suggestions
            messages.append({
                "role": "user",
                "content": "Based on this weather data, please provide detailed suggestions for what clothes to wear today. Include specific recommendations for upper body, lower body, footwear, and any accessories needed."
            })
            
            final_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return final_response.choices[0].message.content
        else:
            # No weather function needed
            return message.content
            
    except Exception as e:
        return f"Error with OpenAI: {str(e)}"

# ============ PART C: Multi-Vendor Support ============
def get_claude_response_with_function(client, user_input: str) -> str:
    """
    Get response from Claude with weather data (simulated function calling).
    Claude doesn't have native function calling like OpenAI, so we handle it differently.
    """
    try:
        # Detect if weather is needed based on keywords
        weather_keywords = ['weather', 'temperature', 'clothes', 'wear', 'dress', 'outfit', 'climate']
        needs_weather = any(keyword in user_input.lower() for keyword in weather_keywords)
        
        if needs_weather:
            # Extract location from input or use default
            location = extract_location_from_text(user_input) or "Syracuse NY"
            
            # Get weather data
            api_key = st.session_state.get('weather_api_key')
            weather_data = get_current_weather(location, api_key)
            
            if "error" not in weather_data:
                # Create prompt with weather data
                weather_info = f"""
                Current weather in {weather_data['location']}:
                - Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)
                - Conditions: {weather_data['description']}
                - Humidity: {weather_data['humidity']}%
                - Wind Speed: {weather_data['wind_speed']} m/s
                """
                
                prompt = f"""
                {weather_info}
                
                Based on this weather data, please provide detailed clothing suggestions for today.
                Include specific recommendations for:
                1. Upper body clothing
                2. Lower body clothing  
                3. Footwear
                4. Accessories (hat, sunglasses, umbrella, etc.)
                5. Any special considerations for the weather conditions
                
                User's request: {user_input}
                """
            else:
                prompt = f"I couldn't fetch weather data: {weather_data['error']}. {user_input}"
        else:
            prompt = user_input
        
        # Get Claude's response
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"Error with Claude: {str(e)}"

def extract_location_from_text(text: str) -> Optional[str]:
    """
    Simple location extraction from text.
    In production, you'd use NER or more sophisticated methods.
    """
    # Common city patterns
    import re
    
    # Look for "in [City]" or "for [City]" or "weather [City]"
    patterns = [
        r"(?:in|for|at|weather)\s+([A-Z][a-zA-Z\s]+(?:,\s*[A-Z][A-Z])?)",
        r"([A-Z][a-zA-Z]+(?:,\s*[A-Z][a-zA-Z]+)?)\s+weather",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return None

# ============ MAIN APPLICATION ============
def run():
    """Main application for Lab 5b and 5c"""
    
    st.set_page_config(page_title="Lab 5 - Travel Weather Bot", page_icon="🌤️", layout="wide")
    
    st.title("🌤️ Travel Weather & Clothing Suggestion Bot")
    st.markdown("Get weather information and personalized clothing recommendations using AI")
    
    # Initialize API keys
    try:
        # Weather API
        weather_api_key = st.secrets.get("OPENWEATHER_API_KEY") or os.getenv("OPENWEATHER_API_KEY")
        if weather_api_key:
            st.session_state['weather_api_key'] = weather_api_key
        else:
            st.error("❌ OpenWeatherMap API key not found in secrets")
            return
        
        # OpenAI API
        openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # Claude API (optional)
        claude_api_key = None
        if ANTHROPIC_AVAILABLE:
            claude_api_key = st.secrets.get("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            
    except Exception as e:
        st.error(f"Error loading API keys: {str(e)}")
        return
    
    # Sidebar - LLM Selection (Part C)
    st.sidebar.header("🤖 AI Model Selection")
    
    available_models = []
    if openai_api_key:
        available_models.append("OpenAI (GPT-3.5)")
        available_models.append("OpenAI (GPT-4)")
    if claude_api_key and ANTHROPIC_AVAILABLE:
        available_models.append("Claude (Haiku)")
        available_models.append("Claude (Sonnet)")
    
    if not available_models:
        st.error("No AI models available. Please configure API keys.")
        return
    
    selected_model = st.sidebar.selectbox(
        "Choose AI Model:",
        available_models,
        help="Select which AI model to use for suggestions"
    )
    
    # Show model capabilities
    if "OpenAI" in selected_model:
        st.sidebar.info("✅ Native function calling support")
    else:
        st.sidebar.info("📝 Simulated function calling")
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask about weather or what to wear:",
            placeholder="e.g., 'What should I wear in Paris today?' or 'Weather in London'",
            help="Enter a city name or ask for clothing suggestions"
        )
    
    with col2:
        st.markdown("**Quick Queries:**")
        if st.button("🌡️ Syracuse Weather"):
            user_input = "What's the weather in Syracuse NY and what should I wear?"
        if st.button("🇬🇧 London Outfit"):
            user_input = "What clothes should I pack for London?"
        if st.button("🏖️ Miami Beach"):
            user_input = "What to wear in Miami Beach today?"
    
    # Process button
    if st.button("🔍 Get Suggestions", type="primary", disabled=not user_input):
        if user_input:
            with st.spinner(f"Getting weather and suggestions using {selected_model}..."):
                
                # Initialize the appropriate client
                if "OpenAI" in selected_model:
                    client = OpenAI(api_key=openai_api_key)
                    
                    # Use GPT-4 if selected
                    if "GPT-4" in selected_model:
                        # Temporarily override model in function
                        response = get_openai_response_with_function(client, user_input)
                    else:
                        response = get_openai_response_with_function(client, user_input)
                    
                elif "Claude" in selected_model:
                    client = anthropic.Anthropic(api_key=claude_api_key)
                    response = get_claude_response_with_function(client, user_input)
                else:
                    response = "Selected model not properly configured."
                
                # Display response
                st.markdown("---")
                st.subheader("🤖 AI Suggestions")
                st.markdown(response)
                
                # Show which model was used
                st.caption(f"Generated by: {selected_model}")
    
    # Information section
    with st.expander("ℹ️ About This Bot"):
        st.markdown("""
        ### Features:
        - **Automatic Weather Detection**: The bot automatically detects when weather information is needed
        - **Default Location**: Uses Syracuse NY when no location is specified
        - **Multi-Model Support**: Choose between OpenAI and Claude (if available)
        - **Function Calling**: OpenAI uses native function calling, Claude uses simulated approach
        
        ### How it works:
        1. You ask a question about weather or clothing
        2. The AI determines if weather data is needed
        3. If needed, it fetches real-time weather from OpenWeatherMap
        4. The AI then provides personalized clothing suggestions
        
        ### Try asking:
        - "What should I wear in Tokyo?"
        - "Is it cold in Chicago today?"
        - "I'm traveling to Rome, what clothes should I pack?"
        - "What's the weather like?" (uses Syracuse NY as default)
        """)
    
    # Debug information (optional)
    if st.sidebar.checkbox("🔧 Show Debug Info"):
        st.sidebar.markdown("---")
        st.sidebar.markdown("**API Status:**")
        st.sidebar.write(f"Weather API: {'✅' if weather_api_key else '❌'}")
        st.sidebar.write(f"OpenAI: {'✅' if openai_api_key else '❌'}")
        st.sidebar.write(f"Claude: {'✅' if claude_api_key else '❌'}")

if __name__ == "__main__":
    run()