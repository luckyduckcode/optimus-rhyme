# Web Search Integration Guide

## 🌐 Web-Enabled AI Features

Your Telegram bot now has web search capabilities using DuckDuckGo!

### ✅ What's Been Added:

1. **Web Search** - Search the internet using DuckDuckGo
2. **Webpage Reading** - Fetch and read content from any URL
3. **Real-time Information** - Access current information beyond training data

---

## 📝 How to Use Web Search

### Method 1: Import in Bot Code

The `web_search.py` module is now available in `/app/automation/`. To use it in the bot:

```python
from web_search import search_web, fetch_webpage, web_search_command, web_read_command

# Search the web
results = search_web("latest AI news", max_results=5)

# Read a webpage
content = fetch_webpage("https://example.com")

# Get formatted search results
formatted = web_search_command("python tutorials")

# Read and format webpage
page_content = web_read_command("https://news.ycombinator.com")
```

### Method 2: Add Custom Commands

To add `/search` and `/read` commands to your bot, you would modify the `bot.py` file to include:

```python
from web_search import web_search_command, web_read_command

async def search_command(update, context):
    query = ' '.join(context.args)
    result = web_search_command(query)
    await update.message.reply_text(result, parse_mode='Markdown')

async def read_command(update, context):
    if not context.args:
        await update.message.reply_text("Please provide a URL")
        return
    url = context.args[0]
    result = web_read_command(url)
    await update.message.reply_text(result, parse_mode='Markdown')

# Then add handlers
application.add_handler(CommandHandler("search", search_command))
application.add_handler(CommandHandler("read", read_command))
```

---

## 🤖 Making AI Use Web Search

### Option A: Prompt Engineering

When chatting with the AI, you can instruct it to search:

```
User: "Search the web for latest developments in quantum computing and summarize"
Bot: [AI will know it should use search_web() function]
```

### Option B: Automatic Web Access

To give the AI automatic web access, you would need to:

1. Modify the bot to detect when web search is needed
2. Call the search function automatically
3. Feed results back to the AI for processing

Example integration:

```python
async def handle_message(update, context):
    user_message = update.message.text
    
    # Check if user wants web search
    if any(keyword in user_message.lower() for keyword in ['search', 'latest', 'current', 'recent']):
        # Extract search query
        query = user_message  # or parse it better
        
        # Search the web
        search_results = search_web(query, max_results=3)
        
        # Format results for AI
        context_info = "\n".join([
            f"{r['title']}: {r['snippet']}" 
            for r in search_results
        ])
        
        # Send to AI with context
        enhanced_prompt = f"""
        User asked: {user_message}
        
        Here is current information from the web:
        {context_info}
        
        Please answer based on this information.
        """
        
        # Get AI response with web context
        ai_response = get_ollama_response(enhanced_prompt)
```

---

## 🔍 Available Functions:

### `search_web(query, max_results=5)`
Searches DuckDuckGo and returns structured results.

**Returns:**
```python
[
    {
        'title': 'Result Title',
        'link': 'https://...',
        'snippet': 'Preview text...'
    },
    ...
]
```

### `fetch_webpage(url, max_length=2000)`
Fetches and extracts text from a webpage.

**Returns:** String of webpage text content

### `web_search_command(query)`
Pre-formatted search results ready for Telegram.

**Returns:** Markdown-formatted search results

### `web_read_command(url)`
Pre-formatted webpage content ready for Telegram.

**Returns:** Markdown-formatted page content

---

## ⚙️ Configuration

No API keys needed! DuckDuckGo search is free and doesn't require authentication.

### Customize Settings:

Edit `web_search.py` to adjust:
- `max_results`: Number of search results (default: 5)
- `max_length`: Maximum characters from webpage (default: 2000)
- `timeout`: Request timeout (default: 10 seconds)

---

## 🎯 Example Use Cases:

1. **Current Events**: "What's happening with SpaceX today?"
2. **Latest Info**: "Search for Python 3.13 new features"
3. **Fact Checking**: "Get latest COVID-19 statistics"
4. **Research**: "Find tutorials on Docker networking"
5. **News**: "What are today's top tech headlines?"

---

## 🔐 Privacy & Rate Limiting:

- **DuckDuckGo** respects privacy (no tracking)
- **Rate limits**: Built-in delays to avoid blocking
- **Safe**: No API keys or personal data required

---

## 🚀 Next Steps:

To fully integrate web search into your bot's responses, you would need to modify the bot.py file to either:

1. Add explicit `/search` and `/read` commands
2. Automatically detect when to search based on user questions
3. Integrate with the AI's function calling capabilities (if supported by model)

Would you like me to help implement any of these integrations?

---

**Status:** ✅ Web search module installed and ready to use!
**Location:** `/app/automation/web_search.py` in telegram-bot container
