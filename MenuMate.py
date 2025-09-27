from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_perplexity import ChatPerplexity
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import re
import difflib
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv
import psycopg2

load_dotenv()

# Initialize Rich 
#let me have this reverted
console = Console()

# Allowed topics for bot
allowed_topics = [ "food", "restaurant", "cuisine", "dining", "recipe","cafe", "coffee", "bistro", "eatery", "snack", "meal"]

# Function to check if user input matches allowed topics
def is_allowed(user_input):
    words = user_input.lower().split()
    for word in words:
        matches = difflib.get_close_matches(word, allowed_topics, cutoff=0.6)  # lower cutoff
        if matches:
            return True
    return False

# Formatting functions
def format_agent_output(text: str) -> None:
    """
    Format agent output using Rich library for beautiful terminal display.
    Handles restaurant listings and general text with proper styling.
    """
    if not text:
        return
    # Check if this looks like a restaurant listing format
    restaurant_pattern = r'\*\*(\d+\.)\s*([^\*]+)\*\*'
    if re.search(restaurant_pattern, text):
        # This looks like a restaurant listing - format it specially
        format_restaurant_listing(text)
    else:
        # General text formatting
        format_general_text(text)

def format_restaurant_listing(text: str) -> None:
    """
    Format restaurant listing with special styling.
    """
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Restaurant title (e.g., **4. Konark Vegetarian Restaurant**)
        restaurant_match = re.match(r'\*\*(\d+\.)\s*([^\*]+)\*\*', line)
        if restaurant_match:
            number = restaurant_match.group(1)
            name = restaurant_match.group(2).strip()
            # Create a beautiful header for the restaurant
            title = Text(f"{number} {name}", style="bold bright_blue")
            console.print(Panel(title, border_style="bright_blue", padding=(0, 1)))
            continue

        # Address line (starts with - **Address:**)
        address_match = re.match(r'-\s*\*\*Address:\*\*\s*(.*)', line)
        if address_match:
            address = address_match.group(1).strip()
            console.print(f"📍 [bold green]Address:[/bold green] [white]{address}[/white]")
            continue

        # Popular Menu Items (starts with - **Popular Menu Items:**)
        menu_match = re.match(r'-\s*\*\*Popular Menu Items:\*\*\s*(.*)', line)
        if menu_match:
            menu_items = menu_match.group(1).strip()
            console.print(f"🍽️  [bold yellow]Popular Items:[/bold yellow] [bright_white]{menu_items}[/bright_white]")
            continue

        # Description (starts with - **Description:**)
        desc_match = re.match(r'-\s*\*\*Description:\*\*\s*(.*)', line)
        if desc_match:
            description = desc_match.group(1).strip()
            console.print(f"📝 [bold cyan]Description:[/bold cyan] [white]{description}[/white]")
            console.print()  # Add spacing after each restaurant
            continue

        # Handle other lines that might not match the pattern
        if line.startswith('-'):
            # Generic bullet point
            content = line[1:].strip()
            console.print(f"  • [white]{content}[/white]")
        else:
            # Regular text
            console.print(f"[white]{line}[/white]")

def format_general_text(text: str) -> None:
    """
    Format general text with basic markdown support.
    """
    try:
        # Try to render as markdown first
        md = Markdown(text)
        console.print(md)
    except Exception:
        # Fallback to plain text with some basic formatting
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                console.print()
                continue
            # Handle bold text **text**
            line = re.sub(r'\*\*([^\*]+)\*\*', r'[bold]\1[/bold]', line)
            # Handle bullet points
            if line.startswith('-') or line.startswith('•'):
                line = f"  • [white]{line[1:].strip()}[/white]"
            else:
                line = f"[white]{line}[/white]"
            console.print(line)

# Initialize the LLM with Perplexity Pro
llm = ChatPerplexity(model="sonar-pro", temperature=0.7, timeout=60, api_key=os.getenv("PPLX_API_KEY"))

# Create a prompt template for our agent
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant powered by Perplexity Pro that can find real-time information on the web.
    When asked about restaurants/food only, search for relevant information and provide:
    1. The name and address of the restaurant
    2. Top 5 popular menu items if available
    3. Brief description of the restaurant
    
    For other topics, use your web search capabilities to provide accurate,structured,and up-to-date information.
    Always be concise but informative and remember previous conversations."""),
    MessagesPlaceholder("chat_history", optional=True),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Create the agent
agent = create_openai_functions_agent(llm, [], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)

# PostgreSQL connection 
DB_URL = os.getenv("DB_URL")
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

# Ensure table exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS conversation_history (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    role TEXT,
    message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# Function to save message to DB
def save_message(session_id, role, message):
    cursor.execute(
        "INSERT INTO conversation_history (session_id, role, message) VALUES (%s, %s, %s)",
        (session_id, role, message)
    )
    conn.commit()

# Function to load history from DB into ChatMessageHistory object
def repair_messages(rows):
    """
    Ensure user/assistant alternate for the LLM, without deleting anything from DB.
    """
    fixed = []
    last_role = None
    for role, message in rows:
        if role not in ("user", "assistant"):
            continue

        # If same role repeats, insert a dummy opposite
        if role == last_role:
            dummy = "assistant" if role == "user" else "user"
            fixed.append((dummy, "[no message recorded]"))

        fixed.append((role, message))
        last_role = role

    return fixed


def load_history_from_db(session_id: str) -> ChatMessageHistory:
    history = ChatMessageHistory()
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT role, message FROM conversation_history
        WHERE session_id = %s ORDER BY timestamp ASC
    """, (session_id,))
    rows = cur.fetchall()
    conn.close()

    # Auto-repair for LLM consumption
    fixed_rows = repair_messages(rows)

    for role, message in fixed_rows:
        if role == "user":
            history.add_user_message(message)
        elif role == "assistant":
            history.add_ai_message(message)

    return history


# In-memory store for fast access during session
store = {}

def get_history(cfg_or_sid):
    if isinstance(cfg_or_sid, str):
        sid = cfg_or_sid
    else:
        sid = cfg_or_sid.get("configurable", {}).get("session_id") or cfg_or_sid.get("session_id") or "default"

    if sid not in store:
        history = load_history_from_db(sid)

        # Repair alternation once when loading
        msgs = history.messages
        if msgs and msgs[-1].type == "human":
            history.add_ai_message("[no reply recorded]")

        store[sid] = history

    return store[sid]


# Wrap agent with memory capability
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

if __name__ == "__main__":
    session_id = os.environ.get("SESSION_ID", "mack")

    console.print("\n[bold green]🤖 Memory test shell with PERSISTENT storage.[/bold green]")
    console.print("[dim]Type 'exit' to quit. Commands: /history, /clear, /sessions[/dim]\n")
    
    while True:
        try:
            # Create a styled prompt
            console.print("[bold blue]you>[/bold blue] ", end="")
            user_prompt = input().strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]bye 👋[/yellow]")
            break

        if not user_prompt:
            continue
        
        # Exit commands
        if user_prompt.lower() in {"exit", "quit", ":q", "/exit"}:
            console.print("[yellow]bye 👋[/yellow]")
            break
        
        # Show last 10 history messages
        if user_prompt == "/history":
            cursor.execute(
                "SELECT role, message, timestamp FROM conversation_history WHERE session_id=%s ORDER BY timestamp",
                (session_id,)
            )
            rows = cursor.fetchall()
            if not rows:
                console.print("[dim](no history yet)[/dim]\n")
                continue
            console.print(f"[bold cyan]=== History for session '{session_id}' ===[/bold cyan]")
            for i, (role, message, ts) in enumerate(rows[-10:]):  # last 10
                role_color = "blue" if role=="user" else "green"
                console.print(f"[dim]\\[{i+1}][/dim] [bold {role_color}][{role}][/bold {role_color}] {message}")
            console.print()
            continue
        
        # Clear memory for this session
        if user_prompt == "/clear":
            cursor.execute("DELETE FROM conversation_history WHERE session_id=%s", (session_id,))
            conn.commit()
            store[session_id] = ChatMessageHistory()
            console.print(f"[yellow]Cleared memory for session '{session_id}'[/yellow]\n")
            continue
        
        # List all sessions
        if user_prompt == "/sessions":
            cursor.execute("SELECT DISTINCT session_id FROM conversation_history")
            sessions = cursor.fetchall()
            if not sessions:
                console.print("[dim]No saved sessions found[/dim]\n")
                continue
            console.print("[bold cyan]=== Saved Sessions ===[/bold cyan]")
            for f in sessions:
                session_name = f[0]
                indicator = " [green](current)[/green]" if session_name == session_id else ""
                console.print(f"  • [white]{session_name}[/white]{indicator}")
            console.print()
            continue

        # Save user input to DB no matter what
        save_message(session_id, "user", user_prompt)

        # Topic restriction
        if not is_allowed(user_prompt):
            save_message(session_id, "assistant", "[blocked topic: not food/restaurants]")
            console.print("[yellow]Sorry, I only talk about food and restaurants.[/yellow]\n")
            continue


        try:
            # Show thinking indicator
            with console.status("[bold green]Agent is thinking...", spinner="dots"):
                result = agent_with_history.invoke(
                    {"input": user_prompt},
                    config={"configurable": {"session_id": session_id}},
                )
            # AgentExecutor returns a dict; final text is under 'output'
            console.print("[bold green]agent>[/bold green]")
            output = result.get('output', result)
            format_agent_output(str(output))
            console.print()  # Add spacing after response
            # Save AI response to DB
            save_message(session_id, "assistant", str(output))
        except Exception as e:
            # Don’t hide failures while you’re iterating
            console.print(f"[bold red][error][/bold red] [red]{type(e).__name__}: {e}[/red]\n")