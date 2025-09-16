from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_perplexity import ChatPerplexity
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import json
import os
from pathlib import Path
import re
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

# Initialize Rich console
console = Console()

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
llm = ChatPerplexity(model="sonar-pro", temperature=0.7, timeout=60, api_key = os.getenv("PPLX_API_KEY"))

# Create a prompt template for our agent
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant powered by Perplexity Pro that can find real-time information on the web.
    When asked about restaurants, search for relevant information and provide:
    1. The name and address of the restaurant
    2. Top 5 popular menu items if available
    3. Brief description of the restaurant
    
    For other topics, use your web search capabilities to provide accurate, up-to-date information.
    Always be concise but informative and remember previous conversations."""),
    MessagesPlaceholder("chat_history", optional=True),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Create the agent
agent = create_openai_functions_agent(llm, [], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)

# Persistent memory storage
MEMORY_DIR = Path("chat_memories")
MEMORY_DIR.mkdir(exist_ok=True)

def load_memory(session_id: str) -> ChatMessageHistory:
    """Load chat history from persistent storage"""
    memory_file = MEMORY_DIR / f"{session_id}.json"
    history = ChatMessageHistory()
    
    if memory_file.exists():
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Reconstruct messages from saved data
                for msg_data in data:
                    if msg_data['type'] == 'human':
                        history.add_user_message(msg_data['content'])
                    elif msg_data['type'] == 'ai':
                        history.add_ai_message(msg_data['content'])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load memory for {session_id}: {e}")
    
    return history

def save_memory(session_id: str, history: ChatMessageHistory):
    """Save chat history to persistent storage"""
    memory_file = MEMORY_DIR / f"{session_id}.json"
    
    # Convert messages to serializable format
    messages_data = []
    for msg in history.messages:
        messages_data.append({
            'type': msg.type,
            'content': msg.content
        })
    
    try:
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save memory for {session_id}: {e}")

store = {}

def get_history(cfg_or_sid):
    # Accept either a simple session_id string or a dict-like config
    if isinstance(cfg_or_sid, str):
        sid = cfg_or_sid
    else:
        sid = (
            cfg_or_sid.get("configurable", {}).get("session_id")
            or cfg_or_sid.get("session_id")
            or "default"
        )
    
    # Load from persistent storage if not in memory
    if sid not in store:
        store[sid] = load_memory(sid)
    
    return store[sid]

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

if __name__ == "__main__":
    import os

    # Stable session so history threads correctly across turns
    session_id = os.environ.get("SESSION_ID", "mack")

    console.print("\n[bold green]🤖 Memory test shell with PERSISTENT storage.[/bold green]")
    console.print("[dim]Type 'exit' to quit. Commands: /history, /clear, /sessions[/dim]\n")
    while True:
        try:
            # Create a styled prompt
            console.print("[bold blue]you>[/bold blue] ", end="")
            user_prompt = input().strip()
        except (KeyboardInterrupt, EOFError):
            # Save memory before exiting
            if session_id in store:
                save_memory(session_id, store[session_id])
            console.print("\n[yellow]Saved memory. bye 👋[/yellow]")
            break

        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit", ":q", "/exit"}:
            # Save memory before exiting
            if session_id in store:
                save_memory(session_id, store[session_id])
            console.print("[yellow]Saved memory. bye 👋[/yellow]")
            break
        if user_prompt == "/history":
            hist = store.get(session_id)
            if not hist or not hist.messages:
                console.print("[dim](no history yet)[/dim]\n")
                continue
            console.print(f"[bold cyan]=== History for session '{session_id}' ===[/bold cyan]")
            for i, m in enumerate(hist.messages[-10:]):  # last 10
                role = m.type if hasattr(m, "type") else m.__class__.__name__
                role_color = "blue" if role == "human" else "green"
                console.print(f"[dim]\\[{i+1}][/dim] [bold {role_color}][{role}][/bold {role_color}] {getattr(m, 'content', m)}")
            console.print()
            continue
        if user_prompt == "/clear":
            if session_id in store:
                store[session_id] = ChatMessageHistory()
                memory_file = MEMORY_DIR / f"{session_id}.json"
                if memory_file.exists():
                    memory_file.unlink()
                console.print(f"[yellow]Cleared memory for session '{session_id}'[/yellow]\n")
            continue
        if user_prompt == "/sessions":
            memory_files = list(MEMORY_DIR.glob("*.json"))
            if not memory_files:
                console.print("[dim]No saved sessions found[/dim]\n")
                continue
            console.print("[bold cyan]=== Saved Sessions ===[/bold cyan]")
            for f in memory_files:
                session_name = f.stem
                indicator = " [green](current)[/green]" if session_name == session_id else ""
                console.print(f"  • [white]{session_name}[/white]{indicator}")
            console.print()
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
            
            # Save memory after each interaction
            if session_id in store:
                save_memory(session_id, store[session_id])
        except Exception as e:
            # Don’t hide failures while you’re iterating
            console.print(f"[bold red][error][/bold red] [red]{type(e).__name__}: {e}[/red]\n")