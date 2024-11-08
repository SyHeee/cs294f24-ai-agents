import uuid
import asyncio
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
import logging
from playwright.async_api import Page, BrowserContext
from seeact.agent import SeeActAgent

class BrowserState:
    """Represents the state of a browser tab"""
    def __init__(self, 
                 url: str = None, 
                 screenshot_path: str = None,
                 observation: str = None,
                 action: str = None):
        self.url = url
        self.screenshot_path = screenshot_path
        self.observation = observation
        self.last_action = action
        self.timestamp = datetime.now()

    async def capture_from_page(self, page: Page, screenshot_path: str) -> None:
        """Capture state from a browser page"""
        self.url = page.url
        self.screenshot_path = screenshot_path
        await page.screenshot(path=screenshot_path)
        # can add more state capture logic here

class MCTSNode:
    """Node class for Monte Carlo Tree Search with browser state tracking"""
    def __init__(self, 
                 state: BrowserState,
                 parent: Optional['MCTSNode'] = None,
                 tab_id: str = None):
        # Basic MCTS properties
        self.id = str(uuid.uuid4())
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        
        # Browser state tracking
        self.state = state
        self.tab_id = tab_id
        self.is_terminal = False
        self.reward = 0
        
    def uct(self, exploration_weight: float = 1.414) -> float:
        """Calculate UCT value for node selection"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_weight * (2 * np.log(self.parent.visits) / self.visits) ** 0.5
        return exploitation + exploration

    def add_child(self, state: BrowserState, tab_id: str) -> 'MCTSNode':
        """Add a child node with the given state and tab ID"""
        child = MCTSNode(state=state, parent=self, tab_id=tab_id)
        self.children.append(child)
        return child

    def __str__(self) -> str:
        return f"Node(id={self.id[:8]}, visits={self.visits}, value={self.value:.3f}, depth={self.depth})"

class TabManager:
    """Manages browser tabs for MCTS state exploration"""
    def __init__(self, context: BrowserContext, max_tabs: int = 50):
        self.context = context
        self.max_tabs = max_tabs
        self.active_tabs: Dict[str, Page] = {}
        self.tab_states: Dict[str, BrowserState] = {}
        
    async def create_new_tab(self, parent_tab_id: Optional[str] = None) -> tuple[str, Page]:
        """Create a new tab, optionally copying state from parent tab"""
        # Clean up if we're at the tab limit
        await self.cleanup_if_needed()
        
        # Create new tab
        new_tab = await self.context.new_page()
        tab_id = str(uuid.uuid4())
        self.active_tabs[tab_id] = new_tab
        
        # Copy parent state if specified
        if parent_tab_id and parent_tab_id in self.active_tabs:
            parent_tab = self.active_tabs[parent_tab_id]
            await new_tab.goto(parent_tab.url)
            
        return tab_id, new_tab
    
    async def cleanup_if_needed(self) -> None:
        """Remove old tabs if we're at the limit"""
        if len(self.active_tabs) >= self.max_tabs:
            # Keep the most recently used tabs
            sorted_tabs = sorted(
                self.tab_states.items(),
                key=lambda x: x[1].timestamp
            )
            # Remove oldest tabs
            for tab_id, _ in sorted_tabs[:(len(self.active_tabs) - self.max_tabs + 1)]:
                await self.close_tab(tab_id)
    
    async def close_tab(self, tab_id: str) -> None:
        """Close a tab and clean up its resources"""
        if tab_id in self.active_tabs:
            await self.active_tabs[tab_id].close()
            del self.active_tabs[tab_id]
        if tab_id in self.tab_states:
            del self.tab_states[tab_id]
            
    async def get_tab(self, tab_id: str) -> Optional[Page]:
        """Get a tab by its ID"""
        return self.active_tabs.get(tab_id)

class MCTSBrowserManager:
    """Manages the integration of MCTS with browser automation"""
    def __init__(self, 
                 agent: 'SeeActAgent',  # Forward reference to SeeActAgent
                 max_tabs: int = 50,
                 exploration_weight: float = 1.414):
        self.agent = agent
        self.tab_manager = TabManager(agent.session_control['context'], max_tabs)
        self.exploration_weight = exploration_weight
        self.root: Optional[MCTSNode] = None
        
    async def initialize_root(self) -> MCTSNode:
        """Initialize the root node with the current browser state"""
        # Create initial tab
        tab_id, page = await self.tab_manager.create_new_tab()
        
        # Capture initial state
        state = BrowserState()
        screenshot_path = f"{self.agent.main_path}/screenshots/root.png"
        await state.capture_from_page(page, screenshot_path)
        
        # Create root node
        self.root = MCTSNode(state=state, tab_id=tab_id)
        return self.root
    
    async def create_child_state(self, 
                               parent_node: MCTSNode, 
                               action: str) -> tuple[MCTSNode, bool]:
        """Create a new node by applying an action in a new tab"""
        # Create new tab from parent
        tab_id, new_tab = await self.tab_manager.create_new_tab(parent_node.tab_id)
        
        try:
            # Execute action in new tab
            observation = await self.agent.perform_action(new_tab, action)
            
            # Capture new state
            state = BrowserState(
                url=new_tab.url,
                screenshot_path=f"{self.agent.main_path}/screenshots/node_{tab_id}.png",
                observation=observation,
                action=action
            )
            await state.capture_from_page(new_tab, state.screenshot_path)
            
            # Create new node
            new_node = parent_node.add_child(state, tab_id)
            return new_node, True
            
        except Exception as e:
            logging.error(f"Error creating child state: {e}")
            await self.tab_manager.close_tab(tab_id)
            return None, False