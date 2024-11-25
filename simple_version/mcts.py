import numpy as np
import logging
from abc import ABC
from environment import WikiEnv, HotPotQAWrapper, LoggingWrapper
import requests
import string
from dataclasses import dataclass

env = WikiEnv()
env = HotPotQAWrapper(env, split="train")
env = LoggingWrapper(env)


class gpt(ABC):
    def __init__(self):
        ...
    def generate_samples(cls, promptn, n_samples, stop=None):
        ...



@dataclass
class State:
    """ Definition of a state in the tree.
    """
    ###[TODO]: Define the state variables.
    thought: str = ''
    action: str = ''
    observation: str = ''

    def __str__(self):
        return f"State(thought={self.thought}, action={self.action}, observation={self.observation})"


class Node:
    """ Definition of a basic node in the tree.
    """
    def __init__(self, parent=None):
        
        self.parent = parent
        self.children = []
        self.value = -1
        self.depth = 0 # by default it is a root node
        self.id = '0a' # by default it is a root node

    def __str__(self):
        """Return a string representation of the node.
        """
        return f"Node(id={self.id}, depth={self.depth}, value={self.value:.2f})"

    def _reset_node_id(self):
        """Reset the node id for all the children. e.g. if a node is at depth 2 and has 3 children, 
        the children will be at depth 3 and their node id will be 3a, 3b, 3c.
        """
        if self.children == []:
            return
        alphabet = string.ascii_lowercase
        for i, child in self.children:
            child.depth = self.depth + 1
            child.id =str(i)+alphabet[i]


    def state_str(self):
        """Return a string representation of the node's state.
        [TODO]: Exclude the observation for root node.
        """
        state = f"Thought {self.depth}: {self.state['thought']}\n" + \
                f"Action: {self.depth}: {self.state['action']}\n" + \
                f"Observation {self.depth}: {self.state['observation']}"
        
        return state
    
    def to_dict(self):
        """Return a dictionary representation of the node.
        """
        return {
            'parent': self.parent.__str__() if self.parent else None,
            'children': [child.__str__() for child in self.children],
            'value': self.value,
            'depth': self.depth,
        }
    
    @property
    def is_leaf(self):
        """Return True if the node is a leaf node.
        """
        return self.children == []



class MCTSNode(Node):
    """ Definition of a node in the MCTS tree.
    """
    def __init__(self, parent=None, state=None, question=None):
        super().__init__(state, question, parent)
        self.state = State() if state is None else state
        self.question = question
        self.visits = 0
        self.is_terminal = False
        self.reward = 0
        self.exhausted = False # If all children are terminal
        self.em = 0  # Exact match, evaluation metric


    def __str__(self):
        """Return a string representation of the node.
        """
        return f"Node(id={self.id}, depth={self.depth}, value={self.value:.2f}, visits={self.visits})"


    def to_dict(self):
        """Return a dictionary representation of the node.
        """
        return {
            'state': self.state,
            'question': self.question,
            'parent': self.parent.__str__() if self.parent else None,
            'children': [child.__str__() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
            'em': self.em,
        }

    
    def uct(self, C1=1, C2=1, with_depth=False):
        """Upper Confidence bounds applied to Trees (Kocsis and Szepesv ́ ari, 2006). This implementation is copied from the LATS code but slight different from the LATS paper.

        Parameters:
        -----------
        C1 : float
            Exploration weihgt.
        C2 : float
            Depth weight. Effective only if `with_depth` is True.
        with_depth : bool
            Whether to include depth term in the UCT formula.

        Returns:
        --------
        y : float
            The UCT value.
        """
        y = 0
        if self.visits == 0:
            y = self.value
        else:
            exploitation_term = self.value / self.visits
            exploration_term = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
            depth_term = self.depth if with_depth else 0
            y = exploitation_term + C1 * exploration_term + C2 * depth_term
        return y


    def select(self):
        """Select a child node to explore next.
        Note that reward == 1 is a subset of terminal nodes.
        Tree Polocy:
        1) If current node is terminal with reward 0, select a sibling node.
        2) Try to select a terminal child with reward 1.
        3) If 2) failed, try to select a non-terminal child with the highest UCT value.
        4) If 3) failed, backtrack to the parent node select a sibling.
        """
        # If the current node is a leaf node, return itself.
        if self.is_leaf:
            return self
        
        # The below block appears in oroginal LATS code, but it is not possible in the first place. Mostly skipped.
        if self.is_terminal and self.reward != 1:
            logging.info(f"Current node is terminal with reward {self.reward}.")
            return self.parents.select()

        # Start the selection
        logging.info(f"Selecting from {len(self.children)} children at depth {self.depth}.")

        non_terminal_children = [child for child in self.children if not child.is_terminal]
        terminal_children = [child for child in self.children if child.is_terminal]
        terminal_children_reward_1 = [child for child in terminal_children if child.reward == 1]
            
        # if no terminal child has reward 1, select the child node with the highest UCT value.
        node_with_reward_1 = next(terminal_children_reward_1, None)
        if node_with_reward_1:
            logging.info(f"Found terminal node with reward 1 at depth {self.depth}.")
            return node_with_reward_1.select()
        else:
            node_max_uct = max(non_terminal_children, key=lambda child: child.uct(), default=None)
            if node_max_uct:
                logging.info(f"Selected node at depth {node_max_uct.depth} with UCT {node_max_uct.uct()}.")
                return node_max_uct.select()
            else:
                logging.info(f"All children are terminal at depth {self.depth}. Backtracking...")
                if self.parent: 
                    self.parent.children.remove(self)
                return self.parents.select() 


    def expand(self, n_nodes, max_depth=7):
        """Expand the current node by adding children node.
        """
        if self.is_terminal:
            logging.info("Terminal node cannot be expanded.")
            return
        
        if self.depth >= max_depth:
            logging.info("Depth limit reached")
            self.is_terminal = True
            return
        question = self.question
        new_nodes = self._generate_new_states(n_nodes, question)
        self.children.extend(new_nodes)


    def _generate_prompt_trajectory(self, trajectory):
        """ Iteratively generate prompt for the current node and its ancestors.
        Usually the input `trajectory` is [question]. So the returned trajectory will look like
        [question, state_str_n, state_str_n-1, ..., state_str_0].
        """
        trajectory.append(self.state_str())
        while self.parent:
            trajectory = self.parent._generate_prompt(trajectory)
        return trajectory


    def _generate_new_states(self, n_states, question):
        """Generate new states by sampling from the GPT-3 model."""

        # Generate prompt of the entire trajectory for LLM to generate actions
        prompt = self._generate_prompt_trajectory([question])
        # Sample actions from the LLM
        sampled_actions = gpt.generate_samples(prompt, n_states)
        
        unique_states = {}  # Store unique states here
        for action in sampled_actions:
            new_state = self.state.copy()  # Make a copy of the current node's state

            thought_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith(f"Thought {self.depth + 1}")), '')
            action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)

            # Use thought and action to form a unique key
            unique_key = f"{thought_line}::{action_line}"
            
            if unique_key in unique_states:
                continue  # Skip if this state already exists
            
            if action_line:
                action_type = action_line.split('[')[0] if '[' in action_line else action_line
                action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""
                action = f"{action_type.lower()}[{action_param}]"
                attempts = 0
                while attempts < 10:
                    try:
                        observation, reward, terminated, truncated, info = env.step(action)
                    except requests.exceptions.Timeout:
                        attempts += 1

                # Update the new state dictionary
                new_state['thought'] = thought_line
                new_state['action'] = action_line
                new_state['observation'] = observation

                new_node = Node(state=new_state, question=question, parent=self)
                new_node.is_terminal = (reward == 1) or terminated
                new_node.reward = reward
                
                if reward == 1:
                    new_node.em = info.get('em')
                unique_states[unique_key] = new_node  # Add this state to unique_states
                
                logging.info(f"NEW NODE: {new_node}")
                logging.info(f"Feedback: {info}")

        return list(unique_states.values())  # Return unique nodes as a list
    

    def evaluate(self, reward):
        """Update the value of the node.
        """
        self.visits += 1
        self.value += reward


    def backpropagate(self, reward):
        """Backpropagate the reward up the tree.
        """
        self.evaluate(reward)
        if self.parent:
            self.parent.backpropagate(reward)


    def pick_best_child(self, by="visits"):
        """Choose the child node with the highest value.
        """
        if self.is_leaf:
            return self
        
        if by == "visits":
            key = lambda x: x.visits
        elif by == "value":
            key = lambda x: x.value
        else:
            raise ValueError(f"Invalid key: {by}")
        
        return max(self.children, key=key, default=None)


    def mcts_search(idx, n_iterations=50):
        """Perform MCTS search on an MCTCNode.
        The four steps of MCTS (select, expand, evaluate, backpropagate) are defined as member methods of the MCTSNode class.
        """
        x = env.reset(idx)
        root = MCTSNode(state=None, question=x)
        for i in range(n_iterations):
            node = root.select()
            node.expand(n_nodes=5)
            node.evaluate()
            node.backpropagate()

        return root.pick_best_child()