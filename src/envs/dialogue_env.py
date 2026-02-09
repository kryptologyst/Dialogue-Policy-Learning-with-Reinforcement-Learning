"""Dialogue environment for reinforcement learning policy training."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from enum import Enum


class DialogueAction(Enum):
    """Available dialogue actions."""
    GREET = 0
    ASK_QUESTION = 1
    PROVIDE_INFO = 2
    CLARIFY = 3
    CONFIRM = 4
    END_CONVERSATION = 5


class DialogueState(Enum):
    """Dialogue states."""
    INITIAL = 0
    GREETING = 1
    QUESTIONING = 2
    INFORMING = 3
    CLARIFYING = 4
    CONFIRMING = 5
    ENDED = 6


@dataclass
class DialogueContext:
    """Context for dialogue interactions."""
    user_intent: str
    required_info: List[str]
    provided_info: List[str]
    conversation_history: List[Tuple[str, str]]  # (user, agent)
    current_state: DialogueState
    max_turns: int = 20


class DialogueEnvironment(gym.Env):
    """
    A dialogue environment for training RL agents to learn conversation policies.
    
    The agent must gather required information from users through natural conversation
    while maintaining coherence and achieving the conversation goal.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        max_turns: int = 20,
        vocab_size: int = 1000,
        embedding_dim: int = 128,
        seed: Optional[int] = None,
    ):
        """Initialize the dialogue environment.
        
        Args:
            max_turns: Maximum number of conversation turns
            vocab_size: Size of vocabulary for text representation
            embedding_dim: Dimension of text embeddings
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.max_turns = max_turns
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(DialogueAction))
        self.observation_space = spaces.Dict({
            "dialogue_state": spaces.Discrete(len(DialogueState)),
            "turn_count": spaces.Discrete(max_turns + 1),
            "context_embedding": spaces.Box(
                low=-1.0, high=1.0, 
                shape=(embedding_dim,), 
                dtype=np.float32
            ),
            "required_info_mask": spaces.Box(
                low=0, high=1, 
                shape=(10,),  # Max 10 required info items
                dtype=np.float32
            ),
            "provided_info_mask": spaces.Box(
                low=0, high=1, 
                shape=(10,),  # Max 10 provided info items
                dtype=np.float32
            ),
        })
        
        # Initialize state
        self.context: Optional[DialogueContext] = None
        self.turn_count = 0
        self.episode_reward = 0.0
        
        # Set random seed
        if seed is not None:
            self.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed for reproducibility."""
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed(seed)
        
        # Sample a new dialogue scenario
        scenarios = [
            {
                "intent": "book_restaurant",
                "required_info": ["date", "time", "party_size", "cuisine_preference"]
            },
            {
                "intent": "buy_product",
                "required_info": ["product_type", "budget", "delivery_address", "urgency"]
            },
            {
                "intent": "schedule_meeting",
                "required_info": ["participants", "duration", "preferred_time", "location"]
            },
            {
                "intent": "get_technical_support",
                "required_info": ["problem_description", "system_info", "error_messages", "urgency"]
            }
        ]
        
        scenario = random.choice(scenarios)
        
        self.context = DialogueContext(
            user_intent=scenario["intent"],
            required_info=scenario["required_info"],
            provided_info=[],
            conversation_history=[],
            current_state=DialogueState.INITIAL,
            max_turns=self.max_turns
        )
        
        self.turn_count = 0
        self.episode_reward = 0.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.context is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Convert action to DialogueAction
        dialogue_action = DialogueAction(action)
        
        # Execute action and get response
        agent_response = self._execute_action(dialogue_action)
        
        # Simulate user response
        user_response = self._simulate_user_response(dialogue_action)
        
        # Update conversation history
        self.context.conversation_history.append((user_response, agent_response))
        
        # Update state based on action
        self._update_state(dialogue_action)
        
        # Calculate reward
        reward = self._calculate_reward(dialogue_action, user_response)
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.turn_count >= self.max_turns
        
        # Update turn count
        self.turn_count += 1
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: DialogueAction) -> str:
        """Execute the given dialogue action and return agent response."""
        responses = {
            DialogueAction.GREET: [
                "Hello! How can I help you today?",
                "Hi there! What can I assist you with?",
                "Good day! How may I be of service?"
            ],
            DialogueAction.ASK_QUESTION: [
                "Could you tell me more about that?",
                "I need some additional information. Could you help?",
                "What specific details would you like to share?"
            ],
            DialogueAction.PROVIDE_INFO: [
                "Based on what you've told me, here's what I can suggest...",
                "I have some information that might help you.",
                "Let me provide you with some details about that."
            ],
            DialogueAction.CLARIFY: [
                "Just to make sure I understand correctly...",
                "Could you clarify what you mean by that?",
                "I want to make sure I have the right information."
            ],
            DialogueAction.CONFIRM: [
                "So to confirm, you're saying that...",
                "Let me make sure I have this right...",
                "Just to double-check, you want..."
            ],
            DialogueAction.END_CONVERSATION: [
                "Thank you for your time. Have a great day!",
                "I think we've covered everything. Take care!",
                "That's all I need to know. Goodbye!"
            ]
        }
        
        return random.choice(responses[action])
    
    def _simulate_user_response(self, agent_action: DialogueAction) -> str:
        """Simulate user response based on agent action."""
        # Simple user simulation - in practice, this would be more sophisticated
        responses = {
            DialogueAction.GREET: [
                f"I need help with {self.context.user_intent}.",
                f"I'm looking for assistance with {self.context.user_intent}.",
                f"Can you help me with {self.context.user_intent}?"
            ],
            DialogueAction.ASK_QUESTION: [
                "Sure, what would you like to know?",
                "I can provide more details.",
                "What specific information do you need?"
            ],
            DialogueAction.PROVIDE_INFO: [
                "That's helpful, thank you.",
                "I see, that makes sense.",
                "Good to know, thanks for the information."
            ],
            DialogueAction.CLARIFY: [
                "Let me explain that better...",
                "What I mean is...",
                "To clarify..."
            ],
            DialogueAction.CONFIRM: [
                "Yes, that's correct.",
                "Exactly right.",
                "You've got it."
            ],
            DialogueAction.END_CONVERSATION: [
                "Thank you for your help!",
                "Goodbye!",
                "See you later!"
            ]
        }
        
        return random.choice(responses[agent_action])
    
    def _update_state(self, action: DialogueAction) -> None:
        """Update dialogue state based on action."""
        if self.context is None:
            return
        
        state_transitions = {
            DialogueAction.GREET: DialogueState.GREETING,
            DialogueAction.ASK_QUESTION: DialogueState.QUESTIONING,
            DialogueAction.PROVIDE_INFO: DialogueState.INFORMING,
            DialogueAction.CLARIFY: DialogueState.CLARIFYING,
            DialogueAction.CONFIRM: DialogueState.CONFIRMING,
            DialogueAction.END_CONVERSATION: DialogueState.ENDED,
        }
        
        self.context.current_state = state_transitions[action]
    
    def _calculate_reward(self, action: DialogueAction, user_response: str) -> float:
        """Calculate reward for the current action."""
        if self.context is None:
            return 0.0
        
        reward = 0.0
        
        # Base reward for maintaining conversation flow
        reward += 0.1
        
        # Reward for gathering required information
        if action == DialogueAction.ASK_QUESTION:
            # Simulate information gathering (in practice, would use NLP)
            if random.random() < 0.3:  # 30% chance of gathering info
                if len(self.context.provided_info) < len(self.context.required_info):
                    self.context.provided_info.append("info_item")
                    reward += 1.0
        
        # Reward for appropriate conversation ending
        if action == DialogueAction.END_CONVERSATION:
            completion_rate = len(self.context.provided_info) / len(self.context.required_info)
            reward += completion_rate * 5.0
        
        # Penalty for inappropriate actions
        if self.context.current_state == DialogueState.INITIAL and action != DialogueAction.GREET:
            reward -= 0.5
        
        # Penalty for ending too early
        if action == DialogueAction.END_CONVERSATION and len(self.context.provided_info) < len(self.context.required_info) * 0.5:
            reward -= 2.0
        
        return reward
    
    def _is_terminated(self) -> bool:
        """Check if the dialogue should be terminated."""
        if self.context is None:
            return True
        
        return (
            self.context.current_state == DialogueState.ENDED or
            len(self.context.provided_info) >= len(self.context.required_info)
        )
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        if self.context is None:
            raise RuntimeError("Environment not initialized")
        
        # Create context embedding (simplified)
        context_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        
        # Create info masks
        required_info_mask = np.zeros(10, dtype=np.float32)
        provided_info_mask = np.zeros(10, dtype=np.float32)
        
        for i, info in enumerate(self.context.required_info[:10]):
            required_info_mask[i] = 1.0
        
        for i, info in enumerate(self.context.provided_info[:10]):
            provided_info_mask[i] = 1.0
        
        return {
            "dialogue_state": self.context.current_state.value,
            "turn_count": self.turn_count,
            "context_embedding": context_embedding,
            "required_info_mask": required_info_mask,
            "provided_info_mask": provided_info_mask,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        if self.context is None:
            return {}
        
        return {
            "user_intent": self.context.user_intent,
            "required_info": self.context.required_info,
            "provided_info": self.context.provided_info,
            "conversation_history": self.context.conversation_history,
            "episode_reward": self.episode_reward,
            "completion_rate": len(self.context.provided_info) / len(self.context.required_info) if self.context.required_info else 0.0,
        }
    
    def render(self, mode: str = "human") -> Optional[Union[str, np.ndarray]]:
        """Render the environment."""
        if self.context is None:
            return None
        
        if mode == "human":
            print(f"\n=== Dialogue Turn {self.turn_count} ===")
            print(f"State: {self.context.current_state.name}")
            print(f"Intent: {self.context.user_intent}")
            print(f"Required Info: {self.context.required_info}")
            print(f"Provided Info: {self.context.provided_info}")
            print(f"Completion Rate: {len(self.context.provided_info) / len(self.context.required_info):.2f}")
            if self.context.conversation_history:
                print("Recent conversation:")
                for user_msg, agent_msg in self.context.conversation_history[-3:]:
                    print(f"  User: {user_msg}")
                    print(f"  Agent: {agent_msg}")
            print("=" * 50)
        
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        pass
