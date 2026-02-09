"""Streamlit demo application for dialogue policy learning."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import random
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.envs import make_dialogue_env, DialogueAction
from src.algorithms import PPOTrainer, SACTrainer, PolicyGradientTrainer, TrainingConfig
from src.eval.metrics import DialogueEvaluator, DialogueMetrics
from src.utils.seeding import set_seed


# Page configuration
st.set_page_config(
    page_title="Dialogue Policy Learning Demo",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safety warning
st.error("""
**⚠️ SAFETY WARNING ⚠️**

This is a research/educational project for dialogue policy learning.
**NOT FOR PRODUCTION CONTROL OF REAL SYSTEMS.**

This system is designed for:
- Research and educational purposes
- Demonstrating RL concepts in dialogue systems
- Academic study of conversation policies

**DO NOT USE for:**
- Production dialogue systems
- Real-world customer service
- Critical decision-making systems
- Systems requiring guaranteed safety or reliability
""")

# Title
st.title("💬 Dialogue Policy Learning Demo")
st.markdown("Interactive demonstration of reinforcement learning for dialogue policy learning")

# Sidebar
st.sidebar.header("Configuration")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["PPO", "SAC", "Policy Gradient"],
    help="Choose the RL algorithm to use"
)

# Environment parameters
st.sidebar.subheader("Environment Settings")
max_turns = st.sidebar.slider("Max Turns", 5, 50, 20)
vocab_size = st.sidebar.slider("Vocabulary Size", 100, 2000, 1000)
embedding_dim = st.sidebar.slider("Embedding Dimension", 64, 512, 128)

# Training parameters
st.sidebar.subheader("Training Settings")
learning_rate = st.sidebar.slider("Learning Rate", 1e-5, 1e-2, 3e-4, format="%.2e")
gamma = st.sidebar.slider("Discount Factor (γ)", 0.9, 0.999, 0.99)
total_timesteps = st.sidebar.slider("Total Timesteps", 1000, 100000, 10000)

# Evaluation parameters
st.sidebar.subheader("Evaluation Settings")
num_eval_episodes = st.sidebar.slider("Evaluation Episodes", 10, 200, 50)
deterministic = st.sidebar.checkbox("Deterministic Evaluation", True)

# Random seed
seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=2**32-1)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🎮 Interactive Demo", "📊 Training", "📈 Evaluation", "🔍 Analysis"])

with tab1:
    st.header("Interactive Dialogue Demo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Environment")
        
        # Create environment
        if st.button("Create Environment", key="create_env"):
            set_seed(seed)
            env = make_dialogue_env(
                max_turns=max_turns,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                seed=seed,
            )
            st.session_state.env = env
            st.success("Environment created successfully!")
        
        if "env" in st.session_state:
            env = st.session_state.env
            
            # Reset environment
            if st.button("Reset Environment", key="reset_env"):
                obs, info = env.reset(seed=seed)
                st.session_state.obs = obs
                st.session_state.info = info
                st.session_state.done = False
                st.session_state.conversation_history = []
                st.success("Environment reset!")
            
            # Display current state
            if "obs" in st.session_state:
                st.subheader("Current State")
                st.json({
                    "Dialogue State": st.session_state.obs["dialogue_state"],
                    "Turn Count": st.session_state.obs["turn_count"],
                    "Required Info Mask": st.session_state.obs["required_info_mask"].tolist(),
                    "Provided Info Mask": st.session_state.obs["provided_info_mask"].tolist(),
                })
                
                # Action selection
                st.subheader("Action Selection")
                action_names = [action.name for action in DialogueAction]
                selected_action = st.selectbox("Choose Action", action_names)
                action_idx = DialogueAction[selected_action].value
                
                if st.button("Take Action", key="take_action"):
                    if not st.session_state.done:
                        obs, reward, terminated, truncated, info = env.step(action_idx)
                        st.session_state.obs = obs
                        st.session_state.info = info
                        st.session_state.done = terminated or truncated
                        
                        # Store conversation
                        if hasattr(env.unwrapped, 'context') and env.unwrapped.context:
                            if env.unwrapped.context.conversation_history:
                                st.session_state.conversation_history.append(
                                    env.unwrapped.context.conversation_history[-1]
                                )
                        
                        st.success(f"Action taken! Reward: {reward:.2f}")
                        
                        if st.session_state.done:
                            st.warning("Episode completed!")
                    else:
                        st.error("Episode is already completed!")
    
    with col2:
        st.subheader("Conversation History")
        
        if "conversation_history" in st.session_state:
            if st.session_state.conversation_history:
                for i, (user_msg, agent_msg) in enumerate(st.session_state.conversation_history):
                    st.markdown(f"**Turn {i+1}:**")
                    st.markdown(f"👤 **User:** {user_msg}")
                    st.markdown(f"🤖 **Agent:** {agent_msg}")
                    st.markdown("---")
            else:
                st.info("No conversation yet. Take some actions to start!")
        
        # Environment info
        if "info" in st.session_state:
            st.subheader("Episode Info")
            info = st.session_state.info
            st.json({
                "User Intent": info.get("user_intent", "Unknown"),
                "Required Info": info.get("required_info", []),
                "Provided Info": info.get("provided_info", []),
                "Completion Rate": f"{info.get('completion_rate', 0.0):.2%}",
                "Episode Reward": info.get("episode_reward", 0.0),
            })

with tab2:
    st.header("Training")
    
    if st.button("Start Training", key="start_training"):
        with st.spinner("Training in progress..."):
            # Set seed
            set_seed(seed)
            
            # Create environment
            env = make_dialogue_env(
                max_turns=max_turns,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                seed=seed,
            )
            
            # Create training config
            config = TrainingConfig(
                learning_rate=learning_rate,
                gamma=gamma,
            )
            
            # Create agent
            if algorithm == "PPO":
                agent = PPOTrainer(env=env, config=config)
            elif algorithm == "SAC":
                agent = SACTrainer(env=env, config=config)
            else:  # Policy Gradient
                agent = PolicyGradientTrainer(env=env, config=config)
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate training (simplified)
            for step in range(0, total_timesteps, 1000):
                progress_bar.progress(step / total_timesteps)
                status_text.text(f"Training step {step}/{total_timesteps}")
                
                # Simulate training step
                if hasattr(agent, 'train'):
                    agent.train(1000)  # Train for 1000 steps
                elif hasattr(agent, 'train_episode'):
                    for _ in range(10):  # Train 10 episodes
                        agent.train_episode()
            
            progress_bar.progress(1.0)
            status_text.text("Training completed!")
            
            # Store trained agent
            st.session_state.trained_agent = agent
            st.session_state.training_env = env
            
            st.success("Training completed successfully!")

with tab3:
    st.header("Evaluation")
    
    if "trained_agent" in st.session_state:
        st.subheader("Agent Performance")
        
        if st.button("Evaluate Agent", key="evaluate_agent"):
            with st.spinner("Evaluating agent..."):
                # Create evaluator
                evaluator = DialogueEvaluator(
                    env=st.session_state.training_env,
                    num_eval_episodes=num_eval_episodes,
                    seed=seed,
                )
                
                # Evaluate agent
                metrics = evaluator.evaluate_agent(
                    agent=st.session_state.trained_agent,
                    deterministic=deterministic,
                )
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Success Rate", f"{metrics.success_rate:.2%}")
                    st.metric("Completion Rate", f"{metrics.completion_rate:.2%}")
                
                with col2:
                    st.metric("Average Return", f"{metrics.average_return:.2f}")
                    st.metric("Episode Length", f"{metrics.average_episode_length:.1f}")
                
                with col3:
                    st.metric("Coherence Score", f"{metrics.coherence_score:.3f}")
                    st.metric("Diversity Score", f"{metrics.diversity_score:.3f}")
                
                # Detailed metrics
                st.subheader("Detailed Metrics")
                metrics_df = pd.DataFrame([metrics.to_dict()])
                st.dataframe(metrics_df, use_container_width=True)
                
                # Store metrics for analysis
                st.session_state.evaluation_metrics = metrics
    else:
        st.info("Please train an agent first in the Training tab.")

with tab4:
    st.header("Analysis")
    
    if "evaluation_metrics" in st.session_state:
        metrics = st.session_state.evaluation_metrics
        
        # Performance visualization
        st.subheader("Performance Visualization")
        
        # Create performance chart
        performance_data = {
            "Metric": ["Success Rate", "Completion Rate", "Coherence Score", "Diversity Score"],
            "Value": [
                metrics.success_rate,
                metrics.completion_rate,
                metrics.coherence_score,
                metrics.diversity_score,
            ]
        }
        
        df = pd.DataFrame(performance_data)
        fig = px.bar(df, x="Metric", y="Value", title="Agent Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)
        
        # Return distribution (simulated)
        st.subheader("Return Distribution")
        returns = np.random.normal(metrics.average_return, metrics.return_std, 1000)
        fig = px.histogram(x=returns, title="Episode Return Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Action analysis
        st.subheader("Action Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Action Entropy", f"{metrics.action_entropy:.3f}")
        
        with col2:
            st.metric("Action Diversity", f"{metrics.action_diversity:.3f}")
        
        # Recommendations
        st.subheader("Recommendations")
        
        if metrics.success_rate < 0.5:
            st.warning("Low success rate. Consider increasing training time or adjusting hyperparameters.")
        
        if metrics.coherence_score < 0.7:
            st.warning("Low coherence score. The agent may need better dialogue modeling.")
        
        if metrics.diversity_score < 0.5:
            st.info("Low diversity score. Consider adding exploration bonuses.")
        
        if metrics.success_rate > 0.8 and metrics.coherence_score > 0.8:
            st.success("Excellent performance! The agent is performing well.")
    else:
        st.info("Please evaluate an agent first in the Evaluation tab.")

# Footer
st.markdown("---")
st.markdown("""
**About this Demo:**
This interactive demo showcases reinforcement learning for dialogue policy learning.
The system learns to conduct conversations by gathering required information
through natural dialogue interactions.

**Key Features:**
- Multiple RL algorithms (PPO, SAC, Policy Gradient)
- Interactive environment testing
- Comprehensive evaluation metrics
- Real-time performance visualization

**Educational Purpose:**
This demo is designed for educational and research purposes to understand
how RL can be applied to dialogue systems and conversation management.
""")
