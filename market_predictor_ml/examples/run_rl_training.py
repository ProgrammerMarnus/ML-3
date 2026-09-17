"""
Example: Run RL Training for Position Sizing.
Demonstrates training a PPO agent for dynamic position sizing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from market_predictor_ml.data.loader import download_stock_data
from market_predictor_ml.features.engineering import create_all_features
from market_predictor_ml.rl.environment import TradingEnv
from market_predictor_ml.rl.agents import create_ppo_agent


def main():
    print("=" * 60)
    print("RL Training Example: Dynamic Position Sizing")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = download_stock_data("AAPL", start_date="2018-01-01", end_date="2023-01-01")
    print(f"   Loaded {len(df)} days of data")
    
    # Generate features
    print("\n2. Generating features...")
    df_features = create_all_features(df)
    df_features = df_features.dropna()
    print(f"   Created {len(df_features.columns)} features")
    
    # Create environment
    print("\n3. Creating trading environment...")
    env = TradingEnv(
        df=df_features,
        initial_balance=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        reward_type="sharpe",
    )
    
    # Create PPO agent
    print("\n4. Initializing PPO agent...")
    try:
        agent = create_ppo_agent(
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=5,
            verbose=1,
        )
        
        # Train
        print("\n5. Training agent (10,000 timesteps)...")
        agent.learn(total_timesteps=10000)
        
        print("\n✓ Training complete!")
        
        # Save agent
        agent.save("rl_ppo_trader")
        print("   Model saved to: rl_ppo_trader.zip")
        
        # Test
        print("\n6. Testing trained agent...")
        obs, _ = env.reset()
        episode_reward = 0
        
        for _ in range(100):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            if done:
                break
        
        print(f"   Test episode reward: {episode_reward:.4f}")
        print(f"   Final portfolio value: ${info['portfolio_value']:.2f}")
        
    except ImportError as e:
        print(f"\n⚠ stable-baselines3 not installed: {e}")
        print("   Install with: pip install stable-baselines3")
        return
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
