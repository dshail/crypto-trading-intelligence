"""
Crypto Trading Intelligence: Direct Real Data Analysis
======================================================

This module contains the core analysis pipeline for cryptocurrency trading 
sentiment correlation and behavioral trader segmentation.

Author: dshail  
Date: September 2025
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class CryptoTradingAnalyzerDirect:
    """
    Direct mapping version for Hyperliquid data analysis.
    Uses exact column names from actual dataset.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize data containers
        self.fear_greed_df = None
        self.trading_df = None
        self.merged_df = None
        self.trader_profiles = None
        self.models = {}
        
        print("🚀 Direct Mapping Crypto Trading Analyzer Initialized")
        print("="*60)
    
    def load_fear_greed_data(self, file_path):
        """Load Fear & Greed Index data."""
        print("📊 Loading Real Fear & Greed Index Data...")
        
        try:
            self.fear_greed_df = pd.read_csv(file_path)
            
            # Simple column detection for F&G data
            columns = self.fear_greed_df.columns.tolist()
            print(f"📋 F&G columns: {columns}")
            
            # Handle common F&G data formats
            if 'Date' in columns or 'date' in columns:
                date_col = 'Date' if 'Date' in columns else 'date'
                self.fear_greed_df['Date'] = pd.to_datetime(self.fear_greed_df[date_col])
            else:
                # Use first column as date
                self.fear_greed_df['Date'] = pd.to_datetime(self.fear_greed_df.iloc[:, 0])
            
            # Find Fear & Greed Index column
            if 'FearGreedIndex' in columns:
                fg_col = 'FearGreedIndex'
            elif any('fear' in col.lower() and 'greed' in col.lower() for col in columns):
                fg_col = next(col for col in columns if 'fear' in col.lower() and 'greed' in col.lower())
            elif 'value' in columns:
                fg_col = 'value'
            else:
                # Use second column as F&G value
                fg_col = columns[1]
            
            self.fear_greed_df['FearGreedIndex'] = pd.to_numeric(self.fear_greed_df[fg_col], errors='coerce')
            
            # Create Classification
            self.fear_greed_df['Classification'] = self.fear_greed_df['FearGreedIndex'].apply(
                lambda x: 'Fear' if x <= 30 else 'Greed' if x >= 70 else 'Neutral'
            )
            
            # Clean data
            self.fear_greed_df = self.fear_greed_df[['Date', 'FearGreedIndex', 'Classification']].dropna()
            
            print(f"✅ Loaded {len(self.fear_greed_df)} days of F&G data")
            print(f"   - Date range: {self.fear_greed_df['Date'].min()} to {self.fear_greed_df['Date'].max()}")
            
            return self.fear_greed_df
            
        except Exception as e:
            print(f"❌ Error loading F&G data: {str(e)}")
            raise
    
    def load_trading_data(self, file_path):
        """
        Load trading data using EXACT column names from dataset.

        Your exact columns:
        ['Account', 'Coin', 'Execution Price', 'Size Tokens', 'Size USD', 'Side', 
         'Timestamp IST', 'Start Position', 'Direction', 'Closed PnL', 
         'Transaction Hash', 'Order ID', 'Crossed', 'Fee', 'Trade ID', 'Timestamp']
        """
        print("\n📈 Loading Hyperliquid Trading Data with Direct Mapping...")
        
        try:
            # Load the CSV file
            self.trading_df = pd.read_csv(file_path)
            
            print(f"📊 Original data shape: {self.trading_df.shape}")
            print(f"📋 Available columns: {list(self.trading_df.columns)}")
            
            # DIRECT COLUMN MAPPING - No detection, just direct assignment
            print("\n🎯 Applying Direct Column Mapping...")
            
            # Create new dataframe with standardized column names
            standardized_df = pd.DataFrame()
            
            # Map exact column names to standardized names
            try:
                standardized_df['account'] = self.trading_df['Account']
                print("   ✅ account ← 'Account'")
            except KeyError:
                print("   ❌ 'Account' column not found")
                raise
            
            try:
                standardized_df['symbol'] = self.trading_df['Coin']
                print("   ✅ symbol ← 'Coin'")
            except KeyError:
                print("   ❌ 'Coin' column not found")
                raise
            
            try:
                standardized_df['execution_price'] = pd.to_numeric(self.trading_df['Execution Price'], errors='coerce')
                print("   ✅ execution_price ← 'Execution Price'")
            except KeyError:
                print("   ❌ 'Execution Price' column not found")
                raise
            
            # Handle SIZE - you have both 'Size Tokens' and 'Size USD'
            # Let's use 'Size USD' as it's more standardized for analysis
            try:
                if 'Size USD' in self.trading_df.columns:
                    standardized_df['size'] = pd.to_numeric(self.trading_df['Size USD'], errors='coerce')
                    print("   ✅ size ← 'Size USD'")
                elif 'Size Tokens' in self.trading_df.columns:
                    standardized_df['size'] = pd.to_numeric(self.trading_df['Size Tokens'], errors='coerce')
                    print("   ✅ size ← 'Size Tokens'")
                else:
                    raise KeyError("Neither 'Size USD' nor 'Size Tokens' found")
            except Exception as e:
                print(f"   ❌ Size column error: {e}")
                raise
            
            try:
                standardized_df['side'] = self.trading_df['Side']
                print("   ✅ side ← 'Side'")
            except KeyError:
                print("   ❌ 'Side' column not found")
                raise
            
            # Handle TIME - you have both 'Timestamp IST' and 'Timestamp'
            try:
                if 'Timestamp IST' in self.trading_df.columns:
                    standardized_df['time'] = pd.to_datetime(self.trading_df['Timestamp IST'], errors='coerce')
                    print("   ✅ time ← 'Timestamp IST'")
                elif 'Timestamp' in self.trading_df.columns:
                    standardized_df['time'] = pd.to_datetime(self.trading_df['Timestamp'], errors='coerce')
                    print("   ✅ time ← 'Timestamp'")
                else:
                    raise KeyError("Neither 'Timestamp IST' nor 'Timestamp' found")
            except Exception as e:
                print(f"   ❌ Timestamp column error: {e}")
                raise
            
            try:
                standardized_df['closedPnL'] = pd.to_numeric(self.trading_df['Closed PnL'], errors='coerce')
                print("   ✅ closedPnL ← 'Closed PnL'")
            except KeyError:
                print("   ❌ 'Closed PnL' column not found")
                raise
            
            # Handle LEVERAGE - if not present, set to 1.0
            if 'Leverage' in self.trading_df.columns:
                standardized_df['leverage'] = pd.to_numeric(self.trading_df['Leverage'], errors='coerce').fillna(1.0)
                print("   ✅ leverage ← 'Leverage'")
            else:
                standardized_df['leverage'] = 1.0
                print("   ℹ️  leverage set to default 1.0 (column not found)")
            
            # Replace original dataframe
            self.trading_df = standardized_df.copy()
            
            # Clean data - remove rows with essential missing values
            initial_rows = len(self.trading_df)
            self.trading_df = self.trading_df.dropna(subset=['account', 'symbol', 'execution_price', 'size', 'closedPnL', 'time'])
            final_rows = len(self.trading_df)
            
            if initial_rows > final_rows:
                print(f"   🧹 Cleaned data: removed {initial_rows - final_rows} rows with missing values")
            
            # Add derived columns
            self.trading_df['Date'] = self.trading_df['time'].dt.date
            self.trading_df['Date'] = pd.to_datetime(self.trading_df['Date'])
            self.trading_df['hour'] = self.trading_df['time'].dt.hour
            self.trading_df['day_of_week'] = self.trading_df['time'].dt.dayofweek
            self.trading_df['is_weekend'] = (self.trading_df['day_of_week'] >= 5).astype(int)
            
            print(f"\n✅ Successfully loaded {len(self.trading_df)} trading records")
            print(f"   - Unique traders: {self.trading_df['account'].nunique()}")
            print(f"   - Unique symbols: {self.trading_df['symbol'].nunique()}")
            print(f"   - Symbols: {sorted(self.trading_df['symbol'].unique())}")
            print(f"   - Date range: {self.trading_df['Date'].min()} to {self.trading_df['Date'].max()}")
            print(f"   - Total PnL: ${self.trading_df['closedPnL'].sum():,.2f}")
            
            # Display sample
            print(f"\n📋 Sample of Processed Data:")
            display_cols = ['account', 'symbol', 'execution_price', 'size', 'side', 'closedPnL']
            print(self.trading_df[display_cols].head().to_string())
            
            return self.trading_df
            
        except Exception as e:
            print(f"❌ Error loading trading data: {str(e)}")
            raise
    
    def merge_datasets(self):
        """Merge trading data with sentiment data."""
        print("\n🔗 Merging Datasets...")
        
        if self.fear_greed_df is None or self.trading_df is None:
            raise ValueError("Must load both datasets first")
        
        # Merge on Date
        initial_trades = len(self.trading_df)
        self.merged_df = self.trading_df.merge(self.fear_greed_df, on='Date', how='left')
        
        # Handle missing sentiment data
        missing_sentiment = self.merged_df['FearGreedIndex'].isna().sum()
        if missing_sentiment > 0:
            print(f"⚠️  {missing_sentiment} trades missing sentiment data (excluding from analysis)")
            self.merged_df = self.merged_df.dropna(subset=['FearGreedIndex'])
        
        print(f"✅ Merged dataset: {len(self.merged_df)} records with sentiment data")
        print(f"   - Coverage: {len(self.merged_df)/initial_trades*100:.1f}% of trades have sentiment data")
        
        return self.merged_df
    
    def validate_data_quality(self):
        """Validate the merged dataset."""
        print("\n🔍 Data Quality Validation")
        print("="*40)
        
        print("📊 Dataset Overview:")
        print(f"   Total records: {len(self.merged_df):,}")
        print(f"   Date range: {self.merged_df['Date'].min()} to {self.merged_df['Date'].max()}")
        print(f"   Trading days: {self.merged_df['Date'].nunique()}")
        print(f"   Unique traders: {self.merged_df['account'].nunique()}")
        print(f"   Unique symbols: {self.merged_df['symbol'].nunique()}")
        
        print(f"\n📈 PnL Analysis:")
        total_pnl = self.merged_df['closedPnL'].sum()
        avg_pnl = self.merged_df['closedPnL'].mean()
        profitable_trades = (self.merged_df['closedPnL'] > 0).sum()
        win_rate = profitable_trades / len(self.merged_df) * 100
        
        print(f"   Total PnL: ${total_pnl:,.2f}")
        print(f"   Average PnL: ${avg_pnl:.2f}")
        print(f"   Win Rate: {win_rate:.1f}% ({profitable_trades:,}/{len(self.merged_df):,})")
        
        print(f"\n🎭 Sentiment Distribution:")
        for sentiment in ['Fear', 'Neutral', 'Greed']:
            data = self.merged_df[self.merged_df['Classification'] == sentiment]
            if len(data) > 0:
                count = len(data)
                pct = count / len(self.merged_df) * 100
                avg_pnl = data['closedPnL'].mean()
                sentiment_win_rate = (data['closedPnL'] > 0).mean() * 100
                print(f"   {sentiment}: {count:,} trades ({pct:.1f}%), Avg PnL: ${avg_pnl:.2f}, Win Rate: {sentiment_win_rate:.1f}%")
        
        return {
            'total_records': len(self.merged_df),
            'total_pnl': total_pnl,
            'unique_traders': self.merged_df['account'].nunique(),
            'win_rate': win_rate
        }
    
    def behavioral_trader_segmentation(self):
        """Perform trader behavioral segmentation."""
        print("\n🤖 Behavioral Trader Segmentation")
        print("="*50)
        
        # Calculate trader profiles
        trader_profiles = self.merged_df.groupby('account').agg({
            'closedPnL': ['sum', 'mean', 'std', 'count'],
            'size': ['mean', 'std'],
            'leverage': ['mean', 'max'],
            'FearGreedIndex': 'mean',
            'Date': ['min', 'max']
        }).round(4)
        
        # Flatten columns
        trader_profiles.columns = [
            'total_pnl', 'avg_pnl', 'pnl_volatility', 'total_trades',
            'avg_size', 'size_volatility', 'avg_leverage', 'max_leverage',
            'avg_sentiment_exposure', 'first_trade', 'last_trade'
        ]
        
        # Additional metrics
        trader_profiles['trading_days'] = (trader_profiles['last_trade'] - trader_profiles['first_trade']).dt.days + 1
        trader_profiles['trades_per_day'] = trader_profiles['total_trades'] / trader_profiles['trading_days']
        trader_profiles['pnl_per_trade'] = trader_profiles['total_pnl'] / trader_profiles['total_trades']
        trader_profiles['profit_consistency'] = (trader_profiles['avg_pnl'] > 0).astype(int)
        
        # Clustering
        clustering_features = [
            'total_pnl', 'total_trades', 'avg_leverage', 'trades_per_day', 'pnl_per_trade'
        ]
        
        X_cluster = trader_profiles[clustering_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # Simple clustering
        n_clusters = min(5, max(3, len(trader_profiles) // 10))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        trader_profiles['trader_type'] = kmeans.fit_predict(X_scaled)
        
        # Classify trader types based on performance
        trader_type_names = {}
        for cluster_id in sorted(trader_profiles['trader_type'].unique()):
            cluster_data = trader_profiles[trader_profiles['trader_type'] == cluster_id]
            avg_pnl = cluster_data['total_pnl'].mean()
            profitability = cluster_data['profit_consistency'].mean()
            
            if avg_pnl > 10000 and profitability > 0.7:
                trader_type_names[cluster_id] = "Elite Performers"
            elif avg_pnl < -5000:
                trader_type_names[cluster_id] = "High-Risk Losers"  
            elif profitability >= 0.6:
                trader_type_names[cluster_id] = "Conservative Traders"
            else:
                trader_type_names[cluster_id] = "Average Traders"
        
        trader_profiles['trader_type_name'] = trader_profiles['trader_type'].map(trader_type_names)
        self.trader_profiles = trader_profiles
        
        print(f"✅ Segmented {len(trader_profiles)} traders into {n_clusters} types")
        
        for trader_type in trader_profiles['trader_type_name'].unique():
            type_data = trader_profiles[trader_profiles['trader_type_name'] == trader_type]
            count = len(type_data)
            avg_pnl = type_data['total_pnl'].mean()
            win_rate = type_data['profit_consistency'].mean() * 100
            print(f"   🏷️  {trader_type}: {count} traders, ${avg_pnl:.2f} avg PnL, {win_rate:.1f}% profitable")
        
        return trader_profiles
    
    def sentiment_performance_analysis(self):
        """Analyze sentiment-performance relationships."""
        print("\n📊 Sentiment-Performance Analysis")
        print("="*50)
        
        # Add trader types to merged data
        self.merged_df = self.merged_df.merge(
            self.trader_profiles[['trader_type_name']], 
            left_on='account', right_index=True, how='left'
        )
        
        print("📈 Performance by Market Sentiment:")
        sentiment_results = {}
        
        for sentiment in ['Fear', 'Neutral', 'Greed']:
            data = self.merged_df[self.merged_df['Classification'] == sentiment]
            if len(data) > 0:
                avg_pnl = data['closedPnL'].mean()
                win_rate = (data['closedPnL'] > 0).mean() * 100
                trade_count = len(data)
                total_pnl = data['closedPnL'].sum()
                
                sentiment_results[sentiment] = {
                    'avg_pnl': avg_pnl,
                    'win_rate': win_rate,
                    'trade_count': trade_count,
                    'total_pnl': total_pnl
                }
                
                print(f"   {sentiment}: ${avg_pnl:.2f} avg PnL, {win_rate:.1f}% win rate, {trade_count:,} trades")
        
        # Find best performing sentiment
        best_sentiment = max(sentiment_results.keys(), key=lambda x: sentiment_results[x]['avg_pnl'])
        print(f"\n🏆 Best performing sentiment: {best_sentiment} (${sentiment_results[best_sentiment]['avg_pnl']:.2f} avg PnL)")
        
        # Correlations
        correlations = {
            'FearGreed_PnL': self.merged_df[['FearGreedIndex', 'closedPnL']].corr().iloc[0,1],
            'FearGreed_Size': self.merged_df[['FearGreedIndex', 'size']].corr().iloc[0,1]
        }
        
        print(f"\n🔗 Key Correlations:")
        for metric, corr in correlations.items():
            direction = "📈 Positive" if corr > 0 else "📉 Negative"
            print(f"   {metric}: {corr:.4f} ({direction})")
        
        return {
            'sentiment_results': sentiment_results,
            'best_sentiment': best_sentiment,
            'correlations': correlations
        }
    
    def run_complete_analysis(self, fear_greed_file, trading_file):
        """Run the complete analysis pipeline."""
        print("🚀 Starting Complete Real Data Analysis")
        print("="*60)
        
        try:
            # Load data
            self.load_fear_greed_data(fear_greed_file)
            self.load_trading_data(trading_file)
            self.merge_datasets()
            
            # Analyze
            quality_metrics = self.validate_data_quality()
            trader_profiles = self.behavioral_trader_segmentation()
            sentiment_analysis = self.sentiment_performance_analysis()
            
            # Export results
            self.merged_df.to_csv('hyperliquid_analysis_results.csv', index=False)
            trader_profiles.to_csv('hyperliquid_trader_profiles.csv')
            
            print(f"\n🎉 ANALYSIS COMPLETE!")
            print(f"✅ Total Trades Analyzed: {len(self.merged_df):,}")
            print(f"✅ Total PnL: ${quality_metrics['total_pnl']:,.2f}")
            print(f"✅ Overall Win Rate: {quality_metrics['win_rate']:.1f}%")
            print(f"✅ Best Sentiment: {sentiment_analysis['best_sentiment']}")
            print(f"✅ Results exported to CSV files")
            
            return {
                'success': True,
                'total_trades': len(self.merged_df),
                'total_pnl': quality_metrics['total_pnl'],
                'unique_traders': quality_metrics['unique_traders'],
                'best_sentiment': sentiment_analysis['best_sentiment'],
                'sentiment_results': sentiment_analysis['sentiment_results']
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            raise


def main():
    """Main function."""
    print("🎯 Direct Mapping Real Data Analysis")
    print("="*40)
    
    analyzer = CryptoTradingAnalyzerDirect(random_state=42)
    
    # File paths
    fear_greed_file = "fear_greed_index.csv"
    trading_file = "hyperliquid_trading_data.csv"
    
    try:
        results = analyzer.run_complete_analysis(fear_greed_file, trading_file)
        
        if results['success']:
            print(f"\n🏆 SUCCESS! Key Insights from Your Real Data:")
            print(f"   📊 Total Trades: {results['total_trades']:,}")
            print(f"   💰 Total PnL: ${results['total_pnl']:,.2f}")
            print(f"   👥 Unique Traders: {results['unique_traders']}")
            print(f"   🎯 Best Market Sentiment: {results['best_sentiment']}")
            
            print(f"\n📈 Sentiment Performance Breakdown:")
            for sentiment, metrics in results['sentiment_results'].items():
                print(f"   {sentiment}: ${metrics['avg_pnl']:.2f} avg, {metrics['win_rate']:.1f}% win rate")
        
        return results
        
    except FileNotFoundError:
        print(f"\n❌ Files not found. Please ensure you have:")
        print(f"   - {fear_greed_file}")
        print(f"   - {trading_file}")
        print(f"   In the same directory as this script.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()