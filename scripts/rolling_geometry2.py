"""                                                                                                
     PRISM Rolling Geometry Analysis                                                                    
                                                                                                        
     Let DuckDB do the heavy lifting. Compute rolling correlations at scale,                            
     then track how cluster structure evolves over time.                                                
                                                                                                        
     Usage:                                                                                             
         python rolling_geometry.py --db data/prism.duckdb --window 252 --step 63                       
                                                                                                        
     Author: PRISM Project                                                                              
     Date: December 2024                                                                                
     """                                                                                                
                                                                                                        
     import duckdb                                                                                      
     import numpy as np                                                                                 
     import pandas as pd                                                                                
     from datetime import datetime, timedelta                                                           
     from dataclasses import dataclass                                                                  
     from typing import List, Dict, Optional, Tuple                                                     
     import argparse                                                                                    
     import sys                                                                                         
     import os                                                                                          
                                                                                                        
     sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                    
     from prism.agents.agent_emergent_clusters import EmergentClusterDetector, ClusteringResult         
                                                                                                        
                                                                                                        
     @dataclass                                                                                         
     class WindowResult:                                                                                
         """Result for a single time window."""                                                         
         window_start: datetime                                                                         
         window_end: datetime                                                                           
         n_indicators: int                                                                              
         n_clusters: int                                                                                
         cluster_members: Dict[int, List[str]]                                                          
         singletons: List[str]                                                                          
         confidence: float                                                                              
         mean_correlation: float                                                                        
         correlation_spread: float  # max - min correlation                                             
                                                                                                        
                                                                                                        
     @dataclass                                                                                         
     class GeometryEvolution:                                                                           
         """How geometry changes over time."""                                                          
         windows: List[WindowResult]                                                                    
                                                                                                        
         # Derived metrics                                                                              
         cluster_births: List[Tuple[datetime, List[str]]]   # When new clusters form                    
         cluster_deaths: List[Tuple[datetime, List[str]]]   # When clusters dissolve                    
         regime_changes: List[Tuple[datetime, str]]          # Major structural shifts                  
                                                                                                        
         def summary(self) -> str:                                                                      
             lines = [                                                                                  
                 "=" * 70,                                                                              
                 "ROLLING GEOMETRY EVOLUTION",                                                          
                 "=" * 70,                                                                              
                 "",                                                                                    
                 f"Windows analyzed: {len(self.windows)}",                                              
                 f"Time span: {self.windows[0].window_start.date()} to                                  
     {self.windows[-1].window_end.date()}",                                                             
                 "",                                                                                    
                 "CLUSTER COUNT OVER TIME:",                                                            
             ]                                                                                          
                                                                                                        
             # Show cluster count timeline                                                              
             for w in self.windows:                                                                     
                 bar = "█" * w.n_clusters + "░" * (5 - w.n_clusters)                                    
                 lines.append(f"  {w.window_end.date()}: {bar} {w.n_clusters} clusters                  
     (ρ={w.mean_correlation:.2f})")                                                                     
                                                                                                        
             lines.append("")                                                                           
             lines.append("REGIME CHANGES:")                                                            
             for dt, desc in self.regime_changes:                                                       
                 lines.append(f"  {dt.date()}: {desc}")                                                 
                                                                                                        
             if self.cluster_births:                                                                    
                 lines.append("")                                                                       
                 lines.append("CLUSTER FORMATIONS:")                                                    
                 for dt, members in self.cluster_births[:10]:                                           
                     lines.append(f"  {dt.date()}: {', '.join(members)}")                               
                                                                                                        
             return '\n'.join(lines)                                                                    
                                                                                                        
                                                                                                        
     def get_available_indicators(conn: duckdb.DuckDBPyConnection) -> List[str]:                        
         """Get all indicators with price data."""                                                      
         query = """                                                                                    
         SELECT DISTINCT indicator_id                                                                   
         FROM core.indicator_data                                                                       
         WHERE value IS NOT NULL                                                                        
         ORDER BY indicator_id                                                                          
         """                                                                                            
         return conn.execute(query).fetchdf()['indicator_id'].tolist()                                  
                                                                                                        
                                                                                                        
     def get_date_range(conn: duckdb.DuckDBPyConnection, indicators: List[str]) -> Tuple[datetime,      
     datetime]:                                                                                         
         """Get common date range for indicators."""                                                    
         ind_list = ','.join(f"'{i}'" for i in indicators)                                              
         query = f"""                                                                                   
         SELECT                                                                                         
             MAX(min_date) as start_date,                                                               
             MIN(max_date) as end_date                                                                  
         FROM (                                                                                         
             SELECT                                                                                     
                 indicator_id,                                                                          
                 MIN(date) as min_date,                                                                 
                 MAX(date) as max_date                                                                  
             FROM core.indicator_data                                                                   
             WHERE indicator_id IN ({ind_list})                                                         
               AND value IS NOT NULL                                                                    
             GROUP BY indicator_id                                                                      
         )                                                                                              
         """                                                                                            
         result = conn.execute(query).fetchone()                                                        
         return result[0], result[1]                                                                    
                                                                                                        
                                                                                                        
     def compute_rolling_correlations_duckdb(                                                           
         conn: duckdb.DuckDBPyConnection,                                                               
         indicators: List[str],                                                                         
         window_start: datetime,                                                                        
         window_end: datetime                                                                           
     ) -> np.ndarray:                                                                                   
         """                                                                                            
         Compute correlation matrix for a window using DuckDB.                                          
         Returns n x n numpy array.                                                                     
         """                                                                                            
         ind_list = ','.join(f"'{i}'" for i in indicators)                                              
         n = len(indicators)                                                                            
                                                                                                        
         # Get returns for this window                                                                  
         query = f"""                                                                                   
         WITH prices AS (                                                                               
             SELECT                                                                                     
                 indicator_id,                                                                          
                 date,                                                                                  
                 value,                                                                                 
                 LAG(value) OVER (PARTITION BY indicator_id ORDER BY date) as prev_value                
             FROM core.indicator_data                                                                   
             WHERE indicator_id IN ({ind_list})                                                         
               AND date BETWEEN '{window_start.date()}' AND '{window_end.date()}'                       
               AND value IS NOT NULL                                                                    
         ),                                                                                             
         returns AS (                                                                                   
             SELECT                                                                                     
                 indicator_id,                                                                          
                 date,                                                                                  
                 (value - prev_value) / NULLIF(prev_value, 0) as ret                                    
             FROM prices                                                                                
             WHERE prev_value IS NOT NULL                                                               
         )                                                                                              
         SELECT indicator_id, date, ret                                                                 
         FROM returns                                                                                   
         ORDER BY date, indicator_id                                                                    
         """                                                                                            
                                                                                                        
         df = conn.execute(query).fetchdf()                                                             
                                                                                                        
         if df.empty:                                                                                   
             return np.eye(n)                                                                           
                                                                                                        
         # Pivot to wide format                                                                         
         pivot = df.pivot(index='date', columns='indicator_id', values='ret')                           
                                                                                                        
         # Only keep indicators we have data for                                                        
         available = [i for i in indicators if i in pivot.columns]                                      
         if len(available) < 2:                                                                         
             return np.eye(n)                                                                           
                                                                                                        
         # Compute correlation                                                                          
         corr = pivot[available].corr()                                                                 
                                                                                                        
         # Build full matrix (with 1s for missing)                                                      
         result = np.eye(n)                                                                             
         id_to_idx = {ind: i for i, ind in enumerate(indicators)}                                       
                                                                                                        
         for i, ind1 in enumerate(available):                                                           
             for j, ind2 in enumerate(available):                                                       
                 idx1 = id_to_idx[ind1]                                                                 
                 idx2 = id_to_idx[ind2]                                                                 
                 result[idx1, idx2] = corr.loc[ind1, ind2]                                              
                                                                                                        
         return result                                                                                  
                                                                                                        
                                                                                                        
     def analyze_rolling_windows(                                                                       
         conn: duckdb.DuckDBPyConnection,                                                               
         indicators: List[str],                                                                         
         window_days: int = 252,                                                                        
         step_days: int = 63,                                                                           
         start_date: datetime = None,                                                                   
         end_date: datetime = None                                                                      
     ) -> GeometryEvolution:                                                                            
         """                                                                                            
         Run rolling window analysis.                                                                   
                                                                                                        
         Args:                                                                                          
             conn: DuckDB connection                                                                    
             indicators: List of indicator IDs                                                          
             window_days: Window size in trading days (252 = 1 year)                                    
             step_days: Step size in trading days (63 = 1 quarter)                                      
             start_date: Analysis start (default: earliest common date)                                 
             end_date: Analysis end (default: latest common date)                                       
         """                                                                                            
         # Get date range                                                                               
         data_start, data_end = get_date_range(conn, indicators)                                        
                                                                                                        
         if start_date is None:                                                                         
             start_date = data_start                                                                    
         if end_date is None:                                                                           
             end_date = data_end                                                                        
                                                                                                        
         print(f"Analyzing {len(indicators)} indicators from {start_date.date()} to                     
     {end_date.date()}")                                                                                
         print(f"Window: {window_days} days, Step: {step_days} days")                                   
         print()                                                                                        
                                                                                                        
         # Initialize detector                                                                          
         detector = EmergentClusterDetector()                                                           
                                                                                                        
         # Generate window dates                                                                        
         windows = []                                                                                   
         current_end = start_date + timedelta(days=window_days)                                         
                                                                                                        
         while current_end <= end_date:                                                                 
             window_start = current_end - timedelta(days=window_days)                                   
                                                                                                        
             # Compute correlations for this window                                                     
             corr_matrix = compute_rolling_correlations_duckdb(                                         
                 conn, indicators, window_start, current_end                                            
             )                                                                                          
                                                                                                        
             # Use absolute correlation as similarity                                                   
             similarity = np.abs(corr_matrix)                                                           
                                                                                                        
             # Detect clusters                                                                          
             result = detector.detect_clusters(similarity, indicators)                                  
                                                                                                        
             # Extract cluster members                                                                  
             cluster_members = {}                                                                       
             for cluster in result.clusters:                                                            
                 cluster_members[cluster.cluster_id] = cluster.members                                  
                                                                                                        
             # Compute correlation stats                                                                
             upper_tri = corr_matrix[np.triu_indices(len(indicators), k=1)]                             
             mean_corr = np.mean(upper_tri) if len(upper_tri) > 0 else 0                                
             corr_spread = np.max(upper_tri) - np.min(upper_tri) if len(upper_tri) > 0 else 0           
                                                                                                        
             window_result = WindowResult(                                                              
                 window_start=window_start,                                                             
                 window_end=current_end,                                                                
                 n_indicators=len(indicators),                                                          
                 n_clusters=result.n_clusters,                                                          
                 cluster_members=cluster_members,                                                       
                 singletons=result.singletons,                                                          
                 confidence=result.confidence,                                                          
                 mean_correlation=mean_corr,                                                            
                 correlation_spread=corr_spread                                                         
             )                                                                                          
                                                                                                        
             windows.append(window_result)                                                              
             print(f"  {current_end.date()}: {result.n_clusters} clusters, ρ={mean_corr:.3f},           
     conf={result.confidence:.1%}")                                                                     
                                                                                                        
             current_end += timedelta(days=step_days)                                                   
                                                                                                        
         # Detect regime changes and cluster births/deaths                                              
         cluster_births = []                                                                            
         cluster_deaths = []                                                                            
         regime_changes = []                                                                            
                                                                                                        
         prev_clusters = set()                                                                          
         prev_n = 0                                                                                     
                                                                                                        
         for w in windows:                                                                              
             current_clusters = set()                                                                   
             for members in w.cluster_members.values():                                                 
                 current_clusters.add(frozenset(members))                                               
                                                                                                        
             # New clusters                                                                             
             for cluster in current_clusters - prev_clusters:                                           
                 if len(cluster) > 1:                                                                   
                     cluster_births.append((w.window_end, list(cluster)))                               
                                                                                                        
             # Dead clusters                                                                            
             for cluster in prev_clusters - current_clusters:                                           
                 if len(cluster) > 1:                                                                   
                     cluster_deaths.append((w.window_end, list(cluster)))                               
                                                                                                        
             # Regime changes (significant cluster count change)                                        
             if abs(w.n_clusters - prev_n) >= 2:                                                        
                 direction = "expansion" if w.n_clusters > prev_n else "collapse"                       
                 regime_changes.append((                                                                
                     w.window_end,                                                                      
                     f"Cluster {direction}: {prev_n} → {w.n_clusters}"                                  
                 ))                                                                                     
                                                                                                        
             # Correlation regime changes                                                               
             if len(windows) > 1:                                                                       
                 idx = windows.index(w)                                                                 
                 if idx > 0:                                                                            
                     prev_corr = windows[idx-1].mean_correlation                                        
                     if abs(w.mean_correlation - prev_corr) > 0.15:                                     
                         direction = "coupling" if w.mean_correlation > prev_corr else "decoupling"     
                         regime_changes.append((                                                        
                             w.window_end,                                                              
                             f"Correlation {direction}: {prev_corr:.2f} → {w.mean_correlation:.2f}"     
                         ))                                                                             
                                                                                                        
             prev_clusters = current_clusters                                                           
             prev_n = w.n_clusters                                                                      
                                                                                                        
         return GeometryEvolution(                                                                      
             windows=windows,                                                                           
             cluster_births=cluster_births,                                                             
             cluster_deaths=cluster_deaths,                                                             
             regime_changes=regime_changes                                                              
         )                                                                                              
                                                                                                        
                                                                                                        
     def main():                                                                                        
         parser = argparse.ArgumentParser(description="PRISM Rolling Geometry Analysis")                
         parser.add_argument('--db', required=True, help='Path to DuckDB database')                     
         parser.add_argument('--indicators', nargs='+', help='Indicators to analyze (default: all)')    
         parser.add_argument('--window', type=int, default=252, help='Window size in days (default:     
     252)')                                                                                             
         parser.add_argument('--step', type=int, default=63, help='Step size in days (default: 63)')    
         parser.add_argument('--start', help='Start date (YYYY-MM-DD)')                                 
         parser.add_argument('--end', help='End date (YYYY-MM-DD)')                                     
         parser.add_argument('--output', help='Output CSV file for results')                            
                                                                                                        
         args = parser.parse_args()                                                                     
                                                                                                        
         # Connect                                                                                      
         conn = duckdb.connect(args.db)                                                                 
                                                                                                        
         # Get indicators                                                                               
         if args.indicators:                                                                            
             indicators = args.indicators                                                               
         else:                                                                                          
             indicators = get_available_indicators(conn)                                                
             print(f"Found {len(indicators)} indicators: {', '.join(indicators[:10])}...")              
                                                                                                        
         # Parse dates                                                                                  
         start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else None                 
         end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else None                       
                                                                                                        
         # Run analysis                                                                                 
         evolution = analyze_rolling_windows(                                                           
             conn=conn,                                                                                 
             indicators=indicators,                                                                     
             window_days=args.window,                                                                   
             step_days=args.step,                                                                       
             start_date=start_date,                                                                     
             end_date=end_date                                                                          
         )                                                                                              
                                                                                                        
         print()                                                                                        
         print(evolution.summary())                                                                     
                                                                                                        
         # Save to CSV if requested                                                                     
         if args.output:                                                                                
             rows = []                                                                                  
             for w in evolution.windows:                                                                
                 rows.append({                                                                          
                     'window_end': w.window_end.date(),                                                 
                     'n_clusters': w.n_clusters,                                                        
                     'n_singletons': len(w.singletons),                                                 
                     'confidence': w.confidence,                                                        
                     'mean_correlation': w.mean_correlation,                                            
                     'correlation_spread': w.correlation_spread,                                        
                     'cluster_0': ','.join(w.cluster_members.get(0, [])),                               
                     'cluster_1': ','.join(w.cluster_members.get(1, [])),                               
                     'cluster_2': ','.join(w.cluster_members.get(2, [])),                               
                     'singletons': ','.join(w.singletons)                                               
                 })                                                                                     
                                                                                                        
             df = pd.DataFrame(rows)                                                                    
             df.to_csv(args.output, index=False)                                                        
             print(f"\nResults saved to {args.output}")                                                 
                                                                                                        
         conn.close()                                                                                   
                                                                                                        
                                                                                                        
     if __name__ == "__main__":                                                                         
         main()       