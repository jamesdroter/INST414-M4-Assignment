import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

baseball_df = pd.read_csv("savant_stats.csv")
baseball_df = baseball_df.dropna()  # Drop missing values

#Select features for clustering
features = ["exit_velocity_avg", "launch_angle_avg", "hard_hit_percent", "on_base_plus_slg", "p_era"]
pitching_data = baseball_df[features]

#Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pitching_data)

#Determine the optimal number of clusters using the Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

#Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.show()

#Based on the elbow, select k=4 and fit the KMeans model
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
baseball_df['Cluster'] = clusters


#Scatterplot of Exit Velocity vs OPS with cluster color
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=baseball_df["exit_velocity_avg"], 
    y=baseball_df["on_base_plus_slg"], 
    hue=baseball_df["Cluster"], 
    palette="viridis", 
    s=100
)
plt.title("Exit Velocity vs OPS by Cluster")
plt.xlabel("Exit Velocity Average")
plt.ylabel("On-Base Plus Slugging Allowed (OPS)")
plt.legend(title="Cluster")
plt.show()

#Radar chart for cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_summary = pd.DataFrame(cluster_centers, columns=features)

#Radar Chart Visualization
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, row in cluster_summary.iterrows():
    data = row.tolist() + row.tolist()[:1]
    ax.plot(angles, data, label=f"Cluster {i}")
    ax.fill(angles, data, alpha=0.25)
ax.set_title("Cluster Profiles")
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=12)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.show()

sil_score = silhouette_score(scaled_data, clusters)
print(f"Silhouette Score for k={optimal_k}: {sil_score}")

baseball_df.to_csv("clustered_pitchers.csv", index=False)
