from sklearn.ensemble import IsolationForest

def detect_anomalies_isolation_forest(data):
    clf = IsolationForest(contamination=0.1)
    clf.fit(data)
    outliers = clf.predict(data)
    anomalies = data[outliers == -1]
    return anomalies

# Example usage:
data = [[1], [2], [3], [10], [15], [100]]
anomalies = detect_anomalies_isolation_forest(data)
print("Anomalies:", anomalies)
