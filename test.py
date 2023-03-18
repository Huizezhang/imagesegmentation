import math


class PixelCluster:
    def __init__(self, max_distance=50):
        self.max_distance = max_distance
        self.clusters = []
        self.points = []

    def add_point(self, point):
        self.points.append(point)
        clusters = []
        for p in self.points:
            if not clusters:
                clusters.append([p])
            else:
                added = False
                for i in range(len(clusters)):
                    if self._distance(p, clusters[i][0]) <= self.max_distance:
                        clusters[i].append(p)
                        added = True
                        break
                if not added:
                    clusters.append([p])
        self.clusters = clusters

    def get_clusters(self):
        return self.clusters

    def _distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

pixel_cluster = PixelCluster()
pixel_cluster.add_point((10, 10))
pixel_cluster.add_point((20, 20))
pixel_cluster.add_point((100, 100))
pixel_cluster.add_point((25, 25))
pixel_cluster.add_point((30, 30))
pixel_cluster.add_point((60, 60))

clusters = pixel_cluster.get_clusters()
print(clusters)