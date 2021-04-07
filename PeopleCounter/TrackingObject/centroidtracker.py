import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
class CentroidTracker:
    def __init__(self, maxDistance, maxDisappear):
        self.maxDistance = maxDistance
        self.maxDisappear = maxDisappear
        self.objects = OrderedDict()
        self.disappear = OrderedDict()
        self.objectID = 0
    def register(self, centroid):
        self.objectID += 1
        self.objects[self.objectID] = centroid
        self.disappear[self.objectID] = 0
    def degister(self, id):
        del self.objects[id]
        del self.disappear[id]
    def update(self, rects):
        if len(rects) == 0:
            for i in list(self.objects.keys()):
                self.disappear[i] += 1
                if self.disappear[i] > self.maxDisappear:
                    self.degister(i)
            return self.objects
        else:
            new_centroids = []
            for rect in rects:
                centroid_X = int((rect[0] + rect[2]) /2)
                centroid_Y = int((rect[1] + rect[3]) /2)
                new_centroids.append([centroid_X, centroid_Y])
            if len(self.objects) == 0:
                    for centroid in new_centroids:
                        self.register(centroid)
            else:
                object_ids = list(self.objects.keys())
                old_centroids = list(self.objects.values())

                D = dist.cdist(np.array(new_centroids), np.array(old_centroids))

                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                usedRows = []
                usedCols = []

                for (row, col) in zip(rows, cols):
                    if row in usedRows or col in usedCols:
                        continue

                    if D[row, col] > self.maxDistance:
                        continue
                    id = object_ids[col]
                    self.objects[id] = new_centroids[row]
                    self.disappear[id] = 0

                    usedRows.append(row)
                    usedCols.append(col)
                unusedRow = [r for r in rows if r not in usedRows]
                unusedCol = [c for c in cols if c not in usedCols]

                for col in unusedCol:
                    id = object_ids[col]
                    self.disappear[id] += 1
                    if self.disappear[id] > self.maxDisappear:
                        self.degister(id)

                for row in unusedRow:
                    self.register(new_centroids[row])
            return self.objects