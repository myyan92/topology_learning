import heapq
import pdb

class Tree(object):
    """ a min heap that also keeps search tree information from RRT.
    Each item in the heapq is a tuple of (priority, id, data),
    where id is the order that samples are generated, to break ties in priority.
    """

    def __init__(self):
        self.V_priority = []
        self.V_data = []
        self.V_count = 0
        self.E = []

    def add_root(self, root):
        # root is a tuple (priority, 0, data).
        self.V_priority.append(root[0])
        self.V_data.append(root[2])
        self.V_count += 1
        self.E.append(None)

    def add_leaf(self, parent, child, traj):
        # Parent and child are tuples (priority, id, data).
        self.V_priority.append(child[0])
        self.V_data.append(child[2])
        self.V_count += 1
        self.E.append( (parent[1], traj) )

    def reconstruct_path(self, leaf):
        # Reconstruct path from root to leaf
        waypoint_path = [leaf[2]]
        actions_path = []
        current = leaf[1]
        while self.E[current] is not None:
            actions_path.append(self.E[current][1])
            current = self.E[current][0]
            waypoint_path.append(self.V_data[current])
        waypoint_path.reverse()
        actions_path.reverse()
        return waypoint_path, actions_path


