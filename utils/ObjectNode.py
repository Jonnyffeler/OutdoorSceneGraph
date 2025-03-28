class ObjectNode:
    def __init__(self, object, tree_id, height=0):
        self.object = object
        self.id = tree_id
        self.node_id = tree_id
        self.parent = None
        self.children = []
        self.height = height
        self.obj_size = len(self.object['pcd'])
    def add_child(self, child, voxel_size):
        import utils.utils_objects as utils_objects
        if child is self:
            assert(1==2)
        self.children.append(child)
        self.object = utils_objects.compute_fused_object([self.object, child.object], voxel_size)
        # update object of all parents
        parent = self.parent
        while parent is not None:
            parent.object = utils_objects.compute_fused_object([parent.object, child.object], voxel_size)
            parent = parent.parent
        # update child
        child.change_parent(self)
        child.change_tree_id(self.id)
        all_children = child.get_all_children()
        for child_change in all_children:
            child_change.change_tree_id(self.id)
    def remove_child(self, child):
        self.children.remove(child)
    def add_children_nodes_only(self, children):
        self.children += children
    def change_parent(self, parent):
        self.parent = parent
    def change_tree_id(self, id):
        self.id = id
    def change_node_id(self, id):
        self.node_id = id
    def get_root(self):
        root_node = self
        while root_node.parent is not None:
            root_node = root_node.parent
        return root_node
    def get_all_parents(self):
        parents = []
        node = self
        while node.parent is not None:
            parents.append(node.parent)
            node = node.parent
        return parents
    def get_all_children(self):
        all_children = []
        for child in self.children:
            all_children.append(child)
            all_children += child.get_all_children()
        return all_children