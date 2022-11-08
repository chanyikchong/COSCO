class Workload:
    def __init__(self):
        self.creation_id = 0
        self.created_containers = list()
        self.deployed_containers = list()

    def get_undeployed_containers(self):
        undeployed = []
        for i, deployed in enumerate(self.deployed_containers):
            if not deployed:
                undeployed.append(self.created_containers[i])
        return undeployed

    def update_deployed_containers(self, creation_ids):
        for cid in creation_ids:
            assert not self.deployed_containers[cid]
            self.deployed_containers[cid] = True
