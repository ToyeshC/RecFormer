import torch
from torch.utils.data import Dataset
from collections import defaultdict

class RecformerDataset(Dataset):
    def __init__(self, args, train, val, test, mode='train'):
        self.args = args
        self.mode = mode

        # These are lists of [user_id, item_id, 0]
        self.train_interactions = train
        self.val_interactions = val
        self.test_interactions = test

        # Build user-to-sequence dictionaries from the interaction lists
        self.user_to_train_seq = defaultdict(list)
        self.user_to_val_seq = defaultdict(list)
        self.user_to_test_seq = defaultdict(list)

        for user, item, _ in self.train_interactions:
            self.user_to_train_seq[user].append(item)
        for user, item, _ in self.val_interactions:
            self.user_to_val_seq[user].append(item)
        for user, item, _ in self.test_interactions:
            self.user_to_test_seq[user].append(item)

        if self.mode == 'train':
            # Training is done on a user-by-user basis
            self.users = [u for u in self.user_to_train_seq if u in self.user_to_val_seq]
        elif self.mode == 'val':
            self.users = [u for u in self.user_to_val_seq if u in self.user_to_train_seq]
        elif self.mode == 'test':
            # A user must have a training sequence to be in the test set
            self.users = [u for u in self.user_to_test_seq if u in self.user_to_train_seq]
        else:
            raise ValueError("mode must be one of train, val, test")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]

        if self.mode == 'train':
            seq = self.user_to_train_seq[user]
            labels = self.user_to_val_seq[user]
            return {'items': seq, 'labels': labels}

        elif self.mode == 'val':
            seq = self.user_to_train_seq[user]
            labels = self.user_to_val_seq[user]
            return {'items': seq, 'label': labels}

        elif self.mode == 'test':
            train_seq = self.user_to_train_seq.get(user, [])
            val_seq = self.user_to_val_seq.get(user, [])
            seq = train_seq + val_seq
            labels = self.user_to_test_seq[user]
            return {'items': seq, 'label': labels}
        
        else:
            raise ValueError("mode must be one of train, val, test") 