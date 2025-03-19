import multiprocessing
import pickle
import hashlib
import lmdb
import numpy as np

class QTable:
    def __init__(self, action_space, db_path='qtable.lmdb', map_size=1e9):
        self.action_space = action_space
        self.db_path = db_path
        self.map_size = int(map_size)
        self.lock = multiprocessing.Lock()
        self._init_db()

    def _init_db(self):
        # Initialize the LMDB environment and database with write_map and map_async
        self.env = lmdb.open(self.db_path, map_size=self.map_size, max_dbs=1, writemap=True, map_async=True, sync=False)
        self.db = self.env.open_db(b'qtable')

    def _get_state_key(self, state):
        # Convert state to a unique key using hashlib for faster serialization.
        return hashlib.sha256(pickle.dumps((state.board, state.mark))).digest()

    def get(self, state):
        key = self._get_state_key(state)
        with self.env.begin(db=self.db, write=False) as txn:
            data = txn.get(key)
            if data is None:
                return np.zeros(self.action_space.n)
            return pickle.loads(data)

    def _put(self, key, arr):
        with self.env.begin(db=self.db, write=True) as txn:
            txn.put(key, pickle.dumps(arr), overwrite=False)

    def update(self, state, action, value):
        key = self._get_state_key(state)
        with self.lock:
            with self.env.begin(db=self.db, write=True) as txn:
                data = txn.get(key)
                if data is None:
                    arr = np.zeros(self.action_space.n)
                    arr[action] = value
                    txn.put(key, pickle.dumps(arr))
                else:
                    arr = pickle.loads(data)
                    arr[action] = value
                    txn.put(key, pickle.dumps(arr))

    def __getstate__(self):
        # Exclude unpicklable objects
        state = self.__dict__.copy()
        if 'env' in state:
            del state['env']
        if 'db' in state:
            del state['db']
        if 'lock' in state:
            del state['lock']
        return state

    def __setstate__(self, state):
        # Restore instance attributes and reinitialize unpicklable ones.
        self.__dict__.update(state)
        self.lock = multiprocessing.Lock()
        self._init_db()
        
    def get_table(self):
        table = {}
        with self.env.begin(db=self.db, write=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                table[key] = pickle.loads(value)  # Use bytes directly
        return table
    
    def get_size(self):
        with self.lock:
            with self.env.begin(db=self.db) as txn:
                return txn.stat()['entries']