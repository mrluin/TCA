from abc import ABCMeta, abstractmethod


class BaseStorageBackend(metaclass=ABCMeta):
    """ Abstract class of storage backends

    All backends need to implement two apis get() and get_text()
    in which,
    get() reads the file as byte stream
    get_text() reads the file as texts
    """
    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class MemcachedBackend(BaseStorageBackend):
    """ Memcached storage backend
    Attributes:
        server_list_cfg (str): config file for memcached server list
        client_cfg (str): config file for memcached client
        sys_path(str|None): additional path to be appended to 'sys.path'
    """
    def __init__(self, server_list_cfg, client_cfg, sys_path=None):
        if sys_path is not None:
            import sys
            sys.path.append(sys_path)
        try:
            import mc
        except ImportError:
            raise ImportError('Please install memcached to enable MemcachedBackend')

        self.server_list_cfg = server_list_cfg
        self.client_cfg = client_cfg
        self._client = mc.MemcachedClient.GetInstacnce(self.server_list_cfg, self.client_cfg)
        self._mc_buffer = mc.pyvector()

    def get_text(self, filepath):
        raise NotImplementedError

    def get(self, filepath):
        filepath = str(filepath)
        import mc
        self._client.Get(filepath, self._mc_buffer)
        value_buf = mc.ConvertBuffer(self._mc_buffer)
        return value_buf


class HardDiskBackend(BaseStorageBackend):
    """ Raw hard disks storage backend """
    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'r') as f:
            value_buf = f.read()
        return value_buf


class LmdbBackend(BaseStorageBackend):
    """ lmdb storage backend

    Args:
        db_path (str|list[str]): lmdb database path for lq, gt, flow
        client_keys (str|list[str]): lmdb client key, e.g. lq, gt, flow
        readonly: lmdb environment parameter
        lock: lmdb environment parameter, if False, when concurrent access occurs, do not lock the database
        readahead: lmdb environment parameter, if false, disable the OS filesystem readahead mechanism, which may improve random read
        performance when a database is larger than RAM

    Attributes:
        db_paths: lmdb database path
        _client: a list of several lmdb envs
    """
    def __init__(self,
                 db_paths,
                 client_keys='default',
                 readonly=True,
                 lock=False,
                 readahead=False,
                 **kwargs):
        try:
            import lmdb
        except ImportError:
            raise ImportError('Please install lmdb to enable lmdbbackend')

        if isinstance(client_keys, str):
            client_keys = [client_keys]

        # change to [str]
        if isinstance(db_paths, list):
            self.db_paths = [str(v) for v in db_paths]
        elif isinstance(db_paths, str):
            self.db_paths = [str(db_paths)]

        assert len(client_keys) == len(self.db_paths), f'client_keys and db_paths should have the same length, ' \
                                                       f'but received {len(client_keys)} and {len(self.db_paths)}'

        self._client = {}
        for client, path in zip(client_keys, self.db_paths):
            self._client[client] = lmdb.open(path, readonly=readonly,
                                             lock=lock, readahead=readahead, **kwargs)

    def get(self, filepath, client_key):
        """ get value from related lmdb database according to client_key and filepath,
            filepath is key
        Args:
            filepath (str| obj:`Path`): filepath is the lmdb key
            client_key (str): used for distinguishing different lmdb envs
        """
        filepath = str(filepath)
        assert client_key in self._client, f"client_key {client_key} is not in lmdb clients"

        client = self._client[client_key]
        with client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode('ascii'))  # key filepath.encode('ascii')

        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class FileClient(object):
    """ general file client to access files in different backends

    The client loads a file or text in a specified backend from its path and return it as a binary file.
    it can also register other backend accessor with a given name and backend class

    Attributes:
        backend 'disk' 'memcached' 'lmdb'
        client (:obj:`BaseStorageBackend``)
    """
    _backends = {
        'disk': HardDiskBackend,
        'memcached': MemcachedBackend,
        'lmdb': LmdbBackend,
    }

    def __init__(self, backend='disk', **kwargs):
        if backend not in self._backends:
            raise ValueError(f"Backend {backend} is not supported. Currently supported ones are {list(self._backends.keys())}")

        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    def get(self, filepath, client_key='default'):
        # client_key is used only for lmdb
        if self.backend == 'lmdb':
            return self.client.get(filepath, client_key)
        else:
            return self.client.get(filepath)

    def get_text(self, filepath):
        return self.client.get_text(filepath)