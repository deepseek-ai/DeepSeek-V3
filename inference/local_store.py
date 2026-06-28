import os
import json
import uuid
import hashlib
import base64
import zlib
from typing import Optional, Dict, Any
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class LocalStore:
    """
    On-device persistent memory with cryptographic identity binding.
    
    Stores encrypted user identity and compressed conversation context.
    The only identifier transmitted to the backend is derived from the identity key.
    """
    
    def __init__(self, storage_dir: str, password: Optional[str] = None):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.key_file = os.path.join(storage_dir, "identity.key")
        self.salt_file = os.path.join(storage_dir, "salt.bin")
        self.context_file = os.path.join(storage_dir, "context.enc")
        
        # Generate or load identity key and encryption key
        self._identity_uuid: Optional[str] = None
        self._enc_key: Optional[bytes] = None
        
        if password:
            self._setup_with_password(password)
        else:
            self._setup_with_uuid()
        
        # Initialize AES-GCM cipher
        self._aesgcm = AESGCM(self._enc_key)
    
    def _setup_with_password(self, password: str):
        """Derive encryption key from password using PBKDF2."""
        salt = self._load_or_generate_salt()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        self._enc_key = kdf.derive(password.encode())
        # Identity UUID derived from password hash (deterministic)
        self._identity_uuid = hashlib.sha256(password.encode()).hexdigest()[:36]
    
    def _setup_with_uuid(self):
        """Generate a random UUIDv4 identity and derive encryption key."""
        if os.path.exists(self.key_file) and os.path.exists(self.salt_file):
            with open(self.key_file, "r") as f:
                self._identity_uuid = f.read().strip()
            salt = self._load_or_generate_salt()
            # Re-derive encryption key from identity and salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            self._enc_key = kdf.derive(self._identity_uuid.encode())
        else:
            # Generate fresh identity
            self._identity_uuid = str(uuid.uuid4())
            salt = os.urandom(16)
            with open(self.salt_file, "wb") as f:
                f.write(salt)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            self._enc_key = kdf.derive(self._identity_uuid.encode())
            with open(self.key_file, "w") as f:
                f.write(self._identity_uuid)
    
    def _load_or_generate_salt(self) -> bytes:
        """Load existing salt or generate a new one (16 bytes)."""
        if os.path.exists(self.salt_file):
            with open(self.salt_file, "rb") as f:
                return f.read()
        else:
            salt = os.urandom(16)
            with open(self.salt_file, "wb") as f:
                f.write(salt)
            return salt
    
    def get_identity_key(self) -> str:
        """Return the UUID-based identity key."""
        return self._identity_uuid or ""
    
    def save_context(self, context: Dict[str, Any]):
        """
        Encrypt and save context dictionary to local storage.
        Compresses the JSON data before encryption.
        """
        json_bytes = json.dumps(context).encode("utf-8")
        compressed = zlib.compress(json_bytes, level=6)
        nonce = os.urandom(12)
        ciphertext = self._aesgcm.encrypt(nonce, compressed, None)
        with open(self.context_file, "wb") as f:
            f.write(nonce + ciphertext)
    
    def load_context(self) -> Optional[Dict[str, Any]]:
        """
        Load and decrypt context from local storage.
        Returns None if no context exists or decryption fails.
        """
        if not os.path.exists(self.context_file):
            return None
        with open(self.context_file, "rb") as f:
            data = f.read()
        if len(data) < 12:
            return None
        nonce = data[:12]
        ciphertext = data[12:]
        try:
            compressed = self._aesgcm.decrypt(nonce, ciphertext, None)
            json_bytes = zlib.decompress(compressed)
            return json.loads(json_bytes.decode("utf-8"))
        except Exception:
            return None
    
    def clear_context(self):
        """Remove the stored context file."""
        if os.path.exists(self.context_file):
            os.remove(self.context_file)
    
    def clear_all(self):
        """Remove all stored data (identity, salt, context)."""
        for fpath in [self.key_file, self.salt_file, self.context_file]:
            if os.path.exists(fpath):
                os.remove(fpath)
