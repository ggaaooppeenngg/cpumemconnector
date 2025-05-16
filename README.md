# A CPU Memory based connector for KV Transfer

It is used to demostrate how to implment a external connector for KV Transfer.

This is adapted from the vLLM SharedStorageMemoryConnector.

The transfer is based on the simple cpu memory buffer, it is useful for small devices without too much GPU memory to hold the transfer buffer.