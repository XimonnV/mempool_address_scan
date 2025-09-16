# mempool_address_scan
To find bitcoin transactions in the mempool of your bitcoin node, in case you don't have the transaction ID but only a receiving or sending Bitcoin address and you want to use your own bitcoin node to check if it has a transaction for the address in its mempool.
For example if you withdraw from a CEX, you typically don't receive a transaction ID, but you do know your receiving address.

# how to use
```
./chmod +x mempool_addr_scan.py 
to test - look up a recent transaction, e.g. from https://mempool.space select the first upcoming block, select a transaction and copy a receiving or sending bitcoin address
./mempool_addr_scan.py --since 10m bc1pqte658tldm9r2pwm3d48t58q23dtpxrcq5wz25peyy2fdcdytnvq7tcxwv
```

# example output:
```
Scanning mempool for transactions related to address: bc1pqte658tldm9r2pwm3d48t58q23dtpxrcq5wz25peyy2fdcdytnvq7tcxwv
ScriptPubKey: 512002f3aa1d7f6eca3505db8b6a75d0e0545ab09878051c255039211496e1a45cd8
Time filter: last 600 seconds

Fetching mempool transaction IDs...
Found 570 transactions in the last 600 seconds
Decoding mempool transactions...
Decode mempool: 100% [##################################################]
Processing decoded transactions...
Process decoded: 100% [##################################################]
Fetching parent transactions...
Fetch parents: 100% [##################################################]
Processing parent transactions...
Process parents: 100% [##################################################]
Processing transactions...
Analyze txs: 100% [##################################################]
txid: 2e5b457f8b15955029ec1ec53476a1267cbdbe99121eedbe4357781e7a90b0c6
  fee: 0.00000508 BTC
  outgoing:
    amount sent from bc1pqte658tldm9r2pwm3d48t58q23dtpxrcq5wz25peyy2fdcdytnvq7tcxwv: 0.00024028 BTC
    receiving addresses:
      bc1plwyrnwdr4z5ktzqwew5c28j9382t7x0mw5xdvh7hkst4xvs66nrqksteg9: 0.00023520 BTC
```
