# dude p550dummy zero-delay benchmark blocker

The dude board was not reachable over the network, so no valid dummy benchmark score was produced.

- SSH alias: `dude`
- Host: `192.168.1.57`
- User: `eswin`
- Observed from WSL: `ssh dude` returned `No route to host`; `ping` had 100% packet loss; TCP port 22 was unreachable.
- Observed by worker agent from Windows: `Test-NetConnection 192.168.1.57 -Port 22` reported `DestinationHostUnreachable`.

This is a network/access blocker, not a Python dependency blocker. No benchmark cases were started on dude.
