# dude blocker

Non-interactive SSH failed with `Permission denied (publickey,password)`.

Install the WSL public key manually:

```bash
ssh-copy-id -i /home/zyz/.ssh/id_ed25519.pub dude
```

Then verify:

```bash
ssh -o BatchMode=yes -o PreferredAuthentications=publickey -o PasswordAuthentication=no dude 'whoami && hostname'
```
