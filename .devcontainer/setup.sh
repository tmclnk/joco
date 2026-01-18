#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install -y tmux netcat-openbsd

# bd cli and claude beans plugin for slash commands
go install github.com/steveyegge/beads/cmd/bd@latest
claude plugin marketplace add steveyegge/beads
claude plugin install beads
