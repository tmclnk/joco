.PHONY: up bash help

help: ## this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

up: ## start a devcontainer
	 devcontainer --workspace-folder . up

zsh: ## launch shell inside devcontainer
	devcontainer exec --workspace-folder . zsh
