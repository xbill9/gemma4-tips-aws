# Root Makefile for managing all subdirectories

# List of subdirectories containing a Makefile
SUBDIRS := gpu-26B-6000-devops-agent \
           gpu-31B-6000-devops-agent \
           gpu-4B-6000-devops-agent \
           gpu-4B-L4-devops-agent \
           gpu-6000-devops-agent \
           gpu-vllm-devops-agent \
           local-devops-agent \
           tpu-26B-devops-agent \
           tpu-31B-devops-agent \
           tpu-12B-v6e1-devops-agent

.PHONY: all clean test lint install deploy help $(SUBDIRS)

# Default target displays help information
all: help

help:
	@echo "========================================================="
	@echo " Gemma-4 DevOps Agents - Root Makefile"
	@echo "========================================================="
	@echo "Available commands:"
	@echo "  make clean   - Run 'make clean' in all subdirectories"
	@echo "  make test    - Run 'make test' in all subdirectories"
	@echo "  make lint    - Run 'make lint' in all subdirectories"
	@echo "  make install - Run 'make install' in all subdirectories"
	@echo "  make deploy  - Run 'make deploy' in all subdirectories"
	@echo "  make help    - Show this help message"
	@echo "========================================================="

# Target-specific variable assignments
clean: TARGET := clean
clean: $(SUBDIRS)

test: TARGET := test
test: $(SUBDIRS)

lint: TARGET := lint
lint: $(SUBDIRS)

install: TARGET := install
install: $(SUBDIRS)

deploy: TARGET := deploy
deploy: $(SUBDIRS)

# Run the specified target in each subdirectory if a Makefile exists
$(SUBDIRS):
	@if [ -f $@/Makefile ]; then \
		if [ -z "$(TARGET)" ]; then \
			echo "⚙️ Executing default target in $@..."; \
			$(MAKE) -C $@; \
		else \
			echo "⚙️ Executing 'make $(TARGET)' in $@..."; \
			$(MAKE) -C $@ $(TARGET); \
		fi \
	fi
