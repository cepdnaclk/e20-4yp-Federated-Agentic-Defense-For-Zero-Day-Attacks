"""
Port Configuration Validation Tests
Tests to ensure port configurations are consistent across the codebase
"""

import os
import re
import yaml
import json
from pathlib import Path

# Define expected configurations
EXPECTED_PORTS = {
    'fl_server': 9090,
    'agentic_ids': 5000,
}

REPO_ROOT = Path(__file__).parent


def test_docker_compose_ports():
    """Verify docker-compose.yml has correct port configurations"""
    compose_file = REPO_ROOT / "docker-compose.yml"
    
    with open(compose_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check FL Server port
    fl_server_env = config['services']['fl-server']['environment']
    assert any('PORT=9090' in env for env in fl_server_env), \
        f"FL Server should use port 9090, found: {fl_server_env}"
    
    # Check IDS agent ports
    for org in ['a', 'b', 'c']:
        agent_key = f'ids-agent-org-{org}'
        agent_env = config['services'][agent_key]['environment']
        
        assert any('PORT=5000' in env for env in agent_env), \
            f"IDS Agent {org} should use port 5000"
        
        assert any('FL_SERVER_URL=http://fl-server:9090' in env for env in agent_env), \
            f"IDS Agent {org} should connect to FL server at port 9090"
    
    # Check packet streamer URLs
    for org in ['a', 'b', 'c']:
        streamer_key = f'pkt-streamer-org-{org}'
        streamer_env = config['services'][streamer_key]['environment']
        
        expected_url = f'API_URL=http://ids-agent-org-{org}:5000/detect'
        assert any(expected_url in env for env in streamer_env), \
            f"Packet streamer {org} should connect to IDS agent at port 5000"
    
    print("✓ docker-compose.yml port configurations are correct")


def test_fl_server_code():
    """Verify FL server code uses correct default port"""
    app_file = REPO_ROOT / "fl-server" / "app.py"
    
    with open(app_file, 'r') as f:
        content = f.read()
    
    # Check for port configuration line
    port_pattern = r'port\s*=\s*int\(os\.environ\.get\("PORT",\s*(\d+)\)\)'
    match = re.search(port_pattern, content)
    
    assert match, "Could not find port configuration in fl-server/app.py"
    
    default_port = int(match.group(1))
    assert default_port == EXPECTED_PORTS['fl_server'], \
        f"FL Server default port should be {EXPECTED_PORTS['fl_server']}, found {default_port}"
    
    print("✓ FL Server code uses correct default port (9090)")


def test_agentic_ids_code():
    """Verify Agentic IDS code uses correct default port"""
    main_file = REPO_ROOT / "agentic-ids-local" / "src" / "main.py"
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Check for port configuration line
    port_pattern = r'PORT\s*=\s*int\(os\.getenv\("PORT",\s*(\d+)\)\)'
    match = re.search(port_pattern, content)
    
    assert match, "Could not find port configuration in agentic-ids-local/src/main.py"
    
    default_port = int(match.group(1))
    assert default_port == EXPECTED_PORTS['agentic_ids'], \
        f"Agentic IDS default port should be {EXPECTED_PORTS['agentic_ids']}, found {default_port}"
    
    print("✓ Agentic IDS code uses correct default port (5000)")


def test_orchestrator_fl_url():
    """Verify Orchestrator uses correct FL server URL"""
    orchestrator_file = REPO_ROOT / "agentic-ids-local" / "src" / "agents" / "Orchestrator" / "orchestrator.py"
    
    with open(orchestrator_file, 'r') as f:
        content = f.read()
    
    # Check for FL_SERVER_URL configuration
    url_pattern = r'FL_SERVER_URL",\s*"(http://[^"]+)"'
    match = re.search(url_pattern, content)
    
    assert match, "Could not find FL_SERVER_URL in orchestrator.py"
    
    default_url = match.group(1)
    assert 'fl-server:9090' in default_url, \
        f"Orchestrator should default to fl-server:9090, found {default_url}"
    
    print("✓ Orchestrator uses correct FL server URL (http://fl-server:9090)")


def test_pkt_streamer_config():
    """Verify packet streamer config.env has API_URL documented"""
    config_file = REPO_ROOT / "pkt-streamer" / "config.env"
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check that API_URL is present
    assert 'API_URL=' in content, "config.env should have API_URL defined"
    
    # Check that port 5000 is referenced
    assert '5000' in content, "config.env should reference port 5000"
    
    # Check for comment about Docker vs local
    has_comment = 'Docker' in content or 'localhost' in content
    assert has_comment, "config.env should have comments about Docker vs local configuration"
    
    print("✓ Packet streamer config.env is documented")


def test_port_consistency():
    """Verify all references to ports are consistent"""
    errors = []
    
    # Files to check for port references
    files_to_check = [
        REPO_ROOT / "docker-compose.yml",
        REPO_ROOT / "fl-server" / "app.py",
        REPO_ROOT / "agentic-ids-local" / "src" / "main.py",
        REPO_ROOT / "agentic-ids-local" / "src" / "agents" / "Orchestrator" / "orchestrator.py",
        REPO_ROOT / "PORT_CONFIGURATION.md",
    ]
    
    for file_path in files_to_check:
        if not file_path.exists():
            errors.append(f"Missing file: {file_path}")
            continue
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for FL server port mentions
        if 'fl-server' in str(file_path).lower() or 'orchestrator' in str(file_path).lower() or 'docker-compose' in str(file_path).lower():
            if '9090' not in content:
                errors.append(f"{file_path.name} should reference FL server port 9090")
        
        # Check for IDS agent port mentions
        if 'agentic-ids' in str(file_path).lower() or 'pkt-streamer' in str(file_path).lower() or 'docker-compose' in str(file_path).lower():
            if 'main.py' in str(file_path) or 'docker-compose' in str(file_path):
                if '5000' not in content:
                    errors.append(f"{file_path.name} should reference IDS agent port 5000")
    
    if errors:
        for error in errors:
            print(f"✗ {error}")
        raise AssertionError(f"Found {len(errors)} consistency issues")
    
    print("✓ Port references are consistent across files")


def run_all_tests():
    """Run all validation tests"""
    tests = [
        test_docker_compose_ports,
        test_fl_server_code,
        test_agentic_ids_code,
        test_orchestrator_fl_url,
        test_pkt_streamer_config,
        test_port_consistency,
    ]
    
    print("Running port configuration validation tests...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed > 0:
        exit(1)
    else:
        print("\n✓ All port configuration tests passed!")
        exit(0)


if __name__ == "__main__":
    run_all_tests()
